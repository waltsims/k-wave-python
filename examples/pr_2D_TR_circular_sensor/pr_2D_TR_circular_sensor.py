import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction import TimeReversal
from kwave.utils.colormap import get_color_map
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_cart_circle

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 2D Time Reversal Reconstruction For A Circular Sensor Example
#
# This example demonstrates the use of k-Wave for the time-reversal
# reconstruction of a two-dimensional photoacoustic wave-field recorded
# over a circular array of sensor elements. The sensor data is simulated
# and then time-reversed using kspaceFirstOrder2D. It builds on the 2D Time
# Reversal Reconstruction For A Line Sensor Example.


def main():
    logging.info("Starting 2D Time Reversal Reconstruction for Circular Sensor example")

    # --------------------
    # SIMULATION
    # --------------------

    # load the initial pressure distribution from an image and scale
    logging.info("Loading initial pressure distribution from image")
    p0_magnitude = 2
    source_image_path = "./examples/EXAMPLE_source_two.bmp"
    source_image = np.array(Image.open(source_image_path))
    p0 = p0_magnitude * source_image
    logging.info(f"Loaded image with shape: {source_image.shape}")

    # assign the grid size and create the computational grid
    logging.info("Setting up computational grid")
    PML_size = 20  # size of the PML in grid points
    Nx = 256 - 2 * PML_size  # number of grid points in the x direction
    Ny = 256 - 2 * PML_size  # number of grid points in the y direction
    x = 10e-3  # total grid size [m]
    y = 10e-3  # total grid size [m]
    dx = x / Nx  # grid point spacing in the x direction [m]
    dy = y / Ny  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))
    logging.info(f"Created grid with dimensions: {Nx}x{Ny}")

    # resize the input image to the desired number of grid points
    logging.info("Resizing input image to match grid dimensions")
    from scipy.ndimage import zoom

    zoom_factors = (Nx / p0.shape[0], Ny / p0.shape[1])
    p0 = zoom(p0, zoom_factors, order=1)
    logging.info(f"Resized image to shape: {p0.shape}")

    # smooth the initial pressure distribution and restore the magnitude
    logging.info("Smoothing initial pressure distribution")
    p0 = smooth(p0, restore_max=True)

    # assign to the source structure
    logging.info("Setting up source structure")
    source = kSource()
    source.p0 = p0

    # define the properties of the propagation medium
    logging.info("Setting up propagation medium")
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # define a centered Cartesian circular sensor
    logging.info("Setting up circular sensor")
    sensor_radius = 4.5e-3  # [m]
    sensor_angle = 3 * np.pi / 2  # [rad]
    sensor_pos = Vector([0, 0])  # [m]
    num_sensor_points = 70
    cart_sensor_mask = make_cart_circle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle)
    logging.info(f"Created circular sensor with {num_sensor_points} points")

    # assign to sensor structure
    sensor = kSensor()
    sensor.mask = cart_sensor_mask
    sensor.record = ["p", "p_final"]

    # create the time array
    logging.info("Creating time array")
    kgrid.makeTime(medium.sound_speed)
    logging.info(f"Time array created with {kgrid.Nt} time steps")

    # set the input options
    logging.info("Setting simulation options")
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,
        save_to_disk=True,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    # run the simulation
    logging.info("Running forward simulation")
    sensor_data = kspace_first_order_2d_gpu(kgrid, source, deepcopy(sensor), medium, simulation_options, execution_options)
    sensor.recorded_pressure = sensor_data["p"].T  # Store the recorded pressure data
    logging.info("Forward simulation completed")

    # add noise to the recorded sensor data
    logging.info("Adding noise to sensor data")
    signal_to_noise_ratio = 40  # [dB]
    from kwave.utils.signals import add_noise

    sensor.recorded_pressure = add_noise(sensor.recorded_pressure, signal_to_noise_ratio, "peak")
    logging.info(f"Added noise with SNR: {signal_to_noise_ratio} dB")

    # create a second computation grid for the reconstruction
    logging.info("Setting up reconstruction grid")
    Nx_recon = 300  # number of grid points in the x direction
    Ny_recon = 300  # number of grid points in the y direction
    dx_recon = x / Nx_recon  # grid point spacing in the x direction [m]
    dy_recon = y / Ny_recon  # grid point spacing in the y direction [m]
    kgrid_recon = kWaveGrid(Vector([Nx_recon, Ny_recon]), Vector([dx_recon, dy_recon]))
    logging.info(f"Created reconstruction grid with dimensions: {Nx_recon}x{Ny_recon}")

    # use the same time array for the reconstruction
    kgrid_recon.setTime(kgrid.Nt, kgrid.dt)

    # reset the initial pressure
    source = kSource()

    # create time reversal handler and run reconstruction
    logging.info("Starting time reversal reconstruction")
    tr = TimeReversal(kgrid_recon, medium, sensor)
    p0_recon = tr(kspace_first_order_2d_gpu, simulation_options, execution_options)
    logging.info("Initial reconstruction completed")

    # create a binary sensor mask of an equivalent continuous circle
    logging.info("Creating binary sensor mask")
    sensor_radius_grid_points = round(sensor_radius / kgrid_recon.dx)
    from kwave.utils.mapgen import make_circle

    binary_sensor_mask = make_circle(
        grid_size=Vector([kgrid_recon.Nx, kgrid_recon.Ny]),
        center=Vector([kgrid_recon.Nx // 2 + 1, kgrid_recon.Ny // 2 + 1]),
        radius=sensor_radius_grid_points,
        arc_angle=sensor_angle,
    )

    # assign to sensor structure
    sensor = kSensor()
    sensor.mask = binary_sensor_mask
    sensor.record = ["p", "p_final"]

    # interpolate data to remove the gaps and assign to sensor structure
    logging.info("Interpolating sensor data")
    from kwave.utils.interp import interp_cart_data

    sensor.recorded_pressure = interp_cart_data(kgrid_recon, sensor_data["p"].T, cart_sensor_mask, binary_sensor_mask)

    # run the time-reversal reconstruction
    logging.info("Starting interpolated reconstruction")
    tr = TimeReversal(kgrid_recon, medium, sensor)
    p0_recon_interp = tr(kspace_first_order_2d_gpu, simulation_options, execution_options)
    logging.info("Interpolated reconstruction completed")

    # --------------------
    # VISUALIZATION
    # --------------------
    logging.info("Starting visualization")

    cmap = get_color_map()

    # plot the initial pressure and sensor distribution
    logging.info("Plotting initial pressure and sensor distribution")
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p0,
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=-1,
        vmax=1,
        cmap=cmap,
    )
    # Plot sensor points separately
    ax.scatter(cart_sensor_mask[1, :] * 1e3, cart_sensor_mask[0, :] * 1e3, color="red", s=10, marker="o")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    fig.colorbar(im, cax=cax)
    plt.title("Initial Pressure and Sensor Distribution")
    plt.savefig("initial_pressure_and_sensor_distribution.png")

    # plot the simulated sensor data
    logging.info("Plotting simulated sensor data")
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        sensor_data["p"].T,
        vmin=-1,
        vmax=1,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("Sensor Position")
    ax.set_xlabel("Time Step")
    fig.colorbar(im, cax=cax)
    plt.title("Simulated Sensor Data")
    plt.savefig("simulated_sensor_data.png")

    # plot the reconstructed initial pressure
    logging.info("Plotting reconstructed initial pressure")
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p0_recon,
        extent=[kgrid_recon.y_vec.min() * 1e3, kgrid_recon.y_vec.max() * 1e3, kgrid_recon.x_vec.max() * 1e3, kgrid_recon.x_vec.min() * 1e3],
        vmin=-1,
        vmax=1,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    fig.colorbar(im, cax=cax)
    plt.title("Reconstructed Initial Pressure")
    plt.savefig("reconstructed_initial_pressure.png")

    # plot the reconstructed initial pressure using the interpolated data
    logging.info("Plotting interpolated reconstruction")
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p0_recon_interp,
        extent=[kgrid_recon.y_vec.min() * 1e3, kgrid_recon.y_vec.max() * 1e3, kgrid_recon.x_vec.max() * 1e3, kgrid_recon.x_vec.min() * 1e3],
        vmin=-1,
        vmax=1,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    fig.colorbar(im, cax=cax)
    plt.title("Reconstructed Initial Pressure (Interpolated)")
    plt.savefig("reconstructed_initial_pressure_interpolated.png")
    # plot a profile for comparison
    logging.info("Plotting pressure profile comparison")
    slice_pos = 4.5e-3  # [m] location of the slice from top of grid [m]
    plt.figure()
    plt.plot(kgrid.y_vec[:, 0] * 1e3, p0[round(slice_pos / kgrid.dx), :], "k--", label="Initial Pressure")
    plt.plot(kgrid_recon.y_vec[:, 0] * 1e3, p0_recon[round(slice_pos / kgrid_recon.dx), :], "r-", label="Point Reconstruction")
    plt.plot(
        kgrid_recon.y_vec[:, 0] * 1e3, p0_recon_interp[round(slice_pos / kgrid_recon.dx), :], "b-", label="Interpolated Reconstruction"
    )
    plt.xlabel("y-position [mm]")
    plt.ylabel("Pressure")
    plt.legend()
    plt.axis("tight")
    plt.ylim([0, 2.1])
    plt.title("Pressure Profile Comparison")
    plt.savefig("pressure_profile_comparison.png")

    logging.info("Example completed successfully")


if __name__ == "__main__":
    main()
