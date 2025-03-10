import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.kspaceLineRecon import kspaceLineRecon
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.colormap import get_color_map
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc

# 2D Time Reversal Reconstruction For A Line Sensor Example

# This example demonstrates the use of k-Wave for the time-reversal
# reconstruction of a two-dimensional photoacoustic wave-field recorded
# over a linear array of sensor elements. The sensor data is simulated and
# then time-reversed using kspaceFirstOrder2D. It builds on the 2D FFT
# Reconstruction For A Line Sensor Example.


def main():
    # --------------------
    # SIMULATION
    # --------------------

    # create the computational grid
    PML_size = 20  # size of the PML in grid points
    N = Vector([128, 256]) - 2 * PML_size  # number of grid points
    d = Vector([0.1e-3, 0.1e-3])  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
    )

    # create initial pressure distribution using makeDisc
    disc_magnitude = 5  # [Pa]
    disc_pos = Vector([60, 140])  # [grid points]
    disc_radius = 5  # [grid points]
    disc_2 = disc_magnitude * make_disc(N, disc_pos, disc_radius)

    disc_pos = Vector([30, 110])  # [grid points]
    disc_radius = 8  # [grid points]
    disc_1 = disc_magnitude * make_disc(N, disc_pos, disc_radius)

    # smooth the initial pressure distribution and restore the magnitude
    p0 = smooth(disc_1 + disc_2, restore_max=True)

    # assign to the source structure
    source = kSource()
    source.p0 = p0

    # define a binary line sensor
    sensor = kSensor()
    sensor.mask = np.zeros(N)
    sensor.mask[0] = 1

    # set the input arguements: force the PML to be outside the computational
    # grid; switch off p0 smoothing within kspaceFirstOrder2D
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,
        save_to_disk=True,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    # run the simulation
    sensor_data = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor_data = sensor_data["p"].T

    # reset the initial pressure and sensor
    source = kSource()
    sensor = kSensor()
    sensor.mask = np.zeros(N)
    sensor.mask[0] = 1

    # assign the time reversal data
    sensor.time_reversal_boundary_data = sensor_data

    # run the time reversal reconstruction
    p0_recon = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)
    p0_recon = p0_recon["p_final"].T

    # add first order compensation for only recording over a half plane
    p0_recon = 2 * p0_recon

    # repeat the FFT reconstruction for comparison
    p_xy = kspaceLineRecon(sensor_data.T, dy=d[1], dt=kgrid.dt.item(), c=medium.sound_speed.item(), pos_cond=True, interp="linear")

    # define a second k-space grid using the dimensions of p_xy
    N_recon = Vector(p_xy.shape)
    d_recon = Vector([kgrid.dt * medium.sound_speed.item(), kgrid.dy])
    kgrid_recon = kWaveGrid(N_recon, d_recon)

    # resample p_xy to be the same size as source.p0
    interp_func = RegularGridInterpolator(
        (kgrid_recon.x_vec[:, 0] - kgrid_recon.x_vec.min(), kgrid_recon.y_vec[:, 0]), p_xy, method="linear"
    )
    query_points = np.stack((kgrid.x - kgrid.x.min(), kgrid.y), axis=-1)
    p_xy_rs = interp_func(query_points)

    # --------------------
    # VISUALIZATION
    # --------------------

    cmap = get_color_map()

    # Plot the initial pressure and sensor distribution
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p0 + sensor.mask * disc_magnitude,
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=-disc_magnitude,
        vmax=disc_magnitude,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    fig.colorbar(im, cax=cax)
    plt.show()

    # Plot the reconstructed initial pressure
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p0_recon,
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=-disc_magnitude,
        vmax=disc_magnitude,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    fig.colorbar(im, cax=cax)
    plt.show()

    # Apply a positivity condition
    p0_recon[p0_recon < 0] = 0

    # Plot the reconstructed initial pressure with positivity condition
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p0_recon,
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=-disc_magnitude,
        vmax=disc_magnitude,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    fig.colorbar(im, cax=cax)
    plt.show()

    # Plot a profile for comparison
    plt.plot(kgrid.y_vec[:, 0] * 1e3, p0[disc_pos[0], :], "k-", label="Initial Pressure")
    plt.plot(kgrid.y_vec[:, 0] * 1e3, p_xy_rs[disc_pos[0], :], "r--", label="FFT Reconstruction")
    plt.plot(kgrid.y_vec[:, 0] * 1e3, p0_recon[disc_pos[0], :], "b:", label="Time Reversal")
    plt.xlabel("y-position [mm]")
    plt.ylabel("Pressure")
    plt.legend()
    plt.axis("tight")
    plt.ylim([0, 5.1])
    plt.show()


if __name__ == "__main__":
    main()
