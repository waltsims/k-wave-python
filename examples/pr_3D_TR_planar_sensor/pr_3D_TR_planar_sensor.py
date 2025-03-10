import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.colormap import get_color_map
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball

# 3D Time Reversal Reconstruction For A Planar Sensor Example

# This example demonstrates the use of k-Wave for the reconstruction
# of a three-dimensional photoacoustic wave-field recorded over a planar
# array of sensor elements.  The sensor data is simulated and then
# time-reversed using kspaceFirstOrder3D. It builds on the 3D FFT
# Reconstruction For A Planar Sensor and 2D Time Reversal Reconstruction
# For A Line Sensor examples.


def main():
    # SIMULATION

    # change scale to 2 to reproduce the higher resolution figures used in the
    # help file
    scale = 1

    # create the computational grid
    PML_size = 10  # size of the PML in grid points
    N = Vector([32, 64, 64]) * scale - 2 * PML_size  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3]) / scale  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using makeBall
    ball_magnitude = 10  # [Pa]
    ball_radius = 3 * scale  # [grid points]
    p0 = ball_magnitude * make_ball(N, N / 2, ball_radius)

    # smooth the initial pressure distribution and restore the magnitude
    p0 = smooth(p0, True)

    # assign to the source structure
    source = kSource()
    source.p0 = p0

    # define a binary planar sensor
    sensor = kSensor()
    sensor.mask = np.zeros(N)
    sensor.mask[0] = 1

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # set the input arguements
    simulation_options = SimulationOptions(
        save_to_disk=True,
        pml_size=PML_size,
        pml_inside=False,
        smooth_p0=False,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    # run the simulation
    sensor_data = kspaceFirstOrder3D(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor_data = sensor_data["p"].T

    # reset the initial pressure
    source = kSource()
    sensor = kSensor()
    sensor.mask = np.zeros(N)
    sensor.mask[0] = 1

    # assign the time reversal data
    sensor.time_reversal_boundary_data = sensor_data

    # run the time-reversal reconstruction
    p0_recon = kspaceFirstOrder3D(kgrid, source, sensor, medium, simulation_options, execution_options)
    p0_recon = p0_recon["p_final"].T

    # add first order compensation for only recording over a half plane
    p0_recon = 2 * p0_recon

    # apply a positivity condition
    p0_recon[p0_recon < 0] = 0

    # VISUALIZATION
    cmap = get_color_map()
    plot_scale = [-10, 10]

    # plot the initial pressure
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(
        p0[:, :, N[2] // 2],
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    axs[0, 0].set_title("x-y plane")
    axs[0, 0].axis("image")

    axs[0, 1].imshow(
        p0[:, N[1] // 2, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    axs[0, 1].set_title("x-z plane")
    axs[0, 1].axis("image")

    axs[1, 0].imshow(
        p0[N[0] // 2, :, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.y_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    axs[1, 0].set_title("y-z plane")
    axs[1, 0].axis("image")

    axs[1, 1].axis("off")
    axs[1, 1].set_title("(All axes in mm)")

    plt.show()

    # plot the reconstructed initial pressure
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(
        p0_recon[:, :, N[2] // 2],
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    axs[0, 0].set_title("x-y plane")
    axs[0, 0].axis("image")

    axs[0, 1].imshow(
        p0_recon[:, N[1] // 2, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    axs[0, 1].set_title("x-z plane")
    axs[0, 1].axis("image")

    axs[1, 0].imshow(
        p0_recon[N[0] // 2, :, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.y_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    axs[1, 0].set_title("y-z plane")
    axs[1, 0].axis("image")

    axs[1, 1].axis("off")
    axs[1, 1].set_title("(All axes in mm)")

    plt.show()


if __name__ == "__main__":
    main()
