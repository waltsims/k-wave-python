import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction import TimeReversal
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
    # --------------------
    # SIMULATION
    # --------------------

    # change scale to 2 to reproduce the higher resolution figures used in the
    # help file
    scale = 1

    # create the computational grid
    PML_size = 10  # size of the PML in grid points
    N = Vector([32, 64, 64]) * scale  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3]) / scale  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create the time array
    kgrid.makeTime(medium.sound_speed)

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
    # Create sensor mask for inner grid (without PML)
    sensor.mask = np.zeros(N, dtype=bool)
    sensor.mask[0, :, :] = 1  # Planar sensor along the first x-plane
    sensor.record = ["p", "p_final"]

    # set the input arguments
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,
        save_to_disk=True,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    # run the simulation
    sensor_data = kspaceFirstOrder3D(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor.recorded_pressure = sensor_data["p"].T  # Store the recorded pressure data

    # reset only the initial pressure source
    source = kSource()
    # remove padding from sensor mask (TODO: this should be done within the simulation. Add an issue.)
    sensor.mask = sensor.mask[PML_size:-PML_size, PML_size:-PML_size, PML_size:-PML_size]

    # create time reversal handler and run reconstruction
    tr = TimeReversal(kgrid, medium, sensor)
    p0_recon = tr(kspaceFirstOrder3D, simulation_options, execution_options)

    # --------------------
    # VISUALIZATION
    # --------------------

    cmap = get_color_map()
    plot_scale = [-10, 10]

    # plot the initial pressure
    fig, axs = plt.subplots(2, 2)

    # x-y plane
    im = axs[0, 0].imshow(
        p0[:, :, N[2] // 2],
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(im, cax=cax)
    axs[0, 0].set_title("x-y plane")
    axs[0, 0].set_ylabel("x-position [mm]")
    axs[0, 0].set_xlabel("y-position [mm]")
    axs[0, 0].axis("image")

    # x-z plane
    im = axs[0, 1].imshow(
        p0[:, N[1] // 2, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(im, cax=cax)
    axs[0, 1].set_title("x-z plane")
    axs[0, 1].set_ylabel("x-position [mm]")
    axs[0, 1].set_xlabel("z-position [mm]")
    axs[0, 1].axis("image")

    # y-z plane
    im = axs[1, 0].imshow(
        p0[N[0] // 2, :, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.y_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(im, cax=cax)
    axs[1, 0].set_title("y-z plane")
    axs[1, 0].set_ylabel("y-position [mm]")
    axs[1, 0].set_xlabel("z-position [mm]")
    axs[1, 0].axis("image")

    axs[1, 1].axis("off")
    axs[1, 1].set_title("(All axes in mm)")

    plt.tight_layout()
    plt.show()

    # plot the reconstructed initial pressure
    fig, axs = plt.subplots(2, 2)

    # x-y plane
    im = axs[0, 0].imshow(
        p0_recon[:, :, N[2] // 2],
        extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(im, cax=cax)
    axs[0, 0].set_title("x-y plane")
    axs[0, 0].set_ylabel("x-position [mm]")
    axs[0, 0].set_xlabel("y-position [mm]")
    axs[0, 0].axis("image")

    # x-z plane
    im = axs[0, 1].imshow(
        p0_recon[:, N[1] // 2, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(im, cax=cax)
    axs[0, 1].set_title("x-z plane")
    axs[0, 1].set_ylabel("x-position [mm]")
    axs[0, 1].set_xlabel("z-position [mm]")
    axs[0, 1].axis("image")

    # y-z plane
    im = axs[1, 0].imshow(
        p0_recon[N[0] // 2, :, :],
        extent=[kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.y_vec.min() * 1e3],
        vmin=plot_scale[0],
        vmax=plot_scale[1],
        cmap=cmap,
    )
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(im, cax=cax)
    axs[1, 0].set_title("y-z plane")
    axs[1, 0].set_ylabel("y-position [mm]")
    axs[1, 0].set_xlabel("z-position [mm]")
    axs[1, 0].axis("image")

    axs[1, 1].axis("off")
    axs[1, 1].set_title("(All axes in mm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
