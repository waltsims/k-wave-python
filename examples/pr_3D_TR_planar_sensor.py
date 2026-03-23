# %% [markdown]
# # 3D Time Reversal Reconstruction For A Planar Sensor Example
# Reconstruct a 3D photoacoustic wave-field using time reversal with a planar sensor array.

# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D  # TimeReversal requires legacy API
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction import TimeReversal
from kwave.utils.colormap import get_color_map
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball


# %%
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

    # %%
    # NOTE: pml_inside=False, data_cast="single" not supported in new API
    # run the simulation
    sensor_data = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        pml_size=PML_size,
        smooth_p0=False,
        backend="cpp",
        device="gpu",
    )
    sensor.recorded_pressure = sensor_data["p"].T  # Store the recorded pressure data

    # reset only the initial pressure source
    source = kSource()

    # create time reversal handler and run reconstruction
    tr = TimeReversal(kgrid, medium, sensor)
    # NOTE: TimeReversal requires the legacy API until it is updated
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,
        save_to_disk=True,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    p0_recon = tr(kspaceFirstOrder3D, simulation_options, execution_options)

    # %%
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


# %%
if __name__ == "__main__":
    main()
