# %% [markdown]
# # 2D FFT Reconstruction For A Line Sensor Example
# Reconstruct a 2D photoacoustic wave-field from a linear array using kspaceLineRecon.

# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.kspaceLineRecon import kspaceLineRecon
from kwave.utils.colormap import get_color_map
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc


# %%
def main():
    # --------------------
    # SIMULATION
    # --------------------

    # create the computational grid
    PML_size = 20  # size of the PML in grid points
    N = Vector([128, 256])  # number of grid points (PML is inside the grid)
    d = Vector([0.1e-3, 0.1e-3])  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using makeDisc
    disc_magnitude = 5  # [Pa]
    disc_pos = Vector([60, 140])
    disc_radius = 5
    disc_2 = disc_magnitude * make_disc(N, disc_pos, disc_radius)

    disc_pos = Vector([30, 110])
    disc_radius = 8
    disc_1 = disc_magnitude * make_disc(N, disc_pos, disc_radius)

    # smooth the initial pressure distribution and restore the magnitude
    p0 = disc_1 + disc_2
    p0 = smooth(p0, restore_max=True)

    source = kSource()
    source.p0 = p0

    # define a binary line sensor
    sensor = kSensor()
    sensor.mask = np.zeros(N)
    sensor.mask[0] = 1

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # %%
    # NOTE: pml_inside=False not supported in new API
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
    sensor_data = sensor_data["p"]  # No transpose needed - fixed in executor

    # reconstruct the initial pressure
    p_xy = kspaceLineRecon(sensor_data.T, dy=d[1], dt=kgrid.dt.item(), c=medium.sound_speed.item(), pos_cond=True, interp="linear")

    # define a second k-space grid using the dimensions of p_xy
    N_recon = Vector(p_xy.shape)
    d_recon = Vector([kgrid.dt.item() * medium.sound_speed.item(), kgrid.dy])
    kgrid_recon = kWaveGrid(N_recon, d_recon)

    # resample p_xy to be the same size as source.p0
    interp_func = RegularGridInterpolator(
        (kgrid_recon.x_vec[:, 0] - kgrid_recon.x_vec.min(), kgrid_recon.y_vec[:, 0]), p_xy, method="linear"
    )
    query_points = np.stack((kgrid.x - kgrid.x.min(), kgrid.y), axis=-1)
    p_xy_rs = interp_func(query_points)

    # %%
    # --------------------
    # VISUALIZATION
    # --------------------

    cmap = get_color_map()

    # plot the initial pressure and sensor distribution
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

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(sensor_data, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    ax.set_ylabel("Sensor Position")
    ax.set_xlabel("Time Step")
    fig.colorbar(im, cax=cax)
    plt.show()

    # plot the reconstructed initial pressure
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        p_xy_rs,
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

    # plot a profile for comparison
    plt.plot(kgrid.y_vec[:, 0] * 1e3, p0[disc_pos[0], :], "k-", label="Initial Pressure")
    plt.plot(kgrid.y_vec[:, 0] * 1e3, p_xy_rs[disc_pos[0], :], "r--", label="Reconstructed Pressure")
    plt.xlabel("y-position [mm]")
    plt.ylabel("Pressure")
    plt.legend()
    plt.axis("tight")
    plt.ylim([0, 5.1])
    plt.show()


# %%
if __name__ == "__main__":
    main()
