"""
3D Time Reversal Reconstruction For A Planar Sensor Example

Ported from: k-Wave/examples/example_pr_3D_TR_planar_sensor.m

Demonstrates the time-reversal reconstruction of a three-dimensional
photoacoustic wave-field recorded over a planar array of sensor elements.
It builds on the 3D FFT Reconstruction For A Planar Sensor and 2D Time
Reversal Reconstruction For A Line Sensor examples.

The forward simulation records the pressure over the first x-plane of the
grid.  For the reconstruction, the recorded data is reversed in time and
enforced as a Dirichlet boundary condition on the same sensor points, and
the final pressure field is taken as the reconstructed initial pressure.
The reconstruction is multiplied by two to compensate for only recording
over a half plane, and a positivity condition is applied.
"""

# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball

# change scale to 2 to reproduce the higher resolution figures used in the
# MATLAB help file
scale = 1
PML_size = 10  # size of the PML in grid points


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """
    # create the computational grid (the PML is added outside this grid)
    N = Vector([32, 64, 64]) * scale  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3]) / scale  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using make_ball
    ball_magnitude = 10  # [Pa]
    ball_radius = 3 * scale  # [grid points]
    p0 = ball_magnitude * make_ball(N, N / 2, ball_radius)

    # smooth the initial pressure distribution and restore the magnitude
    source = kSource()
    source.p0 = smooth(p0.astype(float), restore_max=True)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


# %%
def make_sensor(kgrid):
    """Binary planar sensor over the first x-plane of the grid."""
    sensor_mask = np.zeros(tuple(int(n) for n in kgrid.N), dtype=bool)
    sensor_mask[0, :, :] = True
    return sensor_mask


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the forward simulation and the time-reversal reconstruction.

    Returns:
        dict with keys:
            'p': (Ny*Nz, Nt) recorded sensor data
            'p0_recon': (Nx, Ny, Nz) time-reversal reconstruction
    """
    kwargs = dict(
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,  # already smoothed manually
    )

    # -- forward simulation --
    kgrid, medium, source = setup()
    sensor_mask = make_sensor(kgrid)
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]
    sensor_data = np.asarray(kspaceFirstOrder(kgrid, medium, source, sensor, **kwargs)["p"])

    # -- time-reversal reconstruction --
    tr_source = kSource()
    tr_source.p_mask = sensor_mask
    tr_source.p = np.flip(sensor_data, axis=1)
    tr_source.p_mode = "dirichlet"
    tr_sensor = kSensor(mask=sensor_mask)
    tr_sensor.record = ["p_final"]
    p0_recon = np.asarray(kspaceFirstOrder(kgrid, medium, tr_source, tr_sensor, **kwargs)["p_final"])

    # add first order compensation for only recording over a half plane
    p0_recon = 2 * p0_recon

    # apply a positivity condition
    p0_recon[p0_recon < 0] = 0

    return {"p": sensor_data, "p0_recon": p0_recon}


# %%
def plot_slices(fig, axes, data, kgrid, plot_scale, cmap):
    """Plot the central x-y, x-z and y-z slices of a 3D field."""
    Nx, Ny, Nz = (int(n) for n in kgrid.N)
    x_mm = (kgrid.x_vec.min() * 1e3, kgrid.x_vec.max() * 1e3)
    y_mm = (kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3)
    z_mm = (kgrid.z_vec.min() * 1e3, kgrid.z_vec.max() * 1e3)
    slices = [
        (data[:, :, Nz // 2], [*y_mm, x_mm[1], x_mm[0]], "x-y plane", "y-position [mm]", "x-position [mm]"),
        (data[:, Ny // 2, :], [*z_mm, x_mm[1], x_mm[0]], "x-z plane", "z-position [mm]", "x-position [mm]"),
        (data[Nx // 2, :, :], [*z_mm, y_mm[1], y_mm[0]], "y-z plane", "z-position [mm]", "y-position [mm]"),
    ]
    for ax, (img, extent, title, xlabel, ylabel) in zip(axes, slices):
        im = ax.imshow(img, extent=extent, vmin=plot_scale[0], vmax=plot_scale[1], cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from kwave.utils.colormap import get_color_map

    result = run(quiet=False)
    kgrid, _, source = setup()

    cmap = get_color_map()
    plot_scale = [-10, 10]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    plot_slices(fig, axes[0], source.p0, kgrid, plot_scale, cmap)
    plot_slices(fig, axes[1], result["p0_recon"], kgrid, plot_scale, cmap)
    axes[0, 0].annotate("Initial Pressure", (0, 1.15), xycoords="axes fraction", fontweight="bold")
    axes[1, 0].annotate("Time Reversal Reconstruction", (0, 1.15), xycoords="axes fraction", fontweight="bold")
    fig.tight_layout()
    plt.show()
