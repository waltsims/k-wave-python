"""
2D Time Reversal Reconstruction For A Line Sensor Example

Ported from: k-Wave/examples/example_pr_2D_TR_line_sensor.m

Demonstrates the time-reversal reconstruction of a two-dimensional
photoacoustic wave-field recorded over a linear array of sensor elements.
It builds on the 2D FFT Reconstruction For A Line Sensor example.

The forward simulation records the pressure along the first row of the
grid.  For the reconstruction, the recorded data is reversed in time and
enforced as a Dirichlet boundary condition on the same sensor points, and
the final pressure field is taken as the reconstructed initial pressure.
Because the data is only recorded over a half plane, the reconstruction is
multiplied by two, and a positivity condition is applied.  The FFT-based
reconstruction (kspaceLineRecon) is also computed for comparison.
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
from kwave.utils.mapgen import make_disc

PML_size = 20  # size of the PML in grid points


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """
    # create the computational grid (the PML is added outside this grid)
    Nx = 128 - 2 * PML_size  # number of grid points in the x direction
    Ny = 256 - 2 * PML_size  # number of grid points in the y direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using make_disc
    disc_magnitude = 5  # [Pa]
    disc_2 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([60, 140]), 5)
    disc_1 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([30, 110]), 8)

    # smooth the initial pressure distribution and restore the magnitude
    source = kSource()
    source.p0 = smooth(disc_1 + disc_2, restore_max=True)

    # set time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


# %%
def make_sensor(kgrid):
    """Binary line sensor along the first row of the grid."""
    sensor_mask = np.zeros((int(kgrid.Nx), int(kgrid.Ny)), dtype=bool)
    sensor_mask[0, :] = True
    return sensor_mask


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the forward simulation and the time-reversal reconstruction.

    Returns:
        dict with keys:
            'p': (Ny, Nt) recorded sensor data
            'p0_recon': (Nx, Ny) time-reversal reconstruction
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
    # enforce the time-reversed sensor data on the sensor points and record
    # the final pressure field
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
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.interpolate import RegularGridInterpolator

    from kwave.kspaceLineRecon import kspaceLineRecon
    from kwave.utils.colormap import get_color_map

    result = run(quiet=False)
    p0_recon = result["p0_recon"]
    kgrid, medium, source = setup()
    sensor_mask = make_sensor(kgrid)
    c0 = float(medium.sound_speed)

    # repeat the FFT reconstruction for comparison (kspaceLineRecon expects (Nt, Ny))
    p_xy = kspaceLineRecon(result["p"].T, dy=kgrid.dy, dt=kgrid.dt, c=c0, pos_cond=True, interp="linear")

    # resample p_xy onto the simulation grid
    kgrid_recon = kWaveGrid(Vector(p_xy.shape), Vector([kgrid.dt * c0, kgrid.dy]))
    interp_func = RegularGridInterpolator(
        (kgrid_recon.x_vec[:, 0] - kgrid_recon.x_vec.min(), kgrid_recon.y_vec[:, 0]),
        p_xy,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    p_xy_rs = interp_func(np.stack((kgrid.x - kgrid.x.min(), kgrid.y), axis=-1))

    cmap = get_color_map()
    disc_magnitude = 5
    extent = [kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    images = [
        (source.p0 + sensor_mask * disc_magnitude, "Initial Pressure and Sensor Distribution"),
        (p0_recon, "Time Reversal Reconstruction\nwith Positivity Condition"),
        (p_xy_rs, "FFT Reconstruction"),
    ]
    for ax, (data, title) in zip(axes.flat, images):
        im = ax.imshow(data, extent=extent, vmin=-disc_magnitude, vmax=disc_magnitude, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("y-position [mm]")
        ax.set_ylabel("x-position [mm]")
        fig.colorbar(im, ax=ax)

    # plot a profile through the centre of disc 2 for comparison
    ax = axes[1, 1]
    row = 60 - 1  # disc 2 x-position (1-based 60)
    y_mm = kgrid.y_vec[:, 0] * 1e3
    ax.plot(y_mm, source.p0[row, :], "k-", label="Initial Pressure")
    ax.plot(y_mm, p_xy_rs[row, :], "r--", label="FFT Reconstruction")
    ax.plot(y_mm, p0_recon[row, :], "b:", label="Time Reversal")
    ax.set_xlabel("y-position [mm]")
    ax.set_ylabel("Pressure")
    ax.set_ylim(0, 5.1)
    ax.legend()

    fig.tight_layout()
    plt.show()
