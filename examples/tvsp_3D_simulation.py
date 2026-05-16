"""
Simulations In Three Dimensions (Time-Varying Source) Example

Ported from: k-Wave/examples/example_tvsp_3D_simulation.m

Simulates a time-varying sinusoidal pressure source (2 MHz) within a
three-dimensional heterogeneous propagation medium. The left half of the
domain has a higher sound speed (1800 vs 1500 m/s) and the lower
three-quarters has a higher density (1200 vs 1000 kg/m^3).

The source is a square patch (11 x 11 grid points) at x = Nx/4, centred
in the y-z plane. After propagation, the wavefront refracts at the
sound-speed interface.

Grid: 64 x 64 x 64, dx = dy = dz = 0.1 mm.

It builds on the Monopole Point Source and 3D IVP examples.
"""
# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import filter_time_series


# %%
def setup():
    """Set up simulation physics (grid, medium, source).

    Grid: 64 x 64 x 64, dx = dy = dz = 0.1 mm.
    Medium: heterogeneous sound speed (1500/1800 m/s split at x = Nx/2)
            and density (1000/1200 kg/m^3 split at y = Ny/4).
    Source: square patch at x = Nx//4 - 1 (0-based, i.e. MATLAB Nx/4),
            spanning y in [Ny/2-5, Ny/2+5] and z in [Nz/2-5, Nz/2+5]
            (1-based), emitting a filtered 2 MHz sinusoid.

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 64  # number of grid points in the x direction
    Ny = 64  # number of grid points in the y direction
    Nz = 64  # number of grid points in the z direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    dz = 0.1e-3  # grid point spacing in the z direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the propagation medium (heterogeneous)
    c = 1500 * np.ones((Nx, Ny, Nz))  # [m/s]
    c[: Nx // 2, :, :] = 1800  # MATLAB: 1:Nx/2 (1-based) -> :32 (0-based)
    rho = 1000 * np.ones((Nx, Ny, Nz))  # [kg/m^3]
    rho[:, Ny // 4 - 1 :, :] = 1200  # MATLAB: Ny/4:end = 16:64 (1-based) -> 15: (0-based)
    medium = kWaveMedium(sound_speed=c, density=rho)

    # create the time array (pass full c array -- makeTime uses max internally)
    kgrid.makeTime(c)

    # define a square source element (11 x 11 patch)
    # MATLAB: source.p_mask(Nx/4,
    #           Ny/2 - source_radius : Ny/2 + source_radius,
    #           Nz/2 - source_radius : Nz/2 + source_radius) = 1
    # 1-based: (16, 27:37, 27:37)  -> 0-based: (15, 26:37, 26:37)
    source_radius = 5  # [grid points]
    source = kSource()
    p_mask = np.zeros((Nx, Ny, Nz), dtype=float)
    cx = Nx // 4 - 1  # 15 (0-based)
    y_lo = Ny // 2 - source_radius - 1  # 26 (0-based)
    y_hi = Ny // 2 + source_radius  # 37 (exclusive, 0-based)
    z_lo = Nz // 2 - source_radius - 1  # 26 (0-based)
    z_hi = Nz // 2 + source_radius  # 37 (exclusive, 0-based)
    p_mask[cx, y_lo:y_hi, z_lo:z_hi] = 1
    source.p_mask = p_mask

    # define a time-varying sinusoidal source
    source_freq = 2e6  # [Hz]
    source_mag = 1  # [Pa]
    t_array = np.asarray(kgrid.t_array).ravel()
    source_p = source_mag * np.sin(2 * np.pi * source_freq * t_array)

    # filter the source to remove high frequencies not supported by the grid
    # filter_time_series expects a 2D (num_signals x Nt) array
    source.p = filter_time_series(kgrid, medium, source_p.reshape(1, -1))

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run with a full-grid binary sensor recording p_final.

    The MATLAB original uses a Cartesian sensor with 'nearest' interpolation.
    For parity testing we use a full-grid binary sensor recording p_final only
    (full p recording would require ~4 GB for a 64^3 grid).

    Returns:
        dict: Simulation results with key 'p_final' (Nx x Ny x Nz).
    """
    kgrid, medium, source = setup()

    Nx, Ny, Nz = 64, 64, 64
    sensor = kSensor(mask=np.ones((Nx, Ny, Nz), dtype=bool))
    sensor.record = ["p_final"]

    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    p_final = np.asarray(result["p_final"])

    Nz_half = p_final.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # x-y slice at z = mid
    ax = axes[0]
    im = ax.imshow(p_final[:, :, Nz_half].T, cmap="RdBu_r")
    ax.set_xlabel("x [grid points]")
    ax.set_ylabel("y [grid points]")
    ax.set_title(f"p_final (z={Nz_half} slice)")
    fig.colorbar(im, ax=ax)

    # x-z slice at y = mid
    Ny_half = p_final.shape[1] // 2
    ax = axes[1]
    im = ax.imshow(p_final[:, Ny_half, :].T, cmap="RdBu_r")
    ax.set_xlabel("x [grid points]")
    ax.set_ylabel("z [grid points]")
    ax.set_title(f"p_final (y={Ny_half} slice)")
    fig.colorbar(im, ax=ax)

    # y-z slice at x = mid
    Nx_half = p_final.shape[0] // 2
    ax = axes[2]
    im = ax.imshow(p_final[Nx_half, :, :].T, cmap="RdBu_r")
    ax.set_xlabel("y [grid points]")
    ax.set_ylabel("z [grid points]")
    ax.set_title(f"p_final (x={Nx_half} slice)")
    fig.colorbar(im, ax=ax)

    fig.suptitle("3D Time-Varying Source Example")
    fig.tight_layout()
    plt.show()
