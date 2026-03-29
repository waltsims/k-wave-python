"""
Controlling The Absorbing Boundary Layer Example

Ported from: k-Wave/examples/example_na_controlling_the_PML.m

Demonstrates how to control the parameters of the perfectly matched layer
(PML) that absorbs acoustic waves at the edges of the computational domain.
The first-order k-Wave solvers use an anisotropic absorbing boundary layer
(PML) occupying a strip of 20 grid points around the domain edge by default.
Without it, the FFT-based spatial derivatives would cause waves leaving one
side to wrap around to the opposite side.

Four configurations are shown:
  1. PML with zero absorption (waves wrap around)
  2. PML with absorption set far too high (reflection artifacts)
  3. PML with only 2 grid points (partially effective)
  4. PML placed outside the computational domain (no domain shrinkage)

The default run() uses configuration 1 (PML alpha = 0) which is the most
instructive case: you can clearly see the wrap-around artifacts.

Builds on: example_ivp_homogeneous_medium (same grid, medium, and initial
pressure distribution but without absorption).
"""
# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.mapgen import make_disc


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Grid: 128 x 128, dx = dy = 0.1 mm.
    Medium: c = 1500 m/s (no absorption).
    Source: two initial pressure discs at (50, 50) r=8 mag=5 Pa
            and (80, 60) r=5 mag=3 Pa (1-based centres).

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 128  # number of grid points in the x (row) direction
    Ny = 128  # number of grid points in the y (column) direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using make_disc
    # -- disc 1 --
    disc_magnitude = 5  # [Pa]
    disc_x_pos = 50  # [grid points, 1-based]
    disc_y_pos = 50  # [grid points, 1-based]
    disc_radius = 8  # [grid points]
    disc_1 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([disc_x_pos, disc_y_pos]), disc_radius)

    # -- disc 2 --
    disc_magnitude = 3  # [Pa]
    disc_x_pos = 80  # [grid points, 1-based]
    disc_y_pos = 60  # [grid points, 1-based]
    disc_radius = 5  # [grid points]
    disc_2 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([disc_x_pos, disc_y_pos]), disc_radius)

    source = kSource()
    source.p0 = (disc_1 + disc_2).astype(float)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


def _run_case(case, backend="python", device="cpu", quiet=True):
    """Run one of the four PML configurations.

    Args:
        case: int 1-4 selecting the PML configuration.
        backend: 'python' or 'cpp'.
        device: 'cpu' or 'gpu'.
        quiet: suppress progress output.

    Returns:
        dict: Simulation results with keys 'p', 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx, Ny = 128, 128

    # full-grid binary sensor recording p and p_final
    sensor = kSensor(
        mask=np.ones((Nx, Ny), dtype=float),
        record=["p", "p_final"],
    )

    # select PML configuration
    kwargs = dict(backend=backend, device=device, quiet=quiet)
    if case == 1:
        # PML with no absorption — waves wrap around
        kwargs["pml_alpha"] = 0.0
        kwargs["pml_inside"] = True
    elif case == 2:
        # PML with absorption set far too high — reflection artifacts
        kwargs["pml_alpha"] = 1e6
        kwargs["pml_inside"] = True
    elif case == 3:
        # PML with only 2 grid points — partially effective
        kwargs["pml_size"] = 2
        kwargs["pml_inside"] = True
    elif case == 4:
        # PML outside the computational domain — no domain shrinkage
        kwargs["pml_inside"] = False
    else:
        raise ValueError(f"case must be 1-4, got {case}")

    return kspaceFirstOrder(kgrid, medium, source, sensor, **kwargs)


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run with PML alpha = 0 (default case, PML with no absorption).

    Uses full-grid binary sensor recording p and p_final.
    PML is inside the domain with pml_alpha=0, so waves wrap around.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    return _run_case(1, backend=backend, device=device, quiet=quiet)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    titles = [
        "Case 1: PML alpha = 0 (no absorption)",
        "Case 2: PML alpha = 1e6 (too high)",
        "Case 3: PML size = 2 (too thin)",
        "Case 4: PML outside domain",
    ]

    for i, (ax, title) in enumerate(zip(axes.ravel(), titles), start=1):
        print(f"Running case {i}...")
        result = _run_case(i, quiet=False)
        p_final = np.asarray(result["p_final"])

        im = ax.imshow(p_final.T, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_ylabel("y [grid points]")
        ax.set_xlabel("x [grid points]")
        fig.colorbar(im, ax=ax)

    fig.suptitle("Controlling The Absorbing Boundary Layer", fontsize=14)
    fig.tight_layout()
    plt.show()
