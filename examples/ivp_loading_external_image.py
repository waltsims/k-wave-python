"""
Loading External Image Maps Example

Ported from: k-Wave/examples/example_ivp_loading_external_image.m

Demonstrates how to assign an external image to the initial pressure
distribution for the simulation of an initial value problem within a
two-dimensional homogeneous propagation medium. The image
(EXAMPLE_source_one.png, 128 x 128 grayscale) is loaded, inverted
and scaled to [0, 1], then multiplied by p0_magnitude = 3 Pa.

The sensor is a Cartesian circle (50 points, radius 4 mm) around the
domain centre; k-Wave uses Delaunay interpolation to map Cartesian
sensor points onto the grid.

Grid: 128 x 128, dx = dy = 0.1 mm.
Medium: c = 1500 m/s, alpha_coeff = 0.75 dB/(MHz^y cm), alpha_power = 1.5.

It builds on the Homogeneous Propagation Medium Example.
"""
import os

import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.io import load_image
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matrix import resize

_IMAGE_FILENAME = "EXAMPLE_source_one.png"
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _find_image():
    """Locate the EXAMPLE_source_one.png image file."""
    candidates = [
        os.path.join(_REPO_ROOT, "tests", _IMAGE_FILENAME),  # bundled in k-wave-python
        os.path.join(_REPO_ROOT, "..", "k-wave-cupy", "k-Wave", "examples", _IMAGE_FILENAME),
        os.path.join(os.path.expanduser("~"), "git", "k-wave-cupy", "k-Wave", "examples", _IMAGE_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cannot find {_IMAGE_FILENAME}. Looked in: {candidates}")


def setup():
    """Set up simulation physics (grid, medium, source).

    Loads the external image, scales to [0, 1], multiplies by p0_magnitude,
    and resizes to match the computational grid (128 x 128).

    Returns:
        tuple: (kgrid, medium, source)
    """

    # load the initial pressure distribution from an image and scale
    p0_magnitude = 3  # [Pa]
    image_path = _find_image()
    # TODO: verify load_image handles image inversion the same as MATLAB's loadImage
    p0 = p0_magnitude * load_image(image_path, is_gray=True)

    # create the computational grid
    Nx = 128  # number of grid points in the x (row) direction
    Ny = 128  # number of grid points in the y (column) direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # resize the image to match the computational grid and assign to source
    source = kSource()
    source.p0 = resize(p0, [Nx, Ny])

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # create the time array (pass scalar c -- homogeneous)
    kgrid.makeTime(1500)

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with a full-grid binary sensor recording p and p_final.

    The MATLAB original uses a 50-point Cartesian circle sensor.
    For parity testing we use a full-grid binary sensor so that the
    reference .mat file can be compared element-wise.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx, Ny = 128, 128
    sensor = kSensor(mask=np.ones((Nx, Ny), dtype=bool))
    sensor.record = ["p", "p_final"]

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kgrid, medium, source = setup()
    result = run(quiet=False)

    p0 = np.asarray(source.p0)
    p_final = np.asarray(result["p_final"])

    # --- Figure 1: initial pressure distribution ---
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    extent = [
        kgrid.y_vec[0] * 1e3,
        kgrid.y_vec[-1] * 1e3,
        kgrid.x_vec[-1] * 1e3,
        kgrid.x_vec[0] * 1e3,
    ]
    ax1.imshow(p0, extent=extent, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xlabel("y-position [mm]")
    ax1.set_ylabel("x-position [mm]")
    ax1.set_title("Initial Pressure Distribution")
    ax1.set_aspect("equal")

    # --- Figure 2: final pressure field ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(p_final, extent=extent, cmap="RdBu_r")
    ax2.set_xlabel("y-position [mm]")
    ax2.set_ylabel("x-position [mm]")
    ax2.set_title("Final Pressure Field")
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()
