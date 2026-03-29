"""
Optimising k-Wave Performance Example

Ported from: k-Wave/examples/example_na_optimising_performance.m

Demonstrates a simulation using an image-derived initial pressure distribution
with a Cartesian circular sensor. DataCast options are skipped (not needed
for the Python backend). Uses the binary sensor mask approach (MATLAB case 2)
for optimal performance.
"""
# %%
from pathlib import Path

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


def _find_image():
    """Locate the EXAMPLE_source_two.bmp image file."""
    # try common locations
    candidates = [
        Path(__file__).parents[2] / "tests" / "EXAMPLE_source_two.bmp",
        Path.cwd() / "tests" / "EXAMPLE_source_two.bmp",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("Cannot find EXAMPLE_source_two.bmp. " "Expected in <k-wave-python>/tests/EXAMPLE_source_two.bmp")


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """

    # assign the grid size and create the computational grid
    Nx = 256
    Ny = 256
    x = 10e-3  # [m]
    y = 10e-3  # [m]
    dx = x / Nx  # [m]
    dy = y / Ny  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # load the initial pressure distribution from an image and scale
    p0_magnitude = 2  # [Pa]
    img_path = _find_image()
    p0 = p0_magnitude * load_image(img_path, is_gray=True)
    p0 = resize(p0, [Nx, Ny])

    source = kSource()
    source.p0 = p0.astype(float)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # set time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run with a Cartesian circular sensor.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    # define a centered Cartesian circular sensor
    sensor_radius = 4.5e-3  # [m]
    num_sensor_points = 100
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(mask=sensor_mask)
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


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    p = np.asarray(result["p"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot the sensor data
    ax = axes[0]
    im = ax.imshow(p, aspect="auto", cmap="RdBu_r")
    ax.set_ylabel("Sensor Position")
    ax.set_xlabel("Time Step")
    ax.set_title("Sensor Data")
    fig.colorbar(im, ax=ax)

    # plot the initial pressure distribution
    kgrid, _, source = setup()
    sensor_mask = make_cart_circle(4.5e-3, 100)
    ax = axes[1]
    ax.imshow(
        source.p0.T,
        extent=[
            kgrid.x_vec[0] * 1e3,
            kgrid.x_vec[-1] * 1e3,
            kgrid.y_vec[-1] * 1e3,
            kgrid.y_vec[0] * 1e3,
        ],
        cmap="RdBu_r",
    )
    ax.plot(sensor_mask[0, :] * 1e3, sensor_mask[1, :] * 1e3, "k.", markersize=3)
    ax.set_xlabel("y-position [mm]")
    ax.set_ylabel("x-position [mm]")
    ax.set_title("Initial Pressure + Sensor")
    ax.set_aspect("equal")

    fig.suptitle("Optimising Performance Example")
    fig.tight_layout()
    plt.show()
