"""
2D FFT Reconstruction For A Line Sensor Example (forward simulation only)

Ported from: k-Wave/examples/example_pr_2D_FFT_line_sensor.m

Simulates the propagation of an initial pressure distribution (two smoothed
discs) detected by a binary line sensor along the first row. Only the forward
simulation is ported; the FFT-based reconstruction (kspaceLineRecon) is
post-processing and is not included.
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


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    PML_size = 20  # size of the PML in grid points
    Nx = 128 - 2 * PML_size  # 88
    Ny = 256 - 2 * PML_size  # 216
    dx = 0.1e-3  # [m]
    dy = 0.1e-3  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using make_disc
    disc_magnitude = 5  # [Pa]

    # disc 2 (MATLAB 1-based positions kept as-is for make_disc)
    disc_2 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([60, 140]), 5)

    # disc 1
    disc_1 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([30, 110]), 8)

    source = kSource()
    # smooth the initial pressure distribution and restore the magnitude
    source.p0 = smooth(disc_1 + disc_2, restore_max=True)

    # set time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the forward simulation with a binary line sensor.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx = 88
    Ny = 216

    # define a binary line sensor along the first row
    sensor_mask = np.zeros((Nx, Ny), dtype=bool)
    sensor_mask[0, :] = True
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
        pml_inside=False,
        pml_size=20,
        smooth_p0=False,  # already smoothed manually
    )


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    p = np.asarray(result["p"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot the sensor data
    ax = axes[0]
    im = ax.imshow(p, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_ylabel("Sensor Position")
    ax.set_xlabel("Time Step")
    ax.set_title("Sensor Data (line sensor)")
    fig.colorbar(im, ax=ax)

    # plot the initial pressure
    kgrid, _, source = setup()
    ax = axes[1]
    ax.imshow(
        source.p0.T,
        extent=[
            kgrid.x_vec[0] * 1e3,
            kgrid.x_vec[-1] * 1e3,
            kgrid.y_vec[-1] * 1e3,
            kgrid.y_vec[0] * 1e3,
        ],
        vmin=-5,
        vmax=5,
        cmap="RdBu_r",
    )
    ax.set_ylabel("x-position [mm]")
    ax.set_xlabel("y-position [mm]")
    ax.set_title("Initial Pressure Distribution")
    ax.set_aspect("equal")

    fig.suptitle("2D FFT Line Sensor Example (forward sim)")
    fig.tight_layout()
    plt.show()
