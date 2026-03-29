"""
Recording Particle Velocity Example

Ported from: k-Wave/examples/example_ivp_recording_particle_velocity.m

Demonstrates how to record the particle velocity using a binary sensor mask.
It builds on the Homogeneous and Heterogeneous Propagation Medium examples.

Four sensor points are placed at equal distance from a central disc source,
one in each cardinal direction (+x, -x, +y, -y). You should observe that
sensors along the x-axis show large ux and near-zero uy, while sensors along
the y-axis show the opposite — illustrating the vector nature of acoustic
particle velocity.
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

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 128  # number of grid points in the x direction
    Ny = 128  # number of grid points in the y direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
        density=1000,  # [kg/m^3]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # create time array
    t_end = 6e-6  # [s]
    kgrid.makeTime(1500, t_end=t_end)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5  # [au]
    disc_x_pos = Nx // 2  # [grid points, 1-based]
    disc_y_pos = Ny // 2  # [grid points, 1-based]
    disc_radius = 5  # [grid points]
    source = kSource()
    source.p0 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([disc_x_pos, disc_y_pos]), disc_radius).astype(float)

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run with four binary sensor points at cardinal directions.

    Returns:
        dict: Simulation results with keys 'p', 'ux', 'uy' (4 x time_steps each).
    """
    kgrid, medium, source = setup()

    Nx = 128
    Ny = 128

    # define four sensor points centred about source.p0
    sensor_radius = 40  # [grid points]
    sensor_mask = np.zeros((Nx, Ny), dtype=bool)
    cx, cy = Nx // 2 - 1, Ny // 2 - 1  # 0-based centre
    sensor_mask[cx + sensor_radius, cy] = True  # MATLAB: (Nx/2 + 40, Ny/2)
    sensor_mask[cx - sensor_radius, cy] = True  # MATLAB: (Nx/2 - 40, Ny/2)
    sensor_mask[cx, cy + sensor_radius] = True  # MATLAB: (Nx/2, Ny/2 + 40)
    sensor_mask[cx, cy - sensor_radius] = True  # MATLAB: (Nx/2, Ny/2 - 40)

    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p", "ux", "uy"]

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
    ux = np.asarray(result["ux"])
    uy = np.asarray(result["uy"])

    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)

    labels = ["-x sensor", "+x sensor", "-y sensor", "+y sensor"]
    for i in range(4):
        axes[i, 0].plot(p[i, :], "k-")
        axes[i, 0].set_ylabel("p")
        axes[i, 0].set_title(f"{labels[i]}" if i == 0 else "")

        axes[i, 1].plot(ux[i, :], "k-")
        axes[i, 1].set_ylabel("ux")

        axes[i, 2].plot(uy[i, :], "k-")
        axes[i, 2].set_ylabel("uy")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time Step")

    # Label rows
    for i, label in enumerate(labels):
        axes[i, 0].annotate(
            label,
            xy=(0, 0.5),
            xytext=(-50, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Recording Particle Velocity — 4 Cardinal Sensors", fontsize=14)
    fig.tight_layout()
    plt.show()
