"""
3D FFT Reconstruction For A Planar Sensor Example (forward simulation only)

Ported from: k-Wave/examples/example_pr_3D_FFT_planar_sensor.m

Simulates the propagation of an initial pressure distribution (smoothed ball)
detected by a binary planar sensor at x=1. Only the forward simulation is
ported; the FFT-based reconstruction (kspacePlaneRecon) is post-processing
and is not included.

Note: uses scale=1 (small grid). The DataCast 'single' option from MATLAB
is not needed for the Python backend.
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


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """

    scale = 1

    # create the computational grid
    # MATLAB uses PML_size=10 with PMLInside=false on a 12x44x44 grid,
    # expanding to 32x64x64. We use pml_inside=True on the full 32x64x64
    # grid to avoid validation issues with the Python backend.
    Nx = 32 * scale
    Ny = 64 * scale
    Nz = 64 * scale
    dx = 0.2e-3 / scale  # [m]
    dy = 0.2e-3 / scale  # [m]
    dz = 0.2e-3 / scale  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using make_ball
    # MATLAB: makeBall(Nx, Ny, Nz, Nx/2, Ny/2, Nz/2, ball_radius)
    # make_ball uses 1-based center
    ball_magnitude = 10  # [Pa]
    ball_radius = 3 * scale  # [grid points]
    ball_center = Vector([Nx // 2, Ny // 2, Nz // 2])  # 1-based
    p0 = ball_magnitude * make_ball(Vector([Nx, Ny, Nz]), ball_center, ball_radius)

    source = kSource()
    # smooth the initial pressure distribution and restore the magnitude
    source.p0 = smooth(p0.astype(float), restore_max=True)

    # set time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the forward simulation with a binary planar sensor.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx = source.p0.shape[0]
    Ny = source.p0.shape[1]
    Nz = source.p0.shape[2]

    # define a binary planar sensor at x=0 (first slice)
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[0, :, :] = True
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
        pml_inside=True,  # TODO: use pml_inside=False once validation accepts 2*pml >= N
        pml_size=10,
        smooth_p0=False,  # already smoothed manually
    )


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    p = np.asarray(result["p"])
    p_final = np.asarray(result["p_final"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot the sensor data (reshape to Ny*Nz x Nt)
    ax = axes[0]
    im = ax.imshow(p, aspect="auto", cmap="RdBu_r")
    ax.set_ylabel("Sensor Position")
    ax.set_xlabel("Time Step")
    ax.set_title("Sensor Data (planar sensor)")
    fig.colorbar(im, ax=ax)

    # plot a central slice of p_final
    ax = axes[1]
    nz_mid = p_final.shape[2] // 2
    im = ax.imshow(p_final[:, :, nz_mid].T, cmap="RdBu_r")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_title(f"p_final (z={nz_mid} slice)")
    fig.colorbar(im, ax=ax)

    fig.suptitle("3D FFT Planar Sensor Example (forward sim)")
    fig.tight_layout()
    plt.show()
