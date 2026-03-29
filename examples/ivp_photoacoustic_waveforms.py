"""
Photoacoustic Waveforms in 1D, 2D and 3D Example

Ported from: k-Wave/examples/example_ivp_photoacoustic_waveforms.m

Compares the time-varying pressure signals recorded from a photoacoustic
source in 1D, 2D, and 3D. The waveforms differ because a point source in 1D
corresponds to a plane wave in 3D, and a point source in 2D corresponds to
an infinite line source in 3D.

It builds on the Simulations in One Dimension, Homogeneous Propagation
Medium, and Simulations in Three Dimensions examples.

setup() prepares all three dimensionalities (1D slab, 2D disc, 3D ball).
run() executes the 2D case, which is the most representative.
"""
# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.mapgen import make_ball, make_disc


# %%
def setup():
    """Set up the simulation physics for 1D, 2D, and 3D cases.

    All three cases share the same grid size (Nx=64), spacing, medium
    properties (c=1500 m/s), source radius (2 grid points), and time array
    (dt=2e-9 s, t_end=300e-9 s using kgrid.setTime).

    Returns:
        tuple: (kgrid, medium, source) for the 2D case
            -- the full set of all three cases is stored internally
            for use by the visualization code.
    """

    # =====================================================================
    # SETTINGS
    # =====================================================================

    # size of the computational grid
    Nx = 64  # number of grid points in the x (row) direction
    x = 1e-3  # size of the domain in the x direction [m]
    dx = x / Nx  # grid point spacing in the x direction [m]

    # define the properties of the propagation medium
    c = 1500  # [m/s]

    # size of the initial pressure distribution
    source_radius = 2  # [grid points]

    # distance between the centre of the source and the sensor
    source_sensor_distance = 10  # [grid points]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]

    # MATLAB: round(t_end / dt) + 1 -- round half away from zero
    # For this ratio (300e-9 / 2e-9 = 150.0) the result is exact.
    Nt = int(np.round(t_end / dt)) + 1  # 151

    # =====================================================================
    # 2D CASE (returned by setup)
    # =====================================================================

    # create the computational grid
    kgrid = kWaveGrid(Vector([Nx, Nx]), Vector([dx, dx]))

    # create the time array using setTime (not makeTime)
    kgrid.setTime(Nt, dt)

    # create initial pressure distribution -- disc centered at (Nx/2, Nx/2)
    # MATLAB: makeDisc(Nx, Nx, Nx/2, Nx/2, source_radius) -- 1-based center
    source = kSource()
    source.p0 = make_disc(
        Vector([Nx, Nx]),
        Vector([Nx // 2, Nx // 2]),  # 1-based center: 32
        source_radius,
    ).astype(float)

    medium = kWaveMedium(sound_speed=c)

    return kgrid, medium, source


# %%
def setup_1d():
    """Set up the 1D case (slab source, single sensor point).

    Returns:
        tuple: (kgrid, medium, source, sensor)
    """
    Nx = 64
    x = 1e-3
    dx = x / Nx
    c = 1500
    source_radius = 2
    source_sensor_distance = 10
    dt = 2e-9
    t_end = 300e-9
    Nt = int(np.round(t_end / dt)) + 1

    kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))
    kgrid.setTime(Nt, dt)

    # create initial pressure distribution -- slab
    # MATLAB: source.p0(Nx/2 - source_radius : Nx/2 + source_radius) = 1
    # MATLAB 1-based: indices 30..34 -> Python 0-based: 29..33
    source = kSource()
    p0 = np.zeros(Nx)
    p0[Nx // 2 - source_radius - 1 : Nx // 2 + source_radius] = 1.0
    source.p0 = p0

    # single sensor point
    # MATLAB: sensor.mask(Nx/2 + source_sensor_distance) = 1 -> index 42 (1-based)
    sensor_mask = np.zeros(Nx, dtype=bool)
    sensor_mask[Nx // 2 + source_sensor_distance - 1] = True  # 0-based: 41
    sensor = kSensor(mask=sensor_mask)

    medium = kWaveMedium(sound_speed=c)
    return kgrid, medium, source, sensor


# %%
def setup_3d():
    """Set up the 3D case (ball source, single sensor point).

    Returns:
        tuple: (kgrid, medium, source, sensor)
    """
    Nx = 64
    x = 1e-3
    dx = x / Nx
    c = 1500
    source_radius = 2
    source_sensor_distance = 10
    dt = 2e-9
    t_end = 300e-9
    Nt = int(np.round(t_end / dt)) + 1

    kgrid = kWaveGrid(Vector([Nx, Nx, Nx]), Vector([dx, dx, dx]))
    kgrid.setTime(Nt, dt)

    # create initial pressure distribution -- ball centered at (Nx/2, Nx/2, Nx/2)
    source = kSource()
    source.p0 = make_ball(
        Vector([Nx, Nx, Nx]),
        Vector([Nx // 2, Nx // 2, Nx // 2]),  # 1-based center: 32
        source_radius,
    ).astype(float)

    # single sensor point
    # MATLAB: sensor.mask(Nx/2 - source_sensor_distance, Nx/2, Nx/2) = 1
    # 1-based: (22, 32, 32) -> 0-based: (21, 31, 31)
    sensor_mask = np.zeros((Nx, Nx, Nx), dtype=bool)
    sensor_mask[Nx // 2 - source_sensor_distance - 1, Nx // 2 - 1, Nx // 2 - 1] = True
    sensor = kSensor(mask=sensor_mask)

    medium = kWaveMedium(sound_speed=c)
    return kgrid, medium, source, sensor


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the 2D case with a full-grid binary sensor.

    Returns:
        dict: Simulation results with keys 'p', 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx = 64

    # full-grid binary sensor for parity testing
    sensor = kSensor(mask=np.ones((Nx, Nx), dtype=bool))
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

    # run all three dimensionalities for comparison
    print("Running 1D simulation...")
    kgrid_1d, medium_1d, source_1d, sensor_1d = setup_1d()
    result_1d = kspaceFirstOrder(
        kgrid_1d,
        medium_1d,
        source_1d,
        sensor_1d,
        backend="python",
        device="cpu",
        quiet=False,
        pml_inside=True,
    )

    print("Running 2D simulation...")
    result_2d = run(quiet=False)

    print("Running 3D simulation...")
    kgrid_3d, medium_3d, source_3d, sensor_3d = setup_3d()
    result_3d = kspaceFirstOrder(
        kgrid_3d,
        medium_3d,
        source_3d,
        sensor_3d,
        backend="python",
        device="cpu",
        quiet=False,
        pml_inside=True,
    )

    # extract time series
    p_1d = np.asarray(result_1d["p"]).ravel()
    # for 2D full-grid, pick a single sensor point matching MATLAB's
    # sensor.mask(Nx/2 - source_sensor_distance, Nx/2) = 1
    # 1-based: (22, 32) -> 0-based: (21, 31)
    Nx = 64
    p_2d_full = np.asarray(result_2d["p"])
    sensor_idx = 21 * Nx + 31  # row-major flat index for (21, 31)
    p_2d = p_2d_full[sensor_idx, :]
    p_3d = np.asarray(result_3d["p"]).ravel()

    # normalize
    p_1d_norm = p_1d / np.max(np.abs(p_1d))
    p_2d_norm = p_2d / np.max(np.abs(p_2d))
    p_3d_norm = p_3d / np.max(np.abs(p_3d))

    # compute time axis
    t_end = 300e-9
    dt = 2e-9
    Nt = int(np.round(t_end / dt)) + 1
    t = np.arange(Nt) * dt * 1e9  # [ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, p_1d_norm, "b-", label="1D")
    ax.plot(t, p_2d_norm, "r-", label="2D")
    ax.plot(t, p_3d_norm, "k-", label="3D")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Recorded Pressure [au]")
    ax.legend()
    ax.set_title("Photoacoustic Waveforms in 1D, 2D and 3D")
    ax.set_xlim([0, t[-1]])
    fig.tight_layout()
    plt.show()
