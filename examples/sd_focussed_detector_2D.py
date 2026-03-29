"""
Focussed Detector Example In 2D

Ported from: k-Wave/examples/example_sd_focussed_detector_2D.m

Shows how k-Wave can model the output of a focussed semicircular detector
where the directionality arises from spatially averaging across the detector
surface.  Two simulations are run: one with a disc-shaped source at the
detector's focus, and one with the source displaced from focus.  The
averaged detector signal is much stronger for the on-focus case.

author: Ben Cox and Bradley Treeby
date: 20th January 2010
"""
# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.mapgen import make_circle, make_disc


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Grid: 180 x 180, dx = dy = 0.1 mm.
    Medium: c = 1500 m/s (lossless).
    Source: disc of radius 4 at grid centre with magnitude 2 (on-focus case).

    Two source configurations are used in the MATLAB example (on-focus and
    off-focus).  This function returns the on-focus source; the run()
    function handles both cases internally.

    Returns:
        tuple: (kgrid, medium, source) -- source.p0 is for the on-focus case.
    """

    # create the computational grid
    Nx = 180  # number of grid points in the x (row) direction
    Ny = 180  # number of grid points in the y (column) direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # define the array of temporal points
    t_end = 11e-6  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # place a disc-shaped source at the focus of the detector
    # MATLAB: makeDisc(Nx, Ny, Nx/2, Ny/2, 4)  ->  centre (90, 90) 1-based
    source = kSource()
    source.p0 = 2.0 * make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 4).astype(float)

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run both on-focus and off-focus simulations with a binary arc sensor.

    Sensor: semicircular arc (pi radians) of radius 65 grid points centred
    on the grid.

    Returns:
        dict with keys:
            'sensor_output1': averaged detector signal for on-focus source
            'sensor_output2': averaged detector signal for off-focus source
            'p1': raw sensor data for on-focus (sensor_points x Nt)
            'p2': raw sensor data for off-focus (sensor_points x Nt)
    """
    Nx, Ny = 180, 180

    # define a sensor as part of a circle centred on the grid
    sensor_radius = 65  # [grid points]
    arc_angle = np.pi  # [rad]
    # MATLAB: makeCircle(Nx, Ny, Nx/2, Ny/2, sensor_radius, arc_angle)
    # make_circle centre is 1-based in both languages
    sensor_mask = make_circle(
        Vector([Nx, Ny]),
        Vector([Nx // 2, Ny // 2]),
        sensor_radius,
        arc_angle,
    )

    # --- Simulation 1: source on focus ---
    kgrid, medium, source = setup()
    sensor = kSensor(mask=sensor_mask.astype(bool))

    result1 = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )
    p1 = np.asarray(result1["p"])

    # --- Simulation 2: source off focus ---
    kgrid, medium, _ = setup()
    # MATLAB: makeDisc(Nx, Ny, Nx/2, Ny/2 + 20, 4)  ->  centre (90, 110) 1-based
    source2 = kSource()
    source2.p0 = 2.0 * make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2 + 20]), 4).astype(float)

    sensor2 = kSensor(mask=sensor_mask.astype(bool))
    result2 = kspaceFirstOrder(
        kgrid,
        medium,
        source2,
        sensor2,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )
    p2 = np.asarray(result2["p"])

    # average over sensor points (axis 0)
    num_sensor_pts = sensor_mask.sum()
    sensor_output1 = p1.sum(axis=0) / num_sensor_pts
    sensor_output2 = p2.sum(axis=0) / num_sensor_pts

    return {
        "sensor_output1": sensor_output1,
        "sensor_output2": sensor_output2,
        "p1": p1,
        "p2": p2,
    }


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kgrid, medium, source = setup()
    result = run(quiet=False)

    sensor_output1 = result["sensor_output1"]
    sensor_output2 = result["sensor_output2"]

    # time axis in microseconds
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6

    fig, ax = plt.subplots(figsize=(8, 5))
    energy1 = round(np.sum(sensor_output1**2) * 100) / 100
    energy2 = round(np.sum(sensor_output2**2) * 100) / 100
    ax.plot(t_us, sensor_output1, "k", label=f"Source on focus, sum(output^2) = {energy1}")
    ax.plot(t_us, sensor_output2, "r", label=f"Source off focus, sum(output^2) = {energy2}")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Average Pressure Measured Over Detector [au]")
    ax.legend()
    ax.set_title("Focussed Detector In 2D")
    fig.tight_layout()
    plt.show()
