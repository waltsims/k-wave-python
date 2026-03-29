"""
Focussed Detector In 3D Example

Ported from: k-Wave/examples/example_sd_focussed_detector_3D.m

Shows how k-Wave can model the output of a focussed bowl detector in 3D
where the directionality arises from spatially averaging across the detector
surface.  Two simulations are run with a time-varying sinusoidal point
source: one placed on the bowl's axis of symmetry (near the focus), and
one placed off axis.  The spatially-averaged detector signal is much
stronger for the on-axis case.

author: Ben Cox and Bradley Treeby
date: 29th October 2010
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_bowl


def setup():
    """Set up simulation physics (grid, medium, source).

    Grid: 64 x 64 x 64, dx = dy = dz = 100e-3/64 m (~1.5625 mm).
    Medium: c = 1500 m/s (lossless).
    Source: time-varying 0.25 MHz sinusoidal point source, filtered via
            filterTimeSeries.  The source mask is set per-run (on-axis vs
            off-axis), so this function returns source with p defined but
            p_mask not yet assigned.

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 64  # number of grid points in the x direction
    Ny = 64  # number of grid points in the y direction
    Nz = 64  # number of grid points in the z direction
    dx = 100e-3 / Nx  # grid point spacing [m]
    dy = dx
    dz = dx
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a time varying sinusoidal source
    source_freq = 0.25e6  # [Hz]
    source_mag = 1  # [Pa]
    source = kSource()
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run on-axis and off-axis simulations with a concave bowl sensor.

    Bowl sensor parameters (all 1-based grid points):
        sphere_offset = 10
        diameter = Nx/2 + 1 = 33
        radius = Nx/2 = 32
        bowl_pos = (11, 32, 32)
        focus_pos = (32, 32, 32)

    Source positions (1-based):
        on-axis:  (1 + sphere_offset + radius, Ny/2, Nz/2) = (43, 32, 32)
        off-axis: (43, 38, 38)

    Returns:
        dict with keys:
            'sensor_data1': summed detector signal for on-axis source (1 x Nt)
            'sensor_data2': summed detector signal for off-axis source (1 x Nt)
    """
    Nx, Ny, Nz = 64, 64, 64

    # create a concave sensor (bowl)
    sphere_offset = 10  # [grid points]
    diameter = Nx // 2 + 1  # 33 [grid points]
    radius = Nx // 2  # 32 [grid points]
    # 1-based positions
    bowl_pos = Vector([1 + sphere_offset, Ny // 2, Nz // 2])  # (11, 32, 32)
    focus_pos = Vector([Nx // 2, Ny // 2, Nz // 2])  # (32, 32, 32)
    sensor_mask = make_bowl(Vector([Nx, Ny, Nz]), bowl_pos, radius, diameter, focus_pos)

    # --- Simulation 1: source on axis (near focus) ---
    kgrid, medium, source1 = setup()

    # MATLAB: source1(1 + sphere_offset + radius, Ny/2, Nz/2) = 1
    # 1-based (43, 32, 32) -> 0-based (42, 31, 31)
    p_mask1 = np.zeros((Nx, Ny, Nz), dtype=float)
    p_mask1[42, 31, 31] = 1
    source1.p_mask = p_mask1

    sensor1 = kSensor(mask=sensor_mask.astype(float))

    result1 = kspaceFirstOrder(
        kgrid,
        medium,
        source1,
        sensor1,
        pml_size=10,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )

    # average the data recorded at each grid point
    sensor_data1 = np.asarray(result1["p"]).sum(axis=0)

    # --- Simulation 2: source off axis ---
    kgrid, medium, source2 = setup()

    # MATLAB: source2(1 + sphere_offset + radius, Ny/2 + 6, Nz/2 + 6) = 1
    # 1-based (43, 38, 38) -> 0-based (42, 37, 37)
    p_mask2 = np.zeros((Nx, Ny, Nz), dtype=float)
    p_mask2[42, 37, 37] = 1
    source2.p_mask = p_mask2

    sensor2 = kSensor(mask=sensor_mask.astype(float))

    result2 = kspaceFirstOrder(
        kgrid,
        medium,
        source2,
        sensor2,
        pml_size=10,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )

    sensor_data2 = np.asarray(result2["p"]).sum(axis=0)

    return {
        "sensor_data1": sensor_data1,
        "sensor_data2": sensor_data2,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kgrid, _, _ = setup()
    result = run(quiet=False)

    # time axis in microseconds
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_us, result["sensor_data1"], "b-", label="Source on axis")
    ax.plot(t_us, result["sensor_data2"], "r-", label="Source off axis")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Average Pressure Measured By Focussed Detector [Pa]")
    ax.legend()
    ax.set_title("Focussed Detector In 3D")
    fig.tight_layout()
    plt.show()
