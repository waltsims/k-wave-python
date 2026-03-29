"""
Modelling Sensor Directivity In 2D Example

Ported from: k-Wave/examples/example_sd_directivity_modelling_2D.m

Demonstrates how the sensitivity of a large single-element detector varies
with the angular position of a point-like source.  Eleven equally-spaced
point sources on a semicircle (radius 30 grid points) are fired one at a
time, and the pressure recorded on a 21-point line sensor is summed to
simulate a large-aperture detector.  The resulting directivity pattern shows
the expected roll-off at oblique angles.

Note: This example does NOT use sensor.directivity_angle.  The directivity
arises purely from spatial averaging across the large detector surface.

author: Ben Cox and Bradley Treeby
date: 28th October 2010
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.conversion import cart2grid
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_cart_circle


def setup():
    """Set up simulation physics (grid, medium, source template).

    Grid: 128 x 128, dx = dy = 50e-3/128 m.
    Medium: c = 1500 m/s (lossless).
    Source: filtered 0.25 MHz sinusoid (shared waveform for all source
            positions).  The source *mask* is set per-run inside run().

    Returns:
        tuple: (kgrid, medium, source) -- source.p is the waveform,
               source.p_mask is not yet set.
    """
    # create the computational grid
    Nx = 128
    Ny = 128
    dx = 50e-3 / Nx
    dy = dx
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create the time array
    Nt = 350
    dt = 7e-8  # [s]
    kgrid.setTime(Nt, dt)

    # define a time-varying sinusoidal source (waveform only)
    source_freq = 0.25e6  # [Hz]
    source_mag = 1.0  # [Pa]
    source = kSource()
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run 11 simulations (one per source angle) and return directivity data.

    Sensor: 21-point line at row Nx/2 (0-based: row 64), columns 54..74.
    Sources: 11 points on a semicircular arc of radius 30 grid points,
             snapped to the nearest grid point via cart2grid.

    Returns:
        dict with keys:
            'single_element_data': (Nt, 11) summed sensor output per source
            'source_positions': linear indices of the 11 source grid points
    """
    Nx, Ny = 128, 128
    dx = 50e-3 / Nx

    kgrid_for_cart = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dx]))

    # define a large area detector (21 grid points along y at x = Nx//2)
    # MATLAB: sensor.mask(Nx/2+1, (Ny/2-sz/2+1):(Ny/2+sz/2+1)) = 1
    # 1-based row 65, cols 55..75  ->  0-based row 64, cols 54..74
    sz = 20
    sensor_mask = np.zeros((Nx, Ny), dtype=float)
    sensor_mask[Nx // 2, (Ny // 2 - sz // 2) : (Ny // 2 + sz // 2 + 1)] = 1

    # define equally spaced point sources on a semicircle
    radius = 30  # [grid points]
    points = 11
    circle = make_cart_circle(radius * dx, points, center_pos=Vector([0, 0]), arc_angle=np.pi)

    # snap Cartesian coordinates to nearest grid points
    circle_grid, _, _ = cart2grid(kgrid_for_cart, circle, order="C")

    # find the linear indices of the source points
    source_positions = np.flatnonzero(circle_grid)  # 0-based flat indices

    # pre-allocate output
    kgrid0, _, _ = setup()
    Nt = kgrid0.Nt
    single_element_data = np.zeros((Nt, points))

    # run a simulation for each source position
    for i in range(points):
        kgrid, medium, source = setup()

        # set source mask for this point source
        p_mask = np.zeros((Nx, Ny), dtype=float)
        p_mask.flat[source_positions[i]] = 1
        source.p_mask = p_mask

        sensor = kSensor(mask=sensor_mask.astype(bool))

        result = kspaceFirstOrder(
            kgrid,
            medium,
            source,
            sensor,
            backend=backend,
            device=device,
            quiet=quiet,
            pml_inside=True,
        )
        p = np.asarray(result["p"])

        # sum over sensor points (axis 0) to simulate large-aperture detector
        single_element_data[:, i] = p.sum(axis=0)

    return {
        "single_element_data": single_element_data,
        "source_positions": source_positions,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    kgrid, _, _ = setup()

    single_element_data = result["single_element_data"]
    source_positions = result["source_positions"]

    Nx, Ny = 128, 128
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6

    # -- Figure 1: time series for each source direction --
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_us, single_element_data)
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [au]")
    ax1.set_title("Time Series For Each Source Direction")

    # -- Figure 2: directivity pattern --
    # Reconstruct x, y coordinates of source positions to compute angles
    dx = 50e-3 / Nx
    kgrid_full = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dx]))
    x_vec = np.asarray(kgrid_full.x_vec).ravel()
    y_vec = np.asarray(kgrid_full.y_vec).ravel()
    # unravel flat indices into (row, col)
    rows, cols = np.unravel_index(source_positions, (Nx, Ny))
    x_src = x_vec[rows]
    y_src = y_vec[cols]
    angles = np.arctan(y_src / x_src)  # MATLAB uses atan(y/x), not atan2

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    # MATLAB: max(single_element_data(200:350, :))  ->  0-based rows 199:350
    ax2.plot(angles, np.max(single_element_data[199:350, :], axis=0), "o")
    ax2.set_xlabel("Angle Between Source and Centre of Detector Face [rad]")
    ax2.set_ylabel("Maximum Detected Pressure [au]")
    ax2.set_title("Directivity Pattern")

    plt.tight_layout()
    plt.show()
