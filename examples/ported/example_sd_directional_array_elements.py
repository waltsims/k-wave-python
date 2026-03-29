"""
Focussed 2D Array With Directional Elements

Ported from: k-Wave/examples/example_sd_directional_array_elements.m

Demonstrates the output from a curved semicircular detector array consisting
of 13 elements, each spanning several grid points.  A plane-wave source at
row 139 (0-based) drives a filtered 1 MHz sinusoid.  The PML is disabled on
the y-edges (pml_alpha = [2, 0]) so the source appears infinitely wide.

After the single simulation, the recorded sensor data is split into 13
elements and averaged per-element to give one time series per element.

Note: The original MATLAB example uses the name "directional elements" but
no sensor.directivity_angle feature is involved -- the element directionality
is purely geometric (curved array + spatial averaging).

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
from kwave.utils.mapgen import make_circle


def setup():
    """Set up simulation physics (grid, medium, source).

    Grid: 180 x 180, dx = dy = 0.1 mm.
    Medium: c = 1500 m/s (lossless).
    Source: plane-wave at row 139 (0-based), 1 MHz sinusoid, filtered.

    Returns:
        tuple: (kgrid, medium, source)
    """
    # create the computational grid
    Nx = 180
    Ny = 180
    dx = 0.1e-3  # [m]
    dy = 0.1e-3
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # define the array of temporal points
    t_end = 12e-6  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # define a plane-wave source at row 140 (1-based) = row 139 (0-based)
    source = kSource()
    p_mask = np.zeros((Nx, Ny), dtype=float)
    p_mask[139, :] = 1
    source.p_mask = p_mask

    # define a time-varying sinusoidal source
    source_freq = 1e6  # [Hz]
    source_mag = 0.5  # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)
    source.p = filter_time_series(kgrid, medium, source.p)

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run single simulation with a 13-element semicircular sensor array.

    Sensor: semicircular arc of radius 65 grid points centred on grid,
            divided into 13 elements (with 2-point gaps between them).
    PML: alpha = (2, 0) -- PML active in x, disabled in y.

    Returns:
        dict with keys:
            'element_data': (13, Nt) averaged time series per element
            'p': raw sensor data (num_sensor_points x Nt)
    """
    Nx, Ny = 180, 180

    # define a semicircular sensor centred on the grid
    semicircle_radius = 65  # [grid points]
    arc_angle = np.pi
    # make_circle centre is 1-based in both languages
    arc = make_circle(
        Vector([Nx, Ny]),
        Vector([Nx // 2, Ny // 2]),
        semicircle_radius,
        arc_angle,
    )

    # find indices of the arc grid points
    arc_indices = np.flatnonzero(arc)  # 0-based flat indices
    Nv = len(arc_indices)

    # calculate angles between arc points and grid centre
    dx = 0.1e-3
    kgrid_tmp = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dx]))
    x_vec = np.asarray(kgrid_tmp.x_vec).ravel()
    y_vec = np.asarray(kgrid_tmp.y_vec).ravel()
    rows, cols = np.unravel_index(arc_indices, (Nx, Ny))
    arc_angles = np.arctan2(y_vec[cols], x_vec[rows])

    # sort the angles into ascending order
    sorted_order = np.argsort(arc_angles)
    sorted_arc_indices = arc_indices[sorted_order]

    # divide the semicircle into Ne elements (with 2-point gaps)
    Ne = 13
    sensor_mask = np.zeros((Nx, Ny), dtype=float)
    element_voxel_indices = []  # store per-element flat indices
    for loop in range(Ne):
        # MATLAB: sorted_arc_indices(floor((loop-1)*Nv/Ne)+2 : floor(loop*Nv/Ne)-1)
        # Convert 1-based MATLAB loop to 0-based Python loop (loop_py = loop_m - 1):
        #   start_1based = floor(loop_py * Nv/Ne) + 2  ->  start_0based = ... + 1
        #   stop_1based  = floor((loop_py+1) * Nv/Ne) - 1  ->  Python exclusive end = stop_1based
        start = int(np.floor(loop * Nv / Ne)) + 1  # gap of 1
        end_ = int(np.floor((loop + 1) * Nv / Ne)) - 1  # gap of 1
        voxel_idx = sorted_arc_indices[start:end_]
        element_voxel_indices.append(voxel_idx)
        sensor_mask.flat[voxel_idx] = 1

    # run the simulation
    kgrid, medium, source = setup()
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
        pml_alpha=(2, 0),
    )
    p = np.asarray(result["p"])  # (num_sensor_points, Nt)

    # map sensor data rows back to elements
    # sensor data rows correspond to the non-zero entries of sensor_mask
    # in flat (C-order) index order
    sensor_flat_indices = np.flatnonzero(sensor_mask)  # sorted ascending

    Nt = kgrid.Nt
    element_data = np.zeros((Ne, Nt))
    for loop in range(Ne):
        voxel_idx = element_voxel_indices[loop]
        # find which rows in p correspond to this element's voxels
        data_rows = np.searchsorted(sensor_flat_indices, voxel_idx)
        element_data[loop, :] = p[data_rows, :].mean(axis=0)

    return {
        "element_data": element_data,
        "p": p,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    kgrid, _, _ = setup()

    element_data = result["element_data"]
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6

    # stacked plot of element time series
    fig, ax = plt.subplots(figsize=(10, 8))
    Ne = element_data.shape[0]
    spacing = np.max(np.abs(element_data)) * 2.0
    for i in range(Ne):
        ax.plot(t_us, element_data[i, :] + i * spacing, "k-", linewidth=0.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Time Series Recorded At Each Element")
    ax.set_title("Focussed 2D Array With Directional Elements")
    ax.set_yticks([i * spacing for i in range(Ne)])
    ax.set_yticklabels([f"El {i+1}" for i in range(Ne)])

    plt.tight_layout()
    plt.show()
