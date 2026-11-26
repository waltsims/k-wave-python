"""
Defining A Sensor Using An Array Transducer Example

Ported from: k-Wave/examples/example_at_array_as_sensor.m

Demonstrates how to use the kWaveArray class to define an array transducer
with 20 arc-shaped elements which is then used as a receiver array.  It
builds on the Defining A Source Using An Array Transducer example.

Two simulations are run with the same initial pressure (a small disc and a
rectangle):

1. the sensor is a ring of 20 point detectors at the element centres;
2. the sensor is the binary mask of the 20 arc elements, and the recorded
   data is integrated over each element with kWaveArray.combine_sensor_data.

The arc detectors are less sensitive to the off-axis disc than the point
detectors, because of the spatial averaging over each element.
"""

# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.conversion import cart2grid
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import make_cart_circle, make_disc
from kwave.utils.signals import reorder_binary_sensor_data


# %%
def make_array():
    """Define a ring of 20 arc-shaped elements, all focused on the grid centre.

    Returns:
        tuple: (karray, element_pos) where element_pos is (2, 20) [m].
    """
    karray = kWaveArray()

    # define arc properties
    radius = 100e-3  # [m]
    diameter = 8e-3  # [m]
    ring_radius = 50e-3  # [m]
    num_elements = 20

    # orient all elements towards the center of the grid
    focus_pos = Vector([0, 0])  # [m]

    element_pos = make_cart_circle(ring_radius, num_elements, focus_pos)

    for idx in range(num_elements):
        karray.add_arc_element(element_pos[:, idx], radius, diameter, focus_pos)

    return karray, element_pos


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """
    # create the computational grid
    Nx = 256
    Ny = 256
    dx = 0.5e-3  # [m]
    dy = 0.5e-3  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # make a small disc in the top left of the domain, and a rectangle
    x_offset = 20  # [grid points]
    p0 = make_disc(Vector([Nx, Ny]), Vector([Nx / 4 + x_offset, Ny / 4]), 4).astype(float)
    p0[99:119, 59:199] = 1

    source = kSource()
    source.p0 = p0

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the point-detector and arc-detector simulations.

    Returns:
        dict with keys:
            'point_data': (20, Nt) pressure at the element centres
            'arc_data': (20, Nt) pressure integrated over each arc element
    """
    karray, element_pos = make_array()
    kwargs = dict(backend=backend, device=device, quiet=quiet, pml_inside=True)

    # -- run 1: point detectors at the element centres --
    # snap the Cartesian points to the grid (as in MATLAB, which uses nearest
    # neighbour interpolation for Cartesian masks) and keep track of the order
    kgrid, medium, source = setup()
    point_mask, _, reorder_index = cart2grid(kgrid, element_pos, order="C")
    result = kspaceFirstOrder(kgrid, medium, source, kSensor(mask=point_mask.astype(bool)), **kwargs)
    point_data = reorder_binary_sensor_data(np.asarray(result["p"]), reorder_index=reorder_index)

    # -- run 2: arc-shaped detectors --
    kgrid, medium, source = setup()
    sensor = kSensor(mask=karray.get_array_binary_mask(kgrid))
    result = kspaceFirstOrder(kgrid, medium, source, sensor, **kwargs)
    arc_data = karray.combine_sensor_data(kgrid, np.asarray(result["p"]), order="C")

    return {"point_data": point_data, "arc_data": arc_data}


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from kwave.utils.colormap import get_color_map

    result = run(quiet=False)
    point_data = result["point_data"]
    arc_data = result["arc_data"]

    kgrid, _, source = setup()
    karray, _ = make_array()
    t_us = np.asarray(kgrid.t_array).squeeze() * 1e6
    Nx, Ny = int(kgrid.Nx), int(kgrid.Ny)
    extent = [kgrid.y_vec[0] * 1e3, kgrid.y_vec[-1] * 1e3, kgrid.x_vec[-1] * 1e3, kgrid.x_vec[0] * 1e3]

    # plot the simulation layout (sensor, initial pressure, PML inside the grid)
    pml_size = 20
    pml_mask = np.zeros((Nx, Ny), dtype=bool)
    pml_mask[:pml_size, :] = True
    pml_mask[:, :pml_size] = True
    pml_mask[-pml_size:, :] = True
    pml_mask[:, -pml_size:] = True
    layout = np.zeros((Nx, Ny))
    layout[pml_mask] = 3
    layout[source.p0 != 0] = 2
    layout[karray.get_array_binary_mask(kgrid)] = 1

    fig, ax = plt.subplots()
    im = ax.imshow(layout, extent=extent, cmap=plt.get_cmap("viridis", 4), vmin=-0.5, vmax=3.5)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["None", "Sensor", "Initial pressure p0", "PML"])
    ax.set_xlabel("y-position [mm]")
    ax.set_ylabel("x-position [mm]")
    ax.set_title("Simulation Layout")

    # plot recorded sensor data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    data_extent = [t_us[0], t_us[-1], point_data.shape[0], 1]
    im1 = ax1.imshow(point_data, aspect="auto", extent=data_extent, cmap=get_color_map(), interpolation="none")
    ax1.set_xlabel(r"Time [$\mu$s]")
    ax1.set_ylabel("Detector Number")
    ax1.set_title("Cartesian point detectors")
    fig.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(arc_data, aspect="auto", extent=data_extent, cmap=get_color_map(), interpolation="none")
    ax2.set_xlabel(r"Time [$\mu$s]")
    ax2.set_ylabel("Detector Number")
    ax2.set_title("Arc detectors")
    fig.colorbar(im2, ax=ax2)
    fig.tight_layout()

    # plot a trace from the recorded sensor data
    fig, ax = plt.subplots()
    ax.plot(t_us, point_data[0, :], label="Cartesian point detectors")
    ax.plot(t_us, arc_data[0, :], label="Arc detectors")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel("Pressure [Pa]")
    ax.legend()

    plt.show()
