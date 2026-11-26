"""
Defining A Source Using An Array Transducer Example

Ported from: k-Wave/examples/example_at_array_as_source.m

Demonstrates how to use the kWaveArray class to define an array transducer
with three arc-shaped elements without staircasing errors.  Each element is
driven by a different tone burst.  kWaveArray computes band-limited
interpolation weights so that each physical element maps onto a set of
off-grid source points, and get_distributed_source_signal() returns one
weighted source signal per grid point.

The original MATLAB example passes an empty sensor and plays a movie of the
wavefield.  Here the full pressure field is recorded (sensor=None) so the
same animation can be shown.

For a more detailed discussion of the underlying techniques, see:
E. S. Wise, B. T. Cox, J. Jaros, & B. E. Treeby (2019). Representing
arbitrary acoustic source and sensor distributions in Fourier collocation
methods. JASA, 146(1), 278-288.
"""

# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import tone_burst


# %%
def make_array():
    """Define the three-element arc array.

    Returns:
        kWaveArray: array with three arc elements, translated and rotated.
    """
    karray = kWaveArray()

    # define arc properties
    radius = 50e-3  # [m]
    diameter = 30e-3  # [m]
    focus_pos = [-20e-3, 0]  # [m]

    # add arc-shaped elements
    karray.add_arc_element([10e-3, -40e-3], radius, diameter, focus_pos)
    karray.add_arc_element([20e-3, 0], radius, diameter, focus_pos)
    karray.add_arc_element([10e-3, 40e-3], radius, diameter, focus_pos)

    # move the array down 10 mm, and rotate by 10 degrees (this moves all the
    # elements together)
    karray.set_array_position([10e-3, 0], 10)

    return karray


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

    karray = make_array()

    # set source signals, one for each physical array element
    f1 = 100e3  # [Hz]
    f2 = 200e3  # [Hz]
    f3 = 500e3  # [Hz]
    sig1 = tone_burst(1 / kgrid.dt, f1, 3).squeeze()
    sig2 = tone_burst(1 / kgrid.dt, f2, 5).squeeze()
    sig3 = tone_burst(1 / kgrid.dt, f3, 5).squeeze()

    # combine source signals into one array
    source_signal = np.zeros((3, max(len(sig1), len(sig2), len(sig3))))
    source_signal[0, : len(sig1)] = sig1
    source_signal[1, : len(sig2)] = sig2
    source_signal[2, : len(sig3)] = sig3

    # assign binary mask from karray to the source mask, and get the
    # distributed source signals (one weighted signal per source grid point)
    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, source_signal, order="C")

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the simulation recording the pressure everywhere.

    Returns:
        dict: Simulation results with keys 'p' (Nx*Ny x Nt) and 'p_final'.
    """
    kgrid, medium, source = setup()

    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        None,  # record the full grid
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from kwave.kspaceFirstOrder import reshape_to_grid
    from kwave.utils.colormap import get_color_map

    kgrid, _, source = setup()
    result = run(quiet=False)
    Nx, Ny = int(kgrid.Nx), int(kgrid.Ny)
    p_field = reshape_to_grid(result["p"], (Nx, Ny))  # (Nx, Ny, Nt)

    extent = [kgrid.y_vec[0] * 1e3, kgrid.y_vec[-1] * 1e3, kgrid.x_vec[-1] * 1e3, kgrid.x_vec[0] * 1e3]
    p_max = np.max(np.abs(p_field))

    # animate the pressure field
    fig, ax = plt.subplots()
    image = ax.imshow(p_field[:, :, 0], extent=extent, vmin=-p_max, vmax=p_max, cmap=get_color_map())
    ax.set_xlabel("y-position [mm]")
    ax.set_ylabel("x-position [mm]")

    def update(frame):
        image.set_data(p_field[:, :, frame])
        ax.set_title(f"Time step {frame + 1}/{kgrid.Nt}")
        return [image]

    ani = FuncAnimation(fig, update, frames=range(0, int(kgrid.Nt), 2), interval=30)

    # plot the source mask and the PML (default size of 20 grid points, inside the grid)
    pml_size = 20
    pml_mask = np.zeros((Nx, Ny), dtype=bool)
    pml_mask[:pml_size, :] = True
    pml_mask[:, :pml_size] = True
    pml_mask[-pml_size:, :] = True
    pml_mask[:, -pml_size:] = True

    fig2, ax2 = plt.subplots()
    ax2.imshow(np.logical_not(source.p_mask.astype(bool) | pml_mask), extent=extent, cmap="gray")
    ax2.set_xlabel("y-position [mm]")
    ax2.set_ylabel("x-position [mm]")
    ax2.set_title("Source and PML Masks")

    plt.show()
