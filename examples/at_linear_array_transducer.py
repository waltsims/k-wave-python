"""
Modelling A Linear Array Transducer Example

Ported from: k-Wave/examples/example_at_linear_array_transducer.m

Demonstrates how to use the kWaveArray class to define a linear array
transducer with 15 rectangular elements.  Electronic focusing is used to
transmit an ultrasound pulse: each element is driven by the same tone burst,
delayed so that the wavefronts arrive at the focus at the same time.  The
array is translated and rotated, and the peak pressure is recorded on the
central x-z plane.  It builds on the Defining A Source Using An Array
Transducer example.

The PML is placed outside the user grid with an automatically chosen size.
"""

# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import tone_burst

# medium parameters
c0 = 1500  # sound speed [m/s]
rho0 = 1000  # density [kg/m^3]

# source parameters
source_f0 = 1e6  # source frequency [Hz]
source_amp = 1e6  # source pressure [Pa]
source_cycles = 5  # number of tone burst cycles
source_focus = 20e-3  # focal length [m]
element_num = 15  # number of elements
element_width = 1e-3  # width [m]
element_length = 10e-3  # elevation height [m]
element_pitch = 2e-3  # pitch [m]

# transducer position
translation = Vector([5e-3, 0, 8e-3])  # [m]
rotation = Vector([0, 20, 0])  # [deg]

# grid parameters
grid_size_x = 40e-3  # [m]
grid_size_y = 20e-3  # [m]
grid_size_z = 40e-3  # [m]

# computational parameters
ppw = 3  # number of points per wavelength
t_end = 35e-6  # total compute time [s]
cfl = 0.5  # CFL number


# %%
def make_array(kgrid):
    """Define the linear array of rectangular elements (position depends on the grid).

    Returns:
        kWaveArray: the translated and rotated linear array.
    """
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)

    # add rectangular elements facing in the z-direction
    for ind in range(element_num):
        # set element y position
        x_pos = 0 - (element_num * element_pitch / 2 - element_pitch / 2) + ind * element_pitch

        # add element (set rotation angle to match the global rotation angle)
        karray.add_rect_element([x_pos, 0, kgrid.z_vec[0].item()], element_width, element_length, rotation)

    # move the array
    karray.set_array_position(translation, rotation)
    return karray


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """
    # calculate the grid spacing based on the PPW and F0
    dx = c0 / (ppw * source_f0)  # [m]

    # compute the size of the grid
    Nx = round(grid_size_x / dx)
    Ny = round(grid_size_y / dx)
    Nz = round(grid_size_z / dx)
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # create the time array
    kgrid.makeTime(c0, cfl, t_end)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=c0, density=rho0)

    # calculate the focusing time delays for each element
    if element_num % 2 != 0:
        ids = np.arange(1, element_num + 1) - np.ceil(element_num / 2)
    else:
        ids = np.arange(1, element_num + 1) - (element_num + 1) / 2
    time_delays = -(np.sqrt((ids * element_pitch) ** 2 + source_focus**2) - source_focus) / c0
    time_delays = time_delays - min(time_delays)

    # create the delayed tone burst signals (one per element)
    source_sig = source_amp * tone_burst(
        1 / kgrid.dt,
        source_f0,
        source_cycles,
        signal_offset=np.round(time_delays / kgrid.dt).astype(int),
    )

    karray = make_array(kgrid)
    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, source_sig, order="C")

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the simulation, recording the peak pressure on the central x-z plane.

    Returns:
        dict with key 'p_max': (Nx, Nz) peak pressure [Pa].
    """
    kgrid, medium, source = setup()
    Nx, Ny, Nz = (int(n) for n in kgrid.N)

    # record the peak pressure on the central x-z plane
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[:, Ny // 2, :] = True
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p_max"]

    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=False,
        pml_size="auto",
    )

    return {"p_max": np.reshape(np.asarray(result["p_max"]), (Nx, Nz))}


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from kwave.utils.colormap import get_color_map
    from kwave.utils.plot import voxel_plot

    kgrid, _, source = setup()

    # plot the source mask
    voxel_plot(np.single(source.p_mask))

    result = run(quiet=False)
    p_max = result["p_max"]

    # plot the peak pressure on the x-z plane
    fig, ax = plt.subplots()
    im = ax.imshow(
        1e-6 * p_max,
        extent=[
            1e3 * kgrid.z_vec[0].item(),
            1e3 * kgrid.z_vec[-1].item(),
            1e3 * kgrid.x_vec[-1].item(),
            1e3 * kgrid.x_vec[0].item(),
        ],
        cmap=get_color_map(),
    )
    ax.set_xlabel("z-position [mm]")
    ax.set_ylabel("x-position [mm]")
    ax.set_title("Pressure Field")
    fig.colorbar(im, ax=ax, label="[MPa]")
    plt.show()
