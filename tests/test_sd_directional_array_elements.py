"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
# noinspection PyUnresolvedReferences
import setup_test
import os
from tempfile import gettempdir

from kwave_py.ksource import kSource
from kwave_py.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave_py.utils.filterutils import filterTimeSeries
from kwave_py.utils.maputils import makeCircle
from kwave_py.utils import dotdict
from kwave_py.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave_py.kmedium import kWaveMedium


def test_sd_directional_array_elements():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 180            # number of grid points in the x (row) direction
    Ny = 180            # number of grid points in the y (column) direction
    dx = 0.1e-3         # grid point spacing in the x direction [m]
    dy = 0.1e-3         # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)


    # define the array of time points [s]
    t_end = 12e-6       # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE A FOCUSSED ARRAY OF DIRECTIONAL ELEMENTS
    # =========================================================================

    # define a semicircular sensor centred on the grid
    semicircle_radius = 65  # [grid points]
    arc = makeCircle(Nx, Ny, Nx//2 - 1, Ny//2 - 1, semicircle_radius, np.pi)

    # find total number and indices of the grid points constituting the semicircle
    arc_indices = matlab_find(arc, val=1, mode='eq')
    Nv = len(arc_indices)

    # calculate angles between grid points in the arc and the centre of the grid
    arc_angles = np.arctan((matlab_mask(kgrid.y, arc_indices) / matlab_mask(kgrid.x, arc_indices - 1)))  # -1 compatibility

    # sort the angles into ascending order, and adjust the indices accordingly
    sorted_index = arc_angles.ravel().argsort()
    sorted_arc_indices = arc_indices[sorted_index]

    # divide the semicircle into Ne separate sensor elements
    Ne = 13
    sensor_mask = np.zeros((Nx, Ny))
    for loop in range(1, Ne + 1):
        # get the indices of the grid points belonging to the current element
        # (there is a two grid point gap between the elements)
        voxel_indices = sorted_arc_indices[((loop - 1) * Nv // Ne) + 1:(loop * Nv // Ne) - 1] - 1  # -1 compatibility

        # add the element to the sensor.mask
        sensor_mask[unflatten_matlab_mask(sensor_mask, voxel_indices)] = 1
    sensor = kSensor(sensor_mask)

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # define an infinitely wide plane wave source (this is achieved by turning off the PML)
    source = kSource()
    source.p_mask = np.zeros((Nx, Ny))
    source.p_mask[139, :] = 1

    # define a time varying sinusoidal source
    source_freq = 1e6   # [Hz]
    source_mag = 0.5    # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)
    source.p = filterTimeSeries(kgrid, medium, source.p)

    # run the simulation with the PML switched off on the sides, so that the
    # source continues up to the edge of the domain (and from there infinitely,
    # because of the periodic assumption implicit in pseudospectral methods)
    # input arguments
    input_filename  = f'example_input.h5'
    input_file_full_path = os.path.join(pathname, input_filename)
    input_args = {
        'PMLAlpha': np.array([2, 0]),
        'SaveToDisk': input_file_full_path,
        'SaveToDiskExit': True
    }

    # run the simulation
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })

    assert compare_against_ref(f'out_sd_directional_array_elements', input_file_full_path), 'Files do not match!'
