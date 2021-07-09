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
from kwave_py.utils import *
from kwave_py.utils import dotdict
from kwave_py.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave_py.kmedium import kWaveMedium
from copy import deepcopy


def test_sd_directivity_modelling_2D():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 128            # number of grid points in the x (row) direction
    Ny = 128            # number of grid points in the y (column) direction
    dx = 50e-3/Nx       # grid point spacing in the x direction [m]
    dy = dx             # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define the array of time points [s]
    Nt = 350
    dt = 7e-8            # [s]
    kgrid.setTime(Nt, dt)

    # define a large area detector
    sz = 20              # [grid points]
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[Nx//2, (Ny//2 - sz//2):(Ny//2 + sz//2) + 1] = 1
    sensor = kSensor(sensor_mask)

    # define equally spaced point sources lying on a circle centred at the
    # centre of the detector face
    radius = 30    # [grid points]
    points = 11
    circle = makeCartCircle(radius * dx, points, [0, 0], np.pi)

    # find the binary sensor mask most closely corresponding to the Cartesian
    # coordinates from makeCartCircle
    circle, _, _ = cart2grid(kgrid, circle)

    # find the indices of the sources in the binary source mask
    source_positions = matlab_find(circle, val=1, mode='eq')

    # define a time varying sinusoidal source
    source = kSource()
    source_freq = 0.25e6    # [Hz]
    source_mag = 1          # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filterTimeSeries(kgrid, medium, source.p)

    # pre-allocate array for storing the output time series
    single_element_data = np.zeros((Nt, points))


    # run a simulation for each of these sources to see the effect that the
    # angle from the detector has on the measured signal
    for source_loop in range(points):

        # select a point source
        source.p_mask = np.zeros((Nx, Ny))
        source.p_mask[unflatten_matlab_mask(source.p_mask, source_positions[source_loop] - 1)] = 1

        # run the simulation
        input_filename  = f'example_input_{source_loop + 1}.h5'
        input_file_full_path = os.path.join(pathname, input_filename)
        input_args = {
            'SaveToDisk': input_file_full_path,
            'SaveToDiskExit': True
        }

        # run the simulation
        kspaceFirstOrder2DC(**{
            'medium': medium,
            'kgrid': kgrid,
            'source': deepcopy(source),
            'sensor': sensor,
            **input_args
        })

        assert compare_against_ref(f'out_sd_directivity_modelling_2D/input_{source_loop + 1}', input_file_full_path), \
            'Files do not match!'
