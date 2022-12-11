"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
import os
from copy import deepcopy
from tempfile import gettempdir

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import *
from kwave.utils.filters import *
from kwave.utils.interp import cart2grid
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matlab import matlab_find, unflatten_matlab_mask
from tests.diff_utils import compare_against_ref


def test_sd_directivity_modelling_3D():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 64            # number of grid points in the x direction
    Ny = 64            # number of grid points in the y direction
    Nz = 64            # number of grid points in the z direction
    dx = 100e-3/Nx     # grid point spacing in the x direction [m]
    dy = dx            # grid point spacing in the y direction [m]
    dz = dx            # grid point spacing in the z direction [m]
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a large area detector
    sz = 16        # [grid points]
    sensor_mask = np.zeros((Nx, Ny, Nz))
    sensor_mask[Nx//2, (Ny//2 - sz//2):(Ny//2 + sz//2 + 1), (Nz//2 - sz//2):(Nz//2 + sz//2 + 1)] = 1
    sensor = kSensor(sensor_mask)

    # define equally spaced point sources lying on a circle centred at the
    # centre of the detector face
    radius = 20    # [grid points]
    points = 11
    circle = make_cart_circle(radius * dx, points, [0, 0], np.pi)
    circle = np.vstack([circle, np.zeros((1, points))])

    # find the binary sensor mask most closely corresponding to the cartesian
    # coordinates from makeCartCircle
    circle3D, _, _ = cart2grid(kgrid, circle)

    # find the indices of the sources in the binary source mask
    source_positions = matlab_find(circle3D, val=1, mode='eq')

    # define a time varying sinusoidal source
    source = kSource()
    source_freq = 0.25e6   # [Hz]
    source_mag = 1         # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    # pre-allocate array for storing the output time series
    single_element_data = np.zeros((kgrid.Nt, points))

    # run a simulation for each of these sources to see the effect that the
    # angle from the detector has on the measured signal
    for source_loop in range(points):
        # select a point source
        source.p_mask = np.zeros((Nx, Ny, Nz))
        source.p_mask[unflatten_matlab_mask(source.p_mask, source_positions[source_loop] - 1)] = 1

        # run the simulation
        input_filename = f'example_input_{source_loop + 1}'
        pathname = gettempdir()
        input_file_full_path = os.path.join(pathname, input_filename + '_input.h5')
        input_args = {
            'pml_size': 10,
            'save_to_disk': True,
            'data_name': input_filename,
            'data_path': gettempdir(),
            'save_to_disk_exit': True
        }

        # run the simulation
        kspaceFirstOrder3DC(**{
            'medium': medium,
            'kgrid': kgrid,
            'source': deepcopy(source),
            'sensor': sensor,
            **input_args
        })

        assert compare_against_ref(f'out_sd_directivity_modelling_3D/input_{source_loop + 1}', input_file_full_path), \
            'Files do not match!'
