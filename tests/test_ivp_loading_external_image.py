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

from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.utils import *
from kwave.utils import dotdict
from kwave.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave.kmedium import kWaveMedium
from copy import deepcopy


def test_ivp_loading_external_image():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # load the initial pressure distribution from an image and scale the
    # magnitude
    p0_magnitude = 3
    p0 = p0_magnitude * loadImage('tests/EXAMPLE_source_one.png', True)

    # create the computational grid
    Nx = 128           # number of grid points in the x (row) direction
    Ny = 128           # number of grid points in the y (column) direction
    dx = 0.1e-3        # grid point spacing in the x direction  [m]
    dy = 0.1e-3        # grid point spacing in the y direction  [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # resize the image to match the size of the computational grid and assign
    # to the source input structure
    source = kSource()
    source.p0 = np.reshape(p0, [Nx, Ny])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # define a centered circular sensor
    sensor_radius = 4e-3   # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # run the first simulation
    input_filename  = f'example_input.h5'
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

    assert compare_against_ref(f'out_ivp_loading_external_image', input_file_full_path), \
        'Files do not match!'
