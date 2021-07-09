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
from kwave_py.utils.maputils import makeDisc, makeCircle
from kwave_py.utils import dotdict
from kwave_py.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave_py.kmedium import kWaveMedium
from copy import deepcopy


def test_sd_focussed_detector_2D():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 180           # number of grid points in the x (row) direction
    Ny = 180           # number of grid points in the y (column) direction
    dx = 0.1e-3        # grid point spacing in the x direction [m]
    dy = 0.1e-3        # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define a sensor as part of a circle centred on the grid
    sensor_radius = 65 # [grid points]
    arc_angle = np.pi     # [rad]
    sensor_mask = makeCircle(Nx, Ny, Nx/2, Ny/2, sensor_radius, arc_angle)
    sensor = kSensor(sensor_mask)

    # define the array of temporal points
    t_end = 11e-6      # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # place a disc-shaped source near the focus of the detector
    source = kSource()
    source.p0 = 2 * makeDisc(Nx, Ny, Nx/2, Ny/2, 4)

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

    assert compare_against_ref(f'out_sd_focussed_detector_2D', input_file_full_path), \
        'Files do not match!'
