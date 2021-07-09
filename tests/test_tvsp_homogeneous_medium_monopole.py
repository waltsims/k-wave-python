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
from kwave_py.utils import dotdict
from kwave_py.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave_py.kmedium import kWaveMedium


def test_tvsp_homogeneous_medium_monopole():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 128            # number of grid points in the x (row) direction
    Ny = 128            # number of grid points in the y (column) direction
    dx = 50e-3/Nx    	# grid point spacing in the x direction [m]
    dy = dx             # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a single source point
    source = kSource()
    source.p_mask = np.zeros((Nx, Ny))
    source.p_mask[-Nx//4 - 1, Ny//2 - 1] = 1

    # define a time varying sinusoidal source
    source_freq = 0.25e6   # [Hz]
    source_mag = 2         # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filterTimeSeries(kgrid, medium, source.p)

    # define a single sensor point
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[Nx//4 - 1, Ny//2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # define the acoustic parameters to record
    sensor.record = ['p', 'p_final']

    # set the input settings
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
        'source': source,
        'sensor': sensor,
        **input_args
    })

    assert compare_against_ref(f'out_tvsp_homogeneous_medium_monopole', input_file_full_path), 'Files do not match!'
