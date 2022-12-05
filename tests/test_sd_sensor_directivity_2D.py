"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
from copy import deepcopy
from tempfile import gettempdir

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensorDirectivity
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from tests.diff_utils import compare_against_ref


def test_sd_sensor_directivity_2D():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # DEFINE THE GRID AND MEDIUM PROPERTIES
    # =========================================================================

    # create the computational grid
    Nx = 64         # number of grid points in the x (row) direction
    Ny = 64         # number of grid points in the y (column) direction
    dx = 1e-3/Nx    # grid point spacing in the x direction [m]
    dy = dx     	# grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define the array of temporal points
    t_end = 600e-9      # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE THE DIRECTIONAL SENSOR ARRAY
    # =========================================================================

    # define a line of sensor points
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[23, 1:62:2] = 1
    sensor = kSensor(sensor_mask)

    # define the angle of max directivity for each sensor point:
    #    0             = max sensitivity in x direction (up/down)
    #    pi/2 or -pi/2 = max sensitivity in y direction (left/right)
    dir_angles = np.arange(-1, 1 + 1/15, 1/15)[:, None] * np.pi/2

    # assign to the directivity mask
    directivity = kSensorDirectivity()
    directivity.angle = np.zeros((Nx, Ny))
    directivity.angle[sensor.mask == 1] = np.squeeze(dir_angles)

    # define the directivity pattern
    directivity.pattern = 'pressure'

    # define the directivity size
    directivity.size = 16 * kgrid.dx

    sensor.directivity = directivity

    # =========================================================================
    # SIMULATION AND VISUALISATION FOR AN INITIAL VALUE PROBLEM
    # =========================================================================

    # define the initial pressure distribution
    source = kSource()
    source_p0 = np.zeros((Nx, Ny))
    source_p0[38:41, :] = 2
    source.p0 = source_p0

    # run the simulation
    input_filename = f'example_def_tran'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename + '_input.h5')
    input_args = {
        'pml_alpha': np.array([2, 0]),
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }

    # run the simulation
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })

    assert compare_against_ref(f'out_sd_sensor_directivity_2D', input_file_full_path, precision=6), \
        'Files do not match!'
