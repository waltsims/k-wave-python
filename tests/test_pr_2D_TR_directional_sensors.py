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
from kwave.ksensor import kSensorDirectivity
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc
from tests.diff_utils import compare_against_ref


def test_pr_2D_TR_directional_sensors():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    PML_size = 20              # size of the PML in grid points
    Nx = 128 - 2 * PML_size    # number of grid points in the x direction
    Ny = 256 - 2 * PML_size    # number of grid points in the y direction
    dx = 0.1e-3                # grid point spacing in the x direction [m]
    dy = 0.1e-3                # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5         # [Pa]
    disc_x_pos = 60            # [grid points]
    disc_y_pos = 140           # [grid points]
    disc_radius = 5            # [grid points]
    disc_2 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    disc_x_pos = 30            # [grid points]
    disc_y_pos = 110           # [grid points]
    disc_radius = 8            # [grid points]
    disc_1 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    # smooth the initial pressure distribution and restore the magnitude
    p0 = smooth(disc_1 + disc_2, True)

    # assign to the source structure
    source = kSource()
    source.p0 = p0

    # define a four-sided, square sensor
    sensor = kSensor()
    sensor.mask = np.zeros((kgrid.Nx, kgrid.Ny))
    sensor.mask[0, :] = 1
    sensor.mask[-1, :] = 1
    sensor.mask[:, 0] = 1
    sensor.mask[:, -1] = 1

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # set the input arguements
    input_filename = f'example_tr_dir'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename + '_input.h5')
    input_args = {
        'pml_inside': False,
        'pml_size': PML_size,
        'smooth_p0': False,
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }

    # run the simulation for omnidirectional detector elements
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': deepcopy(sensor),
        **input_args
    })
    assert compare_against_ref(f'out_pr_2D_TR_directional_sensors/input_1', input_file_full_path), 'Files do not match!'

    # define the directionality of the sensor elements
    directivity = kSensorDirectivity()
    directivity.angle = np.zeros((kgrid.Nx, kgrid.Ny))
    directivity.angle[1, :] = 0    	        # max sensitivity in x direction
    directivity.angle[-1, :] = 0  	        # max sensitivity in x direction
    directivity.angle[:, 1] = np.pi / 2      # max sensitivity in y direction
    directivity.angle[:, -1] = np.pi / 2     # max sensitivity in y direction

    # define the directivity size
    directivity.size = 20 * kgrid.dx

    sensor.directivity = directivity

    # run the simulation with directional elements
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_pr_2D_TR_directional_sensors/input_2', input_file_full_path), 'Files do not match!'
