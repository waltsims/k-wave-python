"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
from tempfile import gettempdir

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from tests.diff_utils import compare_against_ref


def test_na_controlling_the_pml():
    # pathname for the input and output files
    pathname = gettempdir()

    # modify this parameter to run the different examples
    example_number = 1
    # 1: PML with no absorption
    # 2: PML with the absorption value set too high
    # 3: partially effective PML
    # 4: PML set to be outside the computational domain

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 128           # number of grid points in the x (row) direction
    Ny = 128           # number of grid points in the y (column) direction
    dx = 0.1e-3        # grid point spacing in the x direction [m]
    dy = 0.1e-3        # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5 # [Pa]
    disc_x_pos = 50    # [grid points]
    disc_y_pos = 50    # [grid points]
    disc_radius = 8    # [grid points]
    disc_1 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    disc_magnitude = 3 # [Pa]
    disc_x_pos = 80    # [grid points]
    disc_y_pos = 60    # [grid points]
    disc_radius = 5    # [grid points]
    disc_2 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # define a centered circular sensor
    sensor_radius = 4e-3   # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # Example 1
    input_filename = f'example_ivp_cont_pml'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename + '_input.h5')
    input_args = {
        'pml_alpha': 0,
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_na_controlling_the_PML/input_1', input_file_full_path), \
        'Files do not match!'

    # Example 2
    input_args = {
        'pml_alpha': 1e6,
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_na_controlling_the_PML/input_2', input_file_full_path), \
        'Files do not match!'

    # Example 3
    input_args = {
        'pml_size': 2,
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_na_controlling_the_PML/input_3', input_file_full_path), \
        'Files do not match!'

    # Example 4
    input_args = {
        'pml_inside': False,
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_na_controlling_the_PML/input_4', input_file_full_path), \
        'Files do not match!'
