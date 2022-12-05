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
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from tests.diff_utils import compare_against_ref


def test_ivp_comparison_modelling_functions():
    # pathname for the input and output files
    pathname = gettempdir()

    example_number = 1
    # 1: non-absorbing medium, no absorbing boundary layer
    # 2: non-absorbing medium, using PML and ExpandGrid
    # 3: absorbing medium, no absorbing boundary layer
    # 4: absorbing medium, using PML and ExpandGrid

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
    if example_number > 2:
        medium.alpha_power = 1.5   # [dB/(MHz^y cm)]
        medium.alpha_coeff = 0.75  # [dB/(MHz^y cm)]

    # create the time array
    t_end = 6e-6
    kgrid.makeTime(medium.sound_speed, 0.3, t_end)

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

    # define a centered circular sensor pushed right to the edge of the grid
    sensor_radius = 6.3e-3   # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # convert the cartesian sensor mask to a binary sensor mask
    sensor.mask, _, _ = cart2grid(kgrid, sensor.mask)

    # run the simulation using the first order code
    # run the first simulation
    input_filename = f'example_ivp_comp'
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
    assert compare_against_ref(f'out_ivp_comparison_modelling_functions/input_1', input_file_full_path), \
        'Files do not match!'

    # run the simulation using the second order code
    # save_args = {'SaveToDisk', [tempdir 'example_input_2.h5'], 'SaveToDiskExit', true}
    # kspaceSecondOrder(kgrid, medium, source, sensor, 'ExpandGrid', false, save_args{:})

    # run the simulation using the first order code
    input_args = {
        'PMLInside': False,
        'SaveToDisk': input_file_full_path,
        'SaveToDiskExit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_ivp_comparison_modelling_functions/input_2', input_file_full_path), \
        'Files do not match!'

    # run the simulation using the second order code
    # save_args = {'SaveToDisk', [tempdir 'example_input_4.h5'], 'SaveToDiskExit', true}
    # kspaceSecondOrder(kgrid, medium, source, sensor, 'ExpandGrid', true, save_args{:})
