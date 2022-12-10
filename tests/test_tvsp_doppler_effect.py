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
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
from kwave.utils.filters import filter_time_series
from tests.diff_utils import compare_against_ref


def test_tvsp_doppler_effect():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 64            # number of grid points in the x (row) direction
    Ny = Nx*2          # number of grid points in the y (column) direction
    dy = 20e-3/Ny    	# grid point spacing in the y direction [m]
    dx = dy            # grid point spacing in the x direction [m]
    pml_size = 20      # [grid points]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # set the velocity of the moving source
    source_vel = 150               # [m/s]

    # set the relative x-position between the source and sensor
    source_sensor_x_distance = 5   # [grid points]

    # manually create the time array
    Nt = 4500
    dt = 20e-9                     # [s]
    kgrid.setTime(Nt, dt)

    # define a single time varying sinusoidal source
    source_freq = 0.75e6           # [MHz]
    source_mag = 3                 # [Pa]
    source_pressure = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source_pressure = filter_time_series(kgrid, medium, source_pressure)

    # define a line of source points
    source = kSource()
    source_x_pos = 5               # [grid points]
    source.p_mask = np.zeros((Nx, Ny))
    source.p_mask[-pml_size-source_x_pos - 1, pml_size:-pml_size] = 1

    # preallocate an empty pressure source matrix
    num_source_positions = int(source.p_mask.sum())
    source.p = np.zeros((num_source_positions, kgrid.t_array.size))

    # move the source along the source mask by interpolating the pressure series between the source elements
    sensor_index = 1
    t_index = 0
    while t_index < kgrid.t_array.size and sensor_index < num_source_positions - 1:

        # check if the source has moved to the next pair of grid points
        if kgrid.t_array[0, t_index] > (sensor_index*dy/source_vel):
            sensor_index = sensor_index + 1

        # calculate the position of source in between the two current grid
        # points
        exact_pos = source_vel * kgrid.t_array[0, t_index]
        discrete_pos = sensor_index * dy
        pos_ratio = (discrete_pos - exact_pos) / dy

        # update the pressure at the two current grid points using linear interpolation
        source.p[sensor_index - 1, t_index] = pos_ratio * source_pressure[0, t_index]
        source.p[sensor_index, t_index] = (1 - pos_ratio) * source_pressure[0, t_index]

        # update the time index
        t_index = t_index + 1

    # define a single sensor point
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[-pml_size-source_x_pos-source_sensor_x_distance-1, Ny//2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # run the simulation
    input_args = {
        'SaveToDisk': os.path.join(pathname, f'example_input.h5'),
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
    assert compare_against_ref(f'out_tvsp_doppler_effect', input_args['SaveToDisk']), \
        'Files do not match!'
