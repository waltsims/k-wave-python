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

import numpy as np

# noinspection PyUnresolvedReferences
import setup_test
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import filter_time_series
from tests.diff_utils import compare_against_ref


def test_tvsp_doppler_effect():
    # create the computational grid
    pml_size = Vector([20, 20])  # [grid points]
    grid_size_points = Vector([64, 128])  # [grid points]
    grid_size_meters = Vector([10e-3, 20e-3])  # [m]
    grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

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
    source.p_mask = np.zeros(grid_size_points)
    source.p_mask[-pml_size.x-source_x_pos - 1, pml_size.y:-pml_size.y] = 1

    # preallocate an empty pressure source matrix
    num_source_positions = int(source.p_mask.sum())
    source.p = np.zeros((num_source_positions, kgrid.t_array.size))

    # move the source along the source mask by interpolating the pressure series between the source elements
    sensor_index = 1
    t_index = 0
    while t_index < kgrid.t_array.size and sensor_index < num_source_positions - 1:

        # check if the source has moved to the next pair of grid points
        if kgrid.t_array[0, t_index] > (sensor_index*grid_spacing_meters.y/source_vel):
            sensor_index = sensor_index + 1

        # calculate the position of source in between the two current grid
        # points
        exact_pos = source_vel * kgrid.t_array[0, t_index]
        discrete_pos = sensor_index * grid_spacing_meters.y
        pos_ratio = (discrete_pos - exact_pos) / grid_spacing_meters.y

        # update the pressure at the two current grid points using linear interpolation
        source.p[sensor_index - 1, t_index] = pos_ratio * source_pressure[0, t_index]
        source.p[sensor_index, t_index] = (1 - pos_ratio) * source_pressure[0, t_index]

        # update the time index
        t_index = t_index + 1

    # define a single sensor point
    sensor_mask = np.zeros(grid_size_points)
    sensor_mask[-pml_size-source_x_pos-source_sensor_x_distance-1, grid_size_points.y//2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # run the simulation
    input_filename = f'example_doppler_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        save_to_disk=True,
        input_filename=input_filename,
        data_path=pathname,
        save_to_disk_exit=True
    )
    # run the simulation
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref(f'out_tvsp_doppler_effect', input_file_full_path), \
        'Files do not match!'
