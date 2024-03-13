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
import setup_test  # noqa: F401
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


def test_tvsp_homogeneous_medium_monopole():
    # create the computational grid
    grid_size_points = Vector([128, 128])  # [grid points]
    grid_size_meters = Vector([50e-3, 50e-3])  # [m]
    grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a single source point
    source = kSource()
    source.p_mask = np.zeros(grid_size_points)
    source.p_mask[-grid_size_points.x // 4 - 1, grid_size_points.y // 2 - 1] = 1

    # define a time varying sinusoidal source
    source_freq = 0.25e6  # [Hz]
    source_mag = 2  # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    # define a single sensor point
    sensor_mask = np.zeros(grid_size_points)
    sensor_mask[grid_size_points.x // 4 - 1, grid_size_points.y // 2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # define the acoustic parameters to record
    sensor.record = ["p", "p_final"]

    # set the input settings
    input_filename = "example_tvsp_homo_input.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(save_to_disk=True, input_filename=input_filename, data_path=pathname, save_to_disk_exit=True)
    # run the simulation
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    assert compare_against_ref("out_tvsp_homogeneous_medium_monopole", input_file_full_path), "Files do not match!"
