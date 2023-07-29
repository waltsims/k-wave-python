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

from kwave.data import Vector
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test  # noqa: F401
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.utils.mapgen import make_disc, make_cart_circle
from tests.diff_utils import compare_against_ref


def test_ivp_sensor_frequency_response():
    # create the computational grid
    grid_size = Vector([128, 128])  # [grid points]
    grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # define the time array
    kgrid.makeTime(medium.sound_speed)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5 # [Pa]
    disc_pos = Vector([50, 50])  # [grid points]
    disc_radius = 8    # [grid points]
    disc_1 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    disc_magnitude = 3 # [Pa]
    disc_pos = Vector([80, 60])  # [grid points]
    disc_radius = 5    # [grid points]
    disc_2 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # define a centered circular sensor
    sensor_radius = 4e-3   # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # run the simulation
    input_filename = 'example_ivp_sfr_input.h5'
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
    assert compare_against_ref('out_ivp_sensor_frequency_response/input_1', input_file_full_path), \
        'Files do not match!'

    # define the frequency response of the sensor elements
    center_freq = 3e6      # [Hz]
    bandwidth = 80         # [#]
    sensor.frequency_response = np.array([center_freq, bandwidth])

    # re-run the simulation
    simulation_options = SimulationOptions(
        save_to_disk=True,
        input_filename=input_filename,
        data_path=pathname,
        save_to_disk_exit=True
    )
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref('out_ivp_sensor_frequency_response/input_2', input_file_full_path), \
        'Files do not match!'
