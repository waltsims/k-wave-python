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

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import kSensor
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_disc
from tests.diff_utils import compare_against_ref


def test_ivp_opposing_corners_sensor_mask():
    # create the computational grid
    grid_size = Vector([128, 128])  # [grid points]
    grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5  # [Pa]
    disc_pos = Vector([50, 50])  # [grid points]
    disc_radius = 8  # [grid points]
    disc_1 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    disc_magnitude = 3  # [Pa]
    disc_pos = Vector([80, 60])  # [grid points]
    disc_radius = 5  # [grid points]
    disc_2 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # define the first rectangular sensor region by specifying the location of
    # opposing corners
    rect1_x_start = 25
    rect1_y_start = 31
    rect1_x_end = 30
    rect1_y_end = 50

    # define the second rectangular sensor region by specifying the location of
    # opposing corners
    rect2_x_start = 71
    rect2_y_start = 81
    rect2_x_end = 80
    rect2_y_end = 90

    # assign the list of opposing corners to the sensor mask
    sensor_mask = np.array(
        [
            [rect1_x_start, rect1_y_start, rect1_x_end, rect1_y_end],
            [rect2_x_start, rect2_y_start, rect2_x_end, rect2_y_end],
        ]
    ).T
    sensor = kSensor(sensor_mask)

    # input arguments
    input_filename = "example_ivp_corn_input.h5"
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

    assert compare_against_ref("out_ivp_opposing_corners_sensor_mask", input_file_full_path), "Files do not match!"
