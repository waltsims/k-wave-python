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
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.io import load_image
from kwave.utils.mapgen import make_cart_circle
from tests.diff_utils import compare_against_ref


def test_ivp_loading_external_image():
    # load the initial pressure distribution from an image and scale the
    # magnitude
    p0_magnitude = 3
    p0 = p0_magnitude * load_image("tests/EXAMPLE_source_one.png", True)

    # create the computational grid
    grid_size = Vector([128, 128])  # [grid points]
    grid_spacing = 0.1e-3 * Vector([1, 1])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # resize the image to match the size of the computational grid and assign
    # to the source input structure
    source = kSource()
    source.p0 = np.reshape(p0, grid_size)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # define a centered circular sensor
    sensor_radius = 4e-3  # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # run the first simulation
    input_filename = "example_ivp_ext_img_input.h5"
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

    assert compare_against_ref("out_ivp_loading_external_image", input_file_full_path), "Files do not match!"
