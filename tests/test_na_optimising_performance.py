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

from kwave.data import Vector
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test  # noqa: F401
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import kSensor
from kwave.utils.conversion import cart2grid
from kwave.utils.io import load_image
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matrix import resize
from tests.diff_utils import compare_against_ref


def test_na_optimising_performance():
    # change scale to 2 to increase the computational time
    scale = 1

    # assign the grid size and create the computational grid
    grid_size_points = Vector([256, 256])  # [grid points]
    grid_size_meters = Vector([10e-3, 10e-3])  # [m]
    grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # load the initial pressure distribution from an image and scale
    source = kSource()
    p0_magnitude = 2           # [Pa]
    source.p0 = p0_magnitude * load_image('tests/EXAMPLE_source_two.bmp', is_gray=True)
    source.p0 = resize(source.p0, grid_size_points)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define a centered Cartesian circular sensor
    sensor_radius = 4.5e-3     # [m]
    num_sensor_points = 100
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # run the simulation

    # 1: default input options
    input_filename = 'example_opt_perf_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
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
        sensor=deepcopy(sensor),
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref('out_na_optimising_performance/input_1', input_file_full_path), \
        'Files do not match!'

    # 2: nearest neighbour Cartesian interpolation and plotting switched off
    simulation_options = SimulationOptions(
        save_to_disk=True,
        input_filename=input_filename,
        data_path=pathname,
        save_to_disk_exit=True
    )
    # convert Cartesian sensor mask to binary mask
    sensor.mask, _, _ = cart2grid(kgrid, sensor.mask)
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref('out_na_optimising_performance/input_2', input_file_full_path), \
        'Files do not match!'

    # 3: as above with 'data_cast' set to 'single'
    # set input arguments
    simulation_options = SimulationOptions(
        data_cast='single',
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
    assert compare_against_ref('out_na_optimising_performance/input_3', input_file_full_path), \
        'Files do not match!'
