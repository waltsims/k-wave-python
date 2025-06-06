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
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_cart_circle, make_disc
from tests.diff_utils import compare_against_ref


def test_na_controlling_the_pml():
    # modify this parameter to run the different examples
    example_number = 1  # noqa: F841
    # 1: PML with no absorption
    # 2: PML with the absorption value set too high
    # 3: partially effective PML
    # 4: PML set to be outside the computational domain

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    grid_size = Vector([128, 128])  # [grid points]
    grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

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

    # define a centered circular sensor
    sensor_radius = 4e-3  # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)

    # Example 1
    input_filename = "input_1.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_alpha=0, save_to_disk=True, input_filename=input_filename, data_path=pathname, save_to_disk_exit=True
    )
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    assert compare_against_ref("out_na_controlling_the_PML/input_1", input_file_full_path), "Files do not match!"

    # Example 2
    simulation_options = SimulationOptions(
        pml_alpha=1e6, save_to_disk=True, input_filename=input_filename, data_path=pathname, save_to_disk_exit=True
    )
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    assert compare_against_ref("out_na_controlling_the_PML/input_2", input_file_full_path), "Files do not match!"

    # Example 3
    simulation_options = SimulationOptions(
        pml_size=2, save_to_disk=True, input_filename=input_filename, data_path=pathname, save_to_disk_exit=True
    )
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    assert compare_against_ref("out_na_controlling_the_PML/input_3", input_file_full_path), "Files do not match!"

    # Example 4
    simulation_options = SimulationOptions(
        pml_inside=False, save_to_disk=True, input_filename=input_filename, data_path=pathname, save_to_disk_exit=True
    )
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    assert compare_against_ref("out_na_controlling_the_PML/input_4", input_file_full_path), "Files do not match!"
