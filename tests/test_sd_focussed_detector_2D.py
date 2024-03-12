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
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import kSensor
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_disc, make_circle
from tests.diff_utils import compare_against_ref


def test_sd_focussed_detector_2d():
    # create the computational grid
    grid_size = Vector([180, 180])  # [grid points]
    grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define a sensor as part of a circle centred on the grid
    sensor_radius = 65  # [grid points]
    arc_angle = np.pi  # [rad]
    sensor_mask = make_circle(grid_size, grid_size // 2 + 1, sensor_radius, arc_angle)
    sensor = kSensor(sensor_mask)

    # define the array of temporal points
    t_end = 11e-6  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # place a disc-shaped source near the focus of the detector
    source = kSource()
    source.p0 = 2 * make_disc(grid_size, grid_size / 2, 4)

    # run the first simulation
    input_filename = "example_sd_focused_2d_input.h5"
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

    assert compare_against_ref("out_sd_focussed_detector_2D", input_file_full_path), "Files do not match!"
