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
from kwave.ksensor import kSensorDirectivity, kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from tests.diff_utils import compare_against_ref


def test_sd_sensor_directivity_2D():
    # create the computational grid
    grid_size_points = Vector([64, 64])  # [grid points]
    grid_size_meters = Vector([1e-3, 1e-3])  # [m]
    grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define the array of temporal points
    t_end = 600e-9  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE THE DIRECTIONAL SENSOR ARRAY
    # =========================================================================

    # define a line of sensor points
    sensor_mask = np.zeros(grid_size_points)
    sensor_mask[23, 1:62:2] = 1
    sensor = kSensor(sensor_mask)

    # define the angle of max directivity for each sensor point:
    #    0             = max sensitivity in x direction (up/down)
    #    pi/2 or -pi/2 = max sensitivity in y direction (left/right)
    dir_angles = np.arange(-1, 1 + 1 / 15, 1 / 15)[:, None] * np.pi / 2

    # assign to the directivity mask
    directivity = kSensorDirectivity()
    directivity.angle = np.zeros(grid_size_points)
    directivity.angle[sensor.mask == 1] = np.squeeze(dir_angles)

    # define the directivity pattern
    directivity.pattern = "pressure"

    # define the directivity size
    directivity.size = 16 * kgrid.dx

    sensor.directivity = directivity

    # =========================================================================
    # SIMULATION AND VISUALISATION FOR AN INITIAL VALUE PROBLEM
    # =========================================================================

    # define the initial pressure distribution
    source = kSource()
    source_p0 = np.zeros(grid_size_points)
    source_p0[38:41, :] = 2
    source.p0 = source_p0

    # run the simulation
    input_filename = "example_def_tran_input.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_alpha=np.array([2, 0]), save_to_disk=True, input_filename=input_filename, data_path=pathname, save_to_disk_exit=True
    )
    # run the simulation
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )

    assert compare_against_ref("out_sd_sensor_directivity_2D", input_file_full_path, precision=6), "Files do not match!"
