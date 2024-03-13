"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
import os
from tempfile import gettempdir

import numpy as np

# noinspection PyUnresolvedReferences
import setup_test  # noqa: F401
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrderAS import kspaceFirstOrderASC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_disc
from tests.diff_utils import compare_against_ref


def test_ivp_axisymmetric_simulation():
    # create the computational grid
    grid_size = Vector([128, 64])  # [grid points]
    grid_spacing = 0.1e-3 * Vector([1, 1])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500 * np.ones(grid_size),  # [m/s]
        density=1000 * np.ones(grid_size),  # [kg/m^3]
    )
    medium.sound_speed[grid_size.x // 2 - 1 :, :] = 1800  # [m/s]
    medium.density[grid_size.x // 2 - 1 :, :] = 1200  # [kg/m^3]

    # create initial pressure distribution in the shape of a disc - this is
    # generated on a 2D grid that is doubled in size in the radial (y)
    # direction, and then trimmed so that only half the disc is retained
    source = kSource()
    source.p0 = 10 * make_disc(Vector([grid_size.x, 2 * grid_size.y]), Vector([grid_size.x // 4 + 8, grid_size.y + 1]), 5)
    source.p0 = source.p0[:, grid_size.y :]

    # define a Cartesian sensor mask with points in the shape of a circle
    # REPLACED BY FARID cartesian mask with binary. Otherwise, SaveToDisk doesn't work.
    # sensor.mask = makeCartCircle(40 * dx, 50);
    sensor_mask = np.zeros(grid_size)
    sensor_mask[0, :] = 1
    sensor = kSensor(sensor_mask)

    # remove points from sensor mask where y < 0
    sensor.mask[:, sensor.mask[1, :] < 0] = np.nan

    # set the input settings
    input_filename = "example_ivp_axisymmetric_input.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(save_to_disk=True, input_filename=input_filename, save_to_disk_exit=True, data_path=pathname)

    # run the simulation
    kspaceFirstOrderASC(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )

    assert compare_against_ref("out_ivp_axisymmetric_simulation", input_file_full_path), "Files do not match!"
