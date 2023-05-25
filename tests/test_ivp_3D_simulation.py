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
import setup_test
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kSensor
from kwave.options import SimulationExecutionOptions, SimulationOptions
from kwave.utils.mapgen import make_ball
from tests.diff_utils import compare_against_ref


def test_ivp_3D_simulation():
    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    grid_size = Vector([64, 64, 64])  # [grid points]
    grid_spacing = 1e-4 * Vector([1, 1, 1])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500 * np.ones(grid_size),   # [m/s]
        density=1000 * np.ones(grid_size)        # [kg/m^3]
    )
    medium.sound_speed[0:grid_size.x//2, :, :] = 1800
    medium.density[:, grid_size.y//4-1:, :]    = 1200

    # create initial pressure distribution using make_ball
    ball_magnitude = 10    # [Pa]
    ball_location = Vector([38, 32, 32])  # [grid points]
    ball_radius = 5        # [grid points]
    ball_1 = ball_magnitude * make_ball(grid_size, ball_location, ball_radius)

    ball_magnitude = 10    # [Pa]
    ball_location = Vector([20, 20, 20])  # [grid points]
    ball_radius = 3        # [grid points]
    ball_2 = ball_magnitude * make_ball(grid_size, ball_location, ball_radius)

    source = kSource()
    source.p0 = ball_1 + ball_2

    # define a series of Cartesian points to collect the data
    x = np.arange(-22, 23, 2) * grid_spacing.x            # [m]
    y = 22 * grid_spacing.y * np.ones_like(x)             # [m]xw
    z = np.arange(-22, 23, 2) * grid_spacing.z            # [m]
    sensor_mask = np.vstack([x, y, z])
    sensor = kSensor(sensor_mask)

    # input arguments
    input_filename = f'example_ivp_3D_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        data_cast='single',
        cart_interp='nearest',
        save_to_disk=True,
        input_filename=input_filename,
        save_to_disk_exit=True,
        data_path=pathname
    )
    # run the simulation
    kspaceFirstOrder3DC(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref(f'out_ivp_3D_simulation', input_file_full_path), 'Files do not match!'
