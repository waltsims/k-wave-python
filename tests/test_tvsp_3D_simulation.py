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
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kSensor
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import filter_time_series
from tests.diff_utils import compare_against_ref


def test_tvsp_3D_simulation():
    # create the computational grid
    grid_size = Vector([64, 64, 64])  # [grid points]
    grid_spacing = 1e-4 * Vector([1, 1, 1])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500 * np.ones(grid_size), density=1000 * np.ones(grid_size))
    medium.sound_speed[0 : grid_size.x // 2, :, :] = 1800  # [m/s]
    medium.density[:, grid_size.y // 4 - 1 :, :] = 1200  # [kg/m^3]

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a square source element
    source = kSource()
    source_radius = 5  # [grid points]
    source.p_mask = np.zeros(grid_size)
    source.p_mask[
        grid_size.x // 4 - 1,
        grid_size.y // 2 - source_radius - 1 : grid_size.y // 2 + source_radius,
        grid_size.z // 2 - source_radius - 1 : grid_size.z // 2 + source_radius,
    ] = 1

    # define a time varying sinusoidal source
    source_freq = 2e6  # [Hz]
    source_mag = 1  # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    # define a series of Cartesian points to collect the data
    y = np.arange(-20, 21, 2) * grid_spacing.y  # [m]
    z = np.arange(-20, 21, 2) * grid_spacing.z  # [m]
    x = 20 * grid_spacing.x * np.ones(z.shape)  # [m]
    sensor_mask = np.array([x, y, z])
    sensor = kSensor(sensor_mask)

    # define the field parameters to record
    sensor.record = ["p", "p_final"]
    input_filename = "example_tvsp_3d_input.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    # input arguments
    simulation_options = SimulationOptions(
        data_cast="single",
        cart_interp="nearest",
        save_to_disk=True,
        input_filename=input_filename,
        save_to_disk_exit=True,
        data_path=pathname,
    )
    # run the simulation
    kspaceFirstOrder3DC(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    assert compare_against_ref("out_tvsp_3D_simulation", input_file_full_path), "Files do not match!"
