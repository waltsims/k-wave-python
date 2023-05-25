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
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.utils.mapgen import make_ball, make_disc
from tests.diff_utils import compare_against_ref


def test_ivp_photoacoustic_waveforms():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SETTINGS
    # =========================================================================

    # size of the computational grid
    Nx = 64    # number of grid points in the x (row) direction
    x = 1e-3   # size of the domain in the x direction [m]
    dx = x/Nx  # grid point spacing in the x direction [m]

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # size of the initial pressure distribution
    source_radius = 2              # [grid points]

    # distance between the centre of the source and the sensor
    source_sensor_distance = 10    # [grid points]

    # time array
    dt = 2e-9                      # [s]
    t_end = 300e-9                 # [s]

    # =========================================================================
    # ONE DIMENSIONAL SIMULATION
    # =========================================================================
    #
    # # create the computational grid
    # kgrid = kWaveGrid(Nx, dx)
    #
    # # create the time array
    # kgrid.setTime(round(t_end / dt) + 1, dt)
    #
    # # create initial pressure distribution
    # source.p0 = zeros(Nx, 1)
    # source.p0(Nx/2 - source_radius:Nx/2 + source_radius) = 1
    #
    # # define a single sensor point
    # sensor.mask = zeros(Nx, 1)
    # sensor.mask(Nx/2 + source_sensor_distance) = 1
    #
    # # run the simulation
    # sensor_data_1D = kspaceFirstOrder1D(kgrid, medium, source, sensor, input_args{:})

    # =========================================================================
    # TWO DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    grid_size = Vector(2 * [Nx])
    grid_spacing = Vector(2 * [dx])
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    # create initial pressure distribution
    source = kSource()
    source.p0 = make_disc(grid_size, grid_size / 2, source_radius)

    # define a single sensor point
    sensor_mask = np.zeros(grid_size)
    sensor_mask[grid_size.x//2 - source_sensor_distance - 1, grid_size.y//2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # run the simulation
    input_filename = f'example_ivp_pa_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        data_cast='single',
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

    assert compare_against_ref(f'out_ivp_photoacoustic_waveforms/input_1', input_file_full_path), \
        'Files do not match!'

    # =========================================================================
    # THREE DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    grid_size = Vector(3 * [Nx])
    grid_spacing = Vector(3 * [dx])
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    # create initial pressure distribution
    source.p0 = make_ball(grid_size, grid_size / 2, source_radius)

    # define a single sensor point
    sensor.mask = np.zeros(grid_size)
    sensor.mask[grid_size.x//2 - source_sensor_distance - 1, grid_size.y//2 - 1, grid_size.z//2 - 1] = 1

    # run the simulation
    simulation_options = SimulationOptions(
        data_cast='single',
        save_to_disk=True,
        input_filename=input_filename,
        save_to_disk_exit=True,
        data_path=pathname
    )
    # run the simulation
    kspaceFirstOrder3DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref(f'out_ivp_photoacoustic_waveforms/input_2', input_file_full_path), \
        'Files do not match!'
