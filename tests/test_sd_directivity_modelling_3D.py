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
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.conversion import cart2grid
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matlab import matlab_find, unflatten_matlab_mask
from tests.diff_utils import compare_against_ref


def test_sd_directivity_modelling_3D():
    # create the computational grid
    pml_size = Vector([10, 10, 10])  # [grid points]
    grid_size_points = Vector([64, 64, 64])  # [grid points]
    grid_size_meters = Vector([100e-3, 100e-3, 100e-3])  # [m]
    grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a large area detector
    sz = 16  # [grid points]
    sensor_mask = np.zeros(grid_size_points)
    sensor_mask[
        grid_size_points.x // 2,
        (grid_size_points.y // 2 - sz // 2) : (grid_size_points.y // 2 + sz // 2 + 1),
        (grid_size_points.z // 2 - sz // 2) : (grid_size_points.z // 2 + sz // 2 + 1),
    ] = 1
    sensor = kSensor(sensor_mask)

    # define equally spaced point sources lying on a circle centred at the
    # centre of the detector face
    radius = 20  # [grid points]
    points = 11
    circle = make_cart_circle(radius * grid_spacing_meters.x, points, Vector([0, 0]), np.pi)
    circle = np.vstack([circle, np.zeros((1, points))])

    # find the binary sensor mask most closely corresponding to the cartesian
    # coordinates from makeCartCircle
    circle3D, _, _ = cart2grid(kgrid, circle)

    # find the indices of the sources in the binary source mask
    source_positions = matlab_find(circle3D, val=1, mode="eq")

    # define a time varying sinusoidal source
    source = kSource()
    source_freq = 0.25e6  # [Hz]
    source_mag = 1  # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    # pre-allocate array for storing the output time series
    single_element_data = np.zeros((kgrid.Nt, points))  # noqa: F841

    # run a simulation for each of these sources to see the effect that the
    # angle from the detector has on the measured signal
    for source_loop in range(points):
        # select a point source
        source.p_mask = np.zeros(grid_size_points)
        source.p_mask[unflatten_matlab_mask(source.p_mask, source_positions[source_loop] - 1)] = 1

        # run the simulation
        input_filename = f"example_input_{source_loop + 1}_input.h5"
        pathname = gettempdir()
        input_file_full_path = os.path.join(pathname, input_filename)
        simulation_options = SimulationOptions(
            pml_size=pml_size, save_to_disk=True, input_filename=input_filename, save_to_disk_exit=True, data_path=pathname
        )
        # run the simulation
        kspaceFirstOrder3DC(
            medium=medium,
            kgrid=kgrid,
            source=deepcopy(source),
            sensor=sensor,
            simulation_options=simulation_options,
            execution_options=SimulationExecutionOptions(),
        )

        assert compare_against_ref(f"out_sd_directivity_modelling_3D/input_{source_loop + 1}", input_file_full_path), "Files do not match!"
