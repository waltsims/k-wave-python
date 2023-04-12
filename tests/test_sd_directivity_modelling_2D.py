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
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.utils.conversion import cart2grid
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matlab import matlab_find, unflatten_matlab_mask
from tests.diff_utils import compare_against_ref


def test_sd_directivity_modelling_2D():
    # create the computational grid
    Nx = 128            # number of grid points in the x (row) direction
    Ny = 128            # number of grid points in the y (column) direction
    dx = 50e-3/Nx       # grid point spacing in the x direction [m]
    dy = dx             # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define the array of time points [s]
    Nt = 350
    dt = 7e-8            # [s]
    kgrid.setTime(Nt, dt)

    # define a large area detector
    sz = 20              # [grid points]
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[Nx//2, (Ny//2 - sz//2):(Ny//2 + sz//2) + 1] = 1
    sensor = kSensor(sensor_mask)

    # define equally spaced point sources lying on a circle centred at the
    # centre of the detector face
    radius = 30    # [grid points]
    points = 11
    circle = make_cart_circle(radius * dx, points, [0, 0], np.pi)

    # find the binary sensor mask most closely corresponding to the Cartesian
    # coordinates from makeCartCircle
    circle, _, _ = cart2grid(kgrid, circle)

    # find the indices of the sources in the binary source mask
    source_positions = matlab_find(circle, val=1, mode='eq')

    # define a time varying sinusoidal source
    source = kSource()
    source_freq = 0.25e6    # [Hz]
    source_mag = 1          # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    # pre-allocate array for storing the output time series
    single_element_data = np.zeros((Nt, points))


    # run a simulation for each of these sources to see the effect that the
    # angle from the detector has on the measured signal
    for source_loop in range(points):

        # select a point source
        source.p_mask = np.zeros((Nx, Ny))
        source.p_mask[unflatten_matlab_mask(source.p_mask, source_positions[source_loop] - 1)] = 1

        # run the simulation

        input_filename = f'example_input_{source_loop + 1}_input.h5'
        pathname = gettempdir()
        input_file_full_path = os.path.join(pathname, input_filename)
        input_args = SimulationOptions(
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
            sensor=deepcopy(sensor),
            simulation_options=input_args,
            execution_options=SimulationExecutionOptions()
        )

        assert compare_against_ref(f'out_sd_directivity_modelling_2D/input_{source_loop + 1}', input_file_full_path), \
            'Files do not match!'
