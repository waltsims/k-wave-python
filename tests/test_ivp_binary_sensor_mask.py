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
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import kSensor
from kwave.utils.mapgen import make_disc, make_circle
from tests.diff_utils import compare_against_ref


def test_ivp_binary_sensor_mask():
    # create the computational grid
    Nx = 128  # number of grid points in the x (row) direction
    Ny = 128  # number of grid points in the y (column) direction
    dx = 0.1e-3  # grid point spacing in the x direction [m]
    dy = 0.1e-3  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5  # [Pa]
    disc_x_pos = 50  # [grid points]
    disc_y_pos = 50  # [grid points]
    disc_radius = 8  # [grid points]
    disc_1 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    disc_magnitude = 3  # [Pa]
    disc_x_pos = 80  # [grid points]
    disc_y_pos = 60  # [grid points]
    disc_radius = 5  # [grid points]
    disc_2 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # define a binary sensor mask
    sensor_x_pos = Nx // 2  # [grid points]
    sensor_y_pos = Ny // 2  # [grid points]
    sensor_radius = Nx // 2 - 22  # [grid points]
    sensor_arc_angle = 3 * np.pi / 2  # [radians]
    sensor_mask = make_circle(Nx, Ny, sensor_x_pos, sensor_y_pos, sensor_radius, sensor_arc_angle)
    sensor = kSensor(sensor_mask)

    # run the simulation
    input_filename = f'example_ivp_bin_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    input_args = {
        'save_to_disk': True,
        'input_filename': input_filename,
        'data_path': pathname,
        'save_to_disk_exit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_ivp_binary_sensor_mask', input_file_full_path), \
        'Files do not match!'
