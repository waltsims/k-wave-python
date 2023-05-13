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

from kwave.data import Vector
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.utils.filters import smooth
from kwave.utils.io import load_image
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matrix import resize
from tests.diff_utils import compare_against_ref


def test_pr_2d_tr_circular_sensor():
    # load the initial pressure distribution from an image and scale
    p0_magnitude = 2
    p0 = p0_magnitude * load_image('tests/EXAMPLE_source_two.bmp', is_gray=True)

    # assign the grid size and create the computational grid
    PML_size = 20  # size of the PML in grid points
    Nx = 256 - 2 * PML_size  # number of grid points in the x direction
    Ny = 256 - 2 * PML_size  # number of grid points in the y direction
    x = 10e-3  # total grid size [m]
    y = 10e-3  # total grid size [m]
    dx = x / Nx                # grid point spacing in the x direction [m]
    dy = y / Ny                # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # resize the input image to the desired number of grid points
    p0 = resize(p0, [Nx, Ny])

    # smooth the initial pressure distribution and restore the magnitude
    p0 = smooth(p0, True)

    # assign to the source structure
    source = kSource()
    source.p0 = p0

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define a centered Cartesian circular sensor
    sensor_radius = 4.5e-3              # [m]
    sensor_angle = 3 * np.pi / 2        # [rad]
    sensor_pos = Vector([0, 0])         # [m]
    num_sensor_points = 70
    cart_sensor_mask = make_cart_circle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle)

    # assign to sensor structure
    sensor_mask = cart_sensor_mask
    sensor = kSensor(sensor_mask)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # set the input settings
    input_filename = f'example_tr_circ_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_inside=False,
        smooth_p0=False,
        save_to_disk=True,
        input_filename=input_filename,
        data_path=pathname,
        save_to_disk_exit=True
    )
    # run the simulation
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref(f'out_pr_2D_TR_circular_sensor', input_file_full_path), 'Files do not match!'
