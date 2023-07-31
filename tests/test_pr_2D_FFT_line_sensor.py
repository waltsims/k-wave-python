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
import setup_test  # noqa: F401
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc
from tests.diff_utils import compare_against_ref


def test_pr_2D_FFT_line_sensor():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    pml_size = Vector([20, 20])               # size of the PML in grid points
    grid_size = Vector([128, 256]) - 2 * pml_size  # [grid points]
    grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5  # [Pa]
    disc_pos = Vector([60, 140])  # [grid points]
    disc_radius = 5    # [grid points]
    disc_2 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    disc_pos = Vector([30, 110])  # [grid points]
    disc_radius = 8    # [grid points]
    disc_1 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # smooth the initial pressure distribution and restore the magnitude
    source.p0 = smooth(source.p0, True)

    # define a binary line sensor
    sensor_mask = np.zeros(grid_size)
    sensor_mask[0, :] = 1
    sensor = kSensor(sensor_mask)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # set the input arguements: force the PML to be outside the computational
    # grid switch off p0 smoothing within kspaceFirstOrder2D
    input_filename = 'example_fft_line_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)

    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=pml_size,
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
    assert compare_against_ref('out_pr_2D_FFT_line_sensor', input_file_full_path), 'Files do not match!'
