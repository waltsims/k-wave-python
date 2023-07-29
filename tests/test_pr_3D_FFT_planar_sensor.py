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
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test  # noqa: F401
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kSensor
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball
from tests.diff_utils import compare_against_ref


def test_pr_3D_FFT_planar_sensor():
    # change scale to 2 to reproduce the higher resolution figures used in the help file
    scale = 1

    # create the computational grid
    PML_size = 10                   # size of the PML in grid points
    Nx = 32 * scale - 2 * PML_size  # number of grid points in the x direction
    Ny = 64 * scale - 2 * PML_size  # number of grid points in the y direction
    Nz = 64 * scale - 2 * PML_size  # number of grid points in the z direction
    dx = 0.2e-3 / scale             # grid point spacing in the x direction [m]
    dy = 0.2e-3 / scale             # grid point spacing in the y direction [m]
    dz = 0.2e-3 / scale             # grid point spacing in the z direction [m]
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create initial pressure distribution using make_ball
    ball_magnitude = 10         # [Pa]
    ball_radius = 3 * scale     # [grid points]
    p0 = ball_magnitude * make_ball(Nx, Ny, Nz, Nx / 2, Ny / 2, Nz / 2, ball_radius)

    # smooth the initial pressure distribution and restore the magnitude
    source = kSource()
    source.p0 = smooth(p0, True)

    # define a binary planar sensor
    sensor_mask = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz))
    sensor_mask[0, :, :] = 1
    sensor = kSensor(sensor_mask)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # set the input settings
    input_filename = 'example_3D_fft_planar_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,
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

    assert compare_against_ref('out_pr_3D_FFT_planar_sensor', input_file_full_path), 'Files do not match!'
