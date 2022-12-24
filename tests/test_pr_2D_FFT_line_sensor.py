"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
import os
from tempfile import gettempdir

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.ktransducer import *
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
    PML_size = 20               # size of the PML in grid points
    Nx = 128 - 2 * PML_size     # number of grid points in the x (row) direction
    Ny = 256 - 2 * PML_size     # number of grid points in the y (column) direction
    dx = 0.1e-3                 # grid point spacing in the x direction [m]
    dy = 0.1e-3                 # grid point spacing in the y direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5 # [Pa]
    disc_x_pos = 60    # [grid points]
    disc_y_pos = 140  	# [grid points]
    disc_radius = 5    # [grid points]
    disc_2 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    disc_x_pos = 30    # [grid points]
    disc_y_pos = 110 	# [grid points]
    disc_radius = 8    # [grid points]
    disc_1 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # smooth the initial pressure distribution and restore the magnitude
    source.p0 = smooth(source.p0, True)

    # define a binary line sensor
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[0, :] = 1
    sensor = kSensor(sensor_mask)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # set the input arguements: force the PML to be outside the computational
    # grid switch off p0 smoothing within kspaceFirstOrder2D
    input_filename = f'example_fft_line_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    input_args = {
        'pml_inside': False,
        'pml_size': PML_size,
        'smooth_p0': False,
        'save_to_disk': True,
        'input_filename': input_filename,
        'data_path': pathname,
        'save_to_disk_exit': True
    }

    # run the simulation
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_pr_2D_FFT_line_sensor', input_file_full_path), 'Files do not match!'
