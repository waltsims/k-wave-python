"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
# noinspection PyUnresolvedReferences
import setup_test
import os
from tempfile import gettempdir

from kwave_py.ksource import kSource
from kwave_py.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave_py.utils.maputils import makeBall
from kwave_py.utils import dotdict
from kwave_py.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave_py.kmedium import kWaveMedium


def test_ivp_3D_simulation():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 64             # number of grid points in the x direction
    Ny = 64             # number of grid points in the y direction
    Nz = 64             # number of grid points in the z direction
    dx = 0.1e-3         # grid point spacing in the x direction [m]
    dy = 0.1e-3         # grid point spacing in the y direction [m]
    dz = 0.1e-3         # grid point spacing in the z direction [m]
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500 * np.ones((Nx, Ny, Nz)),   # [m/s]
        density=1000 * np.ones((Nx, Ny, Nz))        # [kg/m^3]
    )
    medium.sound_speed[0:Nx//2, :, :] = 1800
    medium.density[:, Ny//4-1:, :]    = 1200

    # create initial pressure distribution using makeBall
    ball_magnitude = 10    # [Pa]
    ball_x_pos = 38        # [grid points]
    ball_y_pos = 32        # [grid points]
    ball_z_pos = 32        # [grid points]
    ball_radius = 5        # [grid points]
    ball_1 = ball_magnitude * makeBall(Nx, Ny, Nz, ball_x_pos, ball_y_pos, ball_z_pos, ball_radius)

    ball_magnitude = 10    # [Pa]
    ball_x_pos = 20        # [grid points]
    ball_y_pos = 20        # [grid points]
    ball_z_pos = 20        # [grid points]
    ball_radius = 3        # [grid points]
    ball_2 = ball_magnitude * makeBall(Nx, Ny, Nz, ball_x_pos, ball_y_pos, ball_z_pos, ball_radius)

    source = kSource()
    source.p0 = ball_1 + ball_2

    # define a series of Cartesian points to collect the data
    x = np.arange(-22, 23, 2) * dx            # [m]
    y = 22 * dy * np.ones_like(x)             # [m]xw
    z = np.arange(-22, 23, 2) * dz            # [m]
    sensor_mask = np.vstack([x, y, z])
    sensor = kSensor(sensor_mask)

    # input arguments
    input_filename  = f'example_input.h5'
    input_file_full_path = os.path.join(pathname, input_filename)
    input_args = {
        'DataCast': 'single',
        'CartInterp': 'nearest',
        'SaveToDisk': input_file_full_path,
        'SaveToDiskExit': True
    }

    # run the simulation
    kspaceFirstOrder3DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })

    assert compare_against_ref(f'out_ivp_3D_simulation', input_file_full_path), 'Files do not match!'
