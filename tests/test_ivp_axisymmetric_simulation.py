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
from kwave_py.kspaceFirstOrderAS import kspaceFirstOrderASC
from kwave_py.utils.maputils import makeDisc
from kwave_py.utils import dotdict
from kwave_py.ktransducer import *
from tests.diff_utils import compare_against_ref
from kwave_py.kmedium import kWaveMedium


def test_ivp_axisymmetric_simulation():
    # pathname for the input and output files
    pathname = gettempdir()

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 128            # number of grid points in the axial (x) direction
    Ny = 64             # number of grid points in the radial (y) direction
    dx = 0.1e-3         # grid point spacing in the axial (x) direction [m]
    dy = 0.1e-3         # grid point spacing in the radial (y) direction [m]
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500 * np.ones((Nx, Ny)),   # [m/s]
        density=1000 * np.ones((Nx, Ny))        # [kg/m^3]
    )
    medium.sound_speed[Nx//2-1:, :] = 1800      # [m/s]
    medium.density[Nx//2-1:, :]     = 1200      # [kg/m^3]

    # create initial pressure distribution in the shape of a disc - this is
    # generated on a 2D grid that is doubled in size in the radial (y)
    # direction, and then trimmed so that only half the disc is retained
    source = kSource()
    source.p0 = 10 * makeDisc(Nx, 2 * Ny, Nx//4 + 8, Ny + 1, 5)
    source.p0 = source.p0[:, Ny:]

    # define a Cartesian sensor mask with points in the shape of a circle
    # REPLACED BY FARID cartesian mask with binary. Otherwise, SaveToDisk doesn't work.
    # sensor.mask = makeCartCircle(40 * dx, 50);
    sensor_mask = np.zeros((Nx, Ny))
    sensor_mask[0, :] = 1
    sensor = kSensor(sensor_mask)

    # remove points from sensor mask where y < 0
    sensor.mask[:, sensor.mask[1, :] < 0] = np.nan

    # set the input settings
    input_filename  = f'example_input.h5'
    input_file_full_path = os.path.join(pathname, input_filename)
    input_args = {
        'SaveToDisk': input_file_full_path,
        'SaveToDiskExit': True
    }

    # run the simulation
    kspaceFirstOrderASC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })

    assert compare_against_ref(f'out_ivp_axisymmetric_simulation', input_file_full_path), 'Files do not match!'
