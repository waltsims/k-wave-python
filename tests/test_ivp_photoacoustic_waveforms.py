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

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import *
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
    kgrid = kWaveGrid([Nx, Nx], [dx, dx])

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    # create initial pressure distribution
    source = kSource()
    source.p0 = make_disc(Nx, Nx, Nx / 2, Nx / 2, source_radius)

    # define a single sensor point
    sensor_mask = np.zeros((Nx, Nx))
    sensor_mask[Nx//2 - source_sensor_distance - 1, Nx//2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # run the simulation
    input_args = {
        'DataCast': 'single',
        'SaveToDisk': os.path.join(pathname, f'example_input.h5'),
        'SaveToDiskExit': True
    }
    kspaceFirstOrder2DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_ivp_photoacoustic_waveforms/input_1', input_args['SaveToDisk']), \
        'Files do not match!'

    # =========================================================================
    # THREE DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    kgrid = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    # create initial pressure distribution
    source.p0 = make_ball(Nx, Nx, Nx, Nx / 2, Nx / 2, Nx / 2, source_radius)

    # define a single sensor point
    sensor.mask = np.zeros((Nx, Nx, Nx))
    sensor.mask[Nx//2 - source_sensor_distance - 1, Nx//2 - 1, Nx//2 - 1] = 1

    # run the simulation
    input_args = {
        'DataCast': 'single',
        'SaveToDisk': os.path.join(pathname, f'example_input.h5'),
        'SaveToDiskExit': True
    }
    kspaceFirstOrder3DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': deepcopy(source),
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_ivp_photoacoustic_waveforms/input_2', input_args['SaveToDisk']), \
        'Files do not match!'
