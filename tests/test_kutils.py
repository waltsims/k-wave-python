from kwave.utils.kutils import check_stability, primefactors, toneBurst, get_alpha_filter, focus
from kwave import kWaveMedium, kWaveGrid
from kwave.ksource import kSource
from kwave.ktransducer import *

import numpy as np


def test_check_stability():
    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    PML_X_SIZE = 20  # [grid points]
    PML_Y_SIZE = 10  # [grid points]
    PML_Z_SIZE = 10  # [grid points]

    # set total number of grid points not including the PML
    Nx = 128 - 2 * PML_X_SIZE  # [grid points]
    Ny = 128 - 2 * PML_Y_SIZE  # [grid points]
    Nz = 64 - 2 * PML_Z_SIZE  # [grid points]

    # set desired grid size in the x-direction not including the PML
    x = 40e-3  # [m]

    # calculate the spacing between the grid points
    dx = x / Nx  # [m]
    dy = dx  # [m]
    dz = dx  # [m]

    # create the k-space grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    check_stability(kgrid, medium)


def test_prime_factors():
    expected_res = [2, 2, 2, 2, 3, 3]
    assert ((np.array(expected_res) - np.array(primefactors(144))) == 0).all()


def test_get_alpha_filters_2D():
    # =========================================================================
    # SETTINGS
    # =========================================================================

    # size of the computational grid
    Nx = 64  # number of grid points in the x (row) direction
    x = 1e-3  # size of the domain in the x direction [m]
    dx = x / Nx  # grid point spacing in the x direction [m]

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # size of the initial pressure distribution
    source_radius = 2  # [grid points]

    # distance between the centre of the source and the sensor
    source_sensor_distance = 10  # [grid points]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]

    # =========================================================================
    # TWO DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    kgrid = kWaveGrid([Nx, Nx], [dx, dx])

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    filter = get_alpha_filter(kgrid, medium, ['max', 'max'])

    assert np.isclose(filter[32], np.array([0., 0.00956799, 0.03793968, 0.08406256, 0.14616399,
                                   0.22185752, 0.30823458, 0.40197633, 0.4994812, 0.59700331,
                                   0.69079646, 0.7772581, 0.85306778, 0.91531474, 0.9616098,
                                   0.99017714, 0.99992254, 1., 1., 1.,
                                   1., 1., 1., 1., 1.,
                                   1., 1., 1., 1., 1.,
                                   1., 1., 1., 1., 1.,
                                   1., 1., 1., 1., 1.,
                                   1., 1., 1., 1., 1.,
                                   1., 1., 0.99992254, 0.99017714, 0.9616098,
                                   0.91531474, 0.85306778, 0.7772581, 0.69079646, 0.59700331,
                                   0.4994812, 0.40197633, 0.30823458, 0.22185752, 0.14616399,
                                   0.08406256, 0.03793968, 0.00956799, 0.])).all()


def test_get_alpha_filters_1D():
    # =========================================================================
    # SETTINGS
    # =========================================================================

    # size of the computational grid
    Nx = 64  # number of grid points in the x (row) direction
    x = 1e-3  # size of the domain in the x direction [m]
    dx = x / Nx  # grid point spacing in the x direction [m]

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # size of the initial pressure distribution
    source_radius = 2  # [grid points]

    # distance between the centre of the source and the sensor
    source_sensor_distance = 10  # [grid points]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]

    # =========================================================================
    # TWO DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    kgrid = kWaveGrid(Nx, dx)

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    get_alpha_filter(kgrid, medium, ['max'])


def test_focus():
    # simulation settings
    DATA_CAST = 'single'
    RUN_SIMULATION = True

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    PML_X_SIZE = 20  # [grid points]
    PML_Y_SIZE = 10  # [grid points]
    PML_Z_SIZE = 10  # [grid points]

    # set total number of grid points not including the PML
    Nx = 256 - 2 * PML_X_SIZE  # [grid points]
    Ny = 128 - 2 * PML_Y_SIZE  # [grid points]
    Nz = 128 - 2 * PML_Z_SIZE  # [grid points]

    # set desired grid size in the x-direction not including the PML
    x = 40e-3  # [m]

    # calculate the spacing between the grid points
    dx = x / Nx  # [m]
    dy = dx  # [m]
    dz = dx  # [m]

    # create the k-space grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================

    # define the properties of the propagation medium
    c0 = 1540
    rho0 = 1000

    medium = kWaveMedium(
        sound_speed=None,  # will be set later
        alpha_coeff=0.75,
        alpha_power=1.5,
        BonA=6
    )

    # create the time array
    t_end = (Nx * dx) * 2.2 / c0  # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================

    # define properties of the input signal
    source_strength = 1e6  # [Pa]
    tone_burst_freq = 1.5e6  # [Hz]
    tone_burst_cycles = 4

    # create the input signal using toneBurst
    input_signal = toneBurst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)

    # scale the source magnitude by the source_strength divided by the
    # impedance (the source is assigned to the particle velocity)
    input_signal = (source_strength / (c0 * rho0)) * input_signal

    # =========================================================================
    # DEFINE THE ULTRASOUND TRANSDUCER
    # =========================================================================

    # physical properties of the transducer
    transducer = dotdict()
    transducer.number_elements = 32  # total number of transducer elements
    transducer.element_width = 2  # width of each element [grid points/voxels]
    transducer.element_length = 24  # length of each element [grid points/voxels]
    transducer.element_spacing = 0  # spacing (kerf  width) between the elements [grid points/voxels]
    transducer.radius = float('inf')  # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = transducer.number_elements * transducer.element_width + (
            transducer.number_elements - 1) * transducer.element_spacing

    # use this to position the transducer in the middle of the computational grid
    transducer.position = np.round([1, Ny / 2 - transducer_width / 2, Nz / 2 - transducer.element_length / 2])

    transducer = kWaveTransducerSimple(kgrid, **transducer)


    source = kSource()

    focus(kgrid, input_signal, source.mask, [0.5, 0.5, 0.5], 1540)


def test_focus():
    # simulation settings
    DATA_CAST = 'single'
    RUN_SIMULATION = True

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    PML_X_SIZE = 20  # [grid points]
    PML_Y_SIZE = 10  # [grid points]
    PML_Z_SIZE = 10  # [grid points]

    # set total number of grid points not including the PML
    Nx = 256 - 2 * PML_X_SIZE  # [grid points]
    Ny = 128 - 2 * PML_Y_SIZE  # [grid points]
    Nz = 128 - 2 * PML_Z_SIZE  # [grid points]

    # set desired grid size in the x-direction not including the PML
    x = 40e-3  # [m]

    # calculate the spacing between the grid points
    dx = x / Nx  # [m]
    dy = dx  # [m]
    dz = dx  # [m]

    # create the k-space grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================

    # define the properties of the propagation medium
    c0 = 1540
    rho0 = 1000

    medium = kWaveMedium(
        sound_speed=None,  # will be set later
        alpha_coeff=0.75,
        alpha_power=1.5,
        BonA=6
    )

    # create the time array
    t_end = (Nx * dx) * 2.2 / c0  # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================

    # define properties of the input signal
    source_strength = 1e6  # [Pa]
    tone_burst_freq = 1.5e6  # [Hz]
    tone_burst_cycles = 4

    # create the input signal using toneBurst
    input_signal = toneBurst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)

    # scale the source magnitude by the source_strength divided by the
    # impedance (the source is assigned to the particle velocity)
    input_signal = (source_strength / (c0 * rho0)) * input_signal

    # =========================================================================
    # DEFINE THE ULTRASOUND TRANSDUCER
    # =========================================================================

    # physical properties of the transducer
    transducer = dotdict()
    transducer.number_elements = 32  # total number of transducer elements
    transducer.element_width = 2  # width of each element [grid points/voxels]
    transducer.element_length = 24  # length of each element [grid points/voxels]
    transducer.element_spacing = 0  # spacing (kerf  width) between the elements [grid points/voxels]
    transducer.radius = float('inf')  # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = transducer.number_elements * transducer.element_width + (
            transducer.number_elements - 1) * transducer.element_spacing

    # use this to position the transducer in the middle of the computational grid
    transducer.position = np.round([1, Ny / 2 - transducer_width / 2, Nz / 2 - transducer.element_length / 2])

    transducer = kWaveTransducerSimple(kgrid, **transducer)
    imaging_system = NotATransducer(transducer, kgrid)

    focus(kgrid, input_signal, imaging_system.mask, [0.5, 0.5, 0.5], 1540)
