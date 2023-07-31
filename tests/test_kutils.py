import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.reconstruction.beamform import focus
from kwave.utils.checks import check_stability
from kwave.utils.dotdictionary import dotdict
from kwave.utils.math import primefactors
from kwave.utils.signals import get_alpha_filter, tone_burst


def test_check_stability():
    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    pml_size = Vector([20, 10, 10])  # [grid points]

    # set total number of grid points not including the PML
    grid_size_points = Vector([128, 64, 64]) - 2 * pml_size  # [grid points]

    # set desired grid size in the x-direction not including the PML
    grid_size_meters = 40e-3  # [m]

    # calculate the spacing between the grid points
    grid_spacing_meters = grid_size_meters / Vector([grid_size_points.x, grid_size_points.x, grid_size_points.x])

    # create the k-space grid
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    check_stability(kgrid, medium)


def test_prime_factors():
    expected_res = [2, 2, 2, 2, 3, 3]
    assert ((np.array(expected_res) - np.array(primefactors(144))) == 0).all()


def test_get_alpha_filters_2d():
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
    # [grid points]
    source_radius = 2  # noqa: F841

    # distance between the centre of the source and the sensor
    # [grid points]
    source_sensor_distance = 10  # noqa: F841

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]

    # =========================================================================
    # TWO DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    grid_size = Vector([Nx, Nx])
    grid_spacing = Vector([dx, dx])
    kgrid = kWaveGrid(grid_size, grid_spacing)

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
    # [grid points]
    source_radius = 2  # noqa: F841

    # distance between the centre of the source and the sensor
    # [grid points]
    source_sensor_distance = 10  # noqa: F841

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]

    # =========================================================================
    # TWO DIMENSIONAL SIMULATION
    # =========================================================================

    # create the computational grid
    grid_size = Vector([Nx])
    grid_spacing = Vector([dx])
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    get_alpha_filter(kgrid, medium, ['max'])


def test_focus():
    # simulation settings
    DATA_CAST = 'single'  # noqa: F841
    RUN_SIMULATION = True  # noqa: F841

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    pml_size_points = Vector([20, 10, 10]) # [grid points]

    # set total number of grid points not including the PML
    grid_size_points = Vector([256, 128, 128]) - 2 * pml_size_points  # [grid points]

    # set desired grid size in the x-direction not including the PML
    grid_size_meters = 40e-3  # [m]

    # calculate the spacing between the grid points
    grid_spacing_meters = grid_size_meters / Vector([grid_size_points.x, grid_size_points.x, grid_size_points.x])  # [m]

    # create the k-space grid
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================

    # define the properties of the propagation medium
    c0 = 1540
    rho0 = 1000

    medium = kWaveMedium(  # noqa: F841
        sound_speed=None,  # will be set later
        alpha_coeff=0.75,
        alpha_power=1.5,
        BonA=6
    )

    # create the time array
    t_end = (grid_size_points.x * grid_spacing_meters.x) * 2.2 / c0  # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================

    # define properties of the input signal
    source_strength = 1e6  # [Pa]
    tone_burst_freq = 1.5e6  # [Hz]
    tone_burst_cycles = 4

    # create the input signal using tone_burst
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)

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
    transducer.position = np.round([
        1,
        grid_size_points.y / 2 - transducer_width / 2,
        grid_size_points.z / 2 - transducer.element_length / 2
    ])

    transducer = kWaveTransducerSimple(kgrid, **transducer)
    imaging_system = NotATransducer(transducer, kgrid)

    focus(kgrid, input_signal, imaging_system.mask, [0.5, 0.5, 0.5], 1540)
