from kwave.utils.kutils import check_stability, primefactors, toneBurst, get_alpha_filter
from kwave import kWaveMedium, kWaveGrid
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

    assert (filter[32] - np.array([0., 0.00956799, 0.03793968, 0.08406256, 0.14616399,
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
                                   0.08406256, 0.03793968, 0.00956799, 0.]) < 0.01).all()


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
