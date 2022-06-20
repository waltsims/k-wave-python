from kwave.utils.kutils import check_stability, primefactors, toneBurst
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



