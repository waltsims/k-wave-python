from kwave.kgrid import kWaveGrid
from kwave.utils.pml import get_optimal_pml_size


def test_get_optimal_pml_size_1d():
    # size of the computational grid
    nx = 64  # number of grid points in the x (row) direction
    x = 1e-3  # size of the domain in the x direction [m]
    dx = x / nx  # grid point spacing in the x direction [m]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]
    # create the computational grid
    kgrid = kWaveGrid(nx, dx)

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    assert 32 == get_optimal_pml_size(kgrid)
    pass


def test_get_optimal_pml_size_2D():
    # size of the computational grid
    nx = 64  # number of grid points in the x (row) direction
    x = 1e-3  # size of the domain in the x direction [m]
    dx = x / nx  # grid point spacing in the x direction [m]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]
    # create the computational grid
    kgrid = kWaveGrid([nx, nx], [dx, dx])

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    assert (32 == get_optimal_pml_size(kgrid)).all()
    pass


def test_get_optimal_pml_size_3D():
    # size of the computational grid
    nx = 64  # number of grid points in the x (row) direction
    x = 1e-3  # size of the domain in the x direction [m]
    dx = x / nx  # grid point spacing in the x direction [m]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]
    # create the computational grid
    kgrid = kWaveGrid([nx, nx, nx], [dx, dx, dx])

    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    assert (32 == get_optimal_pml_size(kgrid)).all()
