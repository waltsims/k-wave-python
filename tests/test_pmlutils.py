import pytest


from kwave.data import Vector

from kwave.kgrid import kWaveGrid

from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC, kspace_first_order_2d_gpu
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_disc, make_cart_circle

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

def test_pml_sizes_2d():
    
    nx = 128  # number of grid points in the x (row) direction
    x = 128e-3  # size of the domain in the x direction [m]
    dx = x / nx  # grid point spacing in the x direction [m]
    ny = 128  # number of grid points in the y (column) direction
    y = 128e-3  # size of the domain in the y direction [m]
    dy = y / ny  # grid point spacing in the y direction [m]

    grid_size = Vector([nx, ny]) # [grid points]
    grid_spacing = Vector([dx, dy])  # [m]

    # time array
    dt = 2e-9  # [s]
    t_end = 300e-9  # [s]
    # create the computational grid
    kgrid = kWaveGrid(grid_size, grid_spacing)
    # create the time array
    kgrid.setTime(round(t_end / dt) + 1, dt)

    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # create initial pressure distribution using make_disc
    disc_magnitude = 5 # [Pa]
    disc_pos = Vector([50, 50])  # [grid points]
    disc_radius = 8    # [grid points]
    disc_1 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    disc_magnitude = 3 # [Pa]
    disc_pos = Vector([80, 60])  # [grid points]
    disc_radius = 5    # [grid points]
    disc_2 = disc_magnitude * make_disc(grid_size, disc_pos, disc_radius)

    source = kSource()
    source.p0 = disc_1 + disc_2

    # define a centered circular sensor
    sensor_radius = 4e-3   # [m]
    num_sensor_points = 50
    sensor_mask = make_cart_circle(sensor_radius, num_sensor_points)
    sensor = kSensor(sensor_mask)
    
    with pytest.raises(ValueError):
        simulation_options = SimulationOptions(pml_sizes=[18,18,18])
        _ = kspace_first_order_2d_gpu(medium=medium,
                                      kgrid=kgrid,
                                      source=source,
                                      sensor=sensor,
                                      simulation_options=simulation_options,
                                      execution_options=SimulationExecutionOptions())
    
    with pytest.raises(ValueError):
        simulation_options = SimulationOptions(pml_sizes=[18,18,18]) 
        _ = kspaceFirstOrder2DC(medium=medium,
                                kgrid=kgrid,
                                source=source,
                                sensor=sensor,
                                simulation_options=simulation_options,
                                execution_options=SimulationExecutionOptions())

    pml_size: int = 19
    simulation_options = SimulationOptions(pml_sizes=pml_size)
    assert ((simulation_options.pml_x_size == pml_size) and (simulation_options.pml_y_size == pml_size)), \
        "pml sizes incorrect when passing int"

    pml_sizes = [21, 22]
    simulation_options = SimulationOptions(pml_sizes=pml_sizes)
    assert ((simulation_options.pml_x_size == pml_sizes[0]) and (simulation_options.pml_y_size == pml_sizes[1])), \
        "pml sizes incorrect when passing list"

    pml_sizes = None
    pml_default: int = 20
    simulation_options = SimulationOptions()
    assert ((simulation_options.pml_x_size == pml_default) and (simulation_options.pml_y_size == pml_default)), \
        "pml sizes incorrect when not defining sizes"


