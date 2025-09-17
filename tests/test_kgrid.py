import numpy as np
import pytest

from kwave.kgrid import kWaveGrid


def test_from_geometry():
    # Test 1D grid
    dimensions = [0.1]  # 10cm domain
    min_element_width = 0.001  # 1mm minimum element
    grid = kWaveGrid.from_geometry(dimensions, min_element_width)
    assert grid.dim == 1
    assert grid.dx == 0.0001  # 0.1mm spacing (min_element_width/10)
    assert grid.Nx == 1000  # 10cm/0.1mm

    # Test 2D grid
    dimensions = [0.1, 0.2]  # 10cm x 20cm domain
    min_element_width = 0.001  # 1mm minimum element
    grid = kWaveGrid.from_geometry(dimensions, min_element_width)
    assert grid.dim == 2
    assert grid.dx == 0.0001  # 0.1mm spacing
    assert grid.dy == 0.0001  # 0.1mm spacing
    assert grid.Nx == 1000  # 10cm/0.1mm
    assert grid.Ny == 2000  # 20cm/0.1mm

    # Test 3D grid
    dimensions = [0.1, 0.2, 0.3]  # 10cm x 20cm x 30cm domain
    min_element_width = 0.001  # 1mm minimum element
    grid = kWaveGrid.from_geometry(dimensions, min_element_width)
    assert grid.dim == 3
    assert grid.dx == 0.0001  # 0.1mm spacing
    assert grid.dy == 0.0001  # 0.1mm spacing
    assert grid.dz == 0.0001  # 0.1mm spacing
    assert grid.Nx == 1000  # 10cm/0.1mm
    assert grid.Ny == 2000  # 20cm/0.1mm
    assert grid.Nz == 3000  # 30cm/0.1mm

    # Test custom points_per_wavelength
    dimensions = [0.1]  # 10cm domain
    min_element_width = 0.001  # 1mm minimum element
    points_per_wavelength = 20  # Double the default
    grid = kWaveGrid.from_geometry(dimensions, min_element_width, points_per_wavelength=points_per_wavelength)
    assert grid.dx == 0.00005  # 0.05mm spacing
    assert grid.Nx == 2000  # 10cm/0.05mm

    # Test error cases
    with pytest.raises(ValueError):
        kWaveGrid.from_geometry([-0.1], 0.01)  # Negative dimension
    with pytest.raises(ValueError):
        kWaveGrid.from_geometry([0.1], -0.01)  # Negative element width


def test_from_domain():
    # Test 1D grid based on at_circular_piston_3D example
    dimensions = [0.032]  # 32mm domain
    frequency = 1e6  # 1MHz
    sound_speed = 1500  # 1500 m/s
    ppw = 3  # points per wavelength
    grid = kWaveGrid.from_domain(dimensions, frequency, sound_speed, points_per_wavelength=ppw)

    wavelength = sound_speed / frequency  # 1.5mm
    expected_spacing = wavelength / ppw  # 0.5mm
    expected_points = int(np.ceil(dimensions[0] / expected_spacing))

    assert grid.dim == 1
    assert np.isclose(grid.dx, expected_spacing)
    assert grid.Nx == expected_points

    # Test 2D grid with different sound speeds
    dimensions = [0.032, 0.023]  # 32mm x 23mm domain (from example)
    frequency = 1e6  # 1MHz
    sound_speed_min = 1500  # 1500 m/s
    sound_speed_max = 2000  # 2000 m/s
    grid = kWaveGrid.from_domain(dimensions, frequency, sound_speed_min, sound_speed_max, points_per_wavelength=ppw)

    wavelength = sound_speed_min / frequency  # 1.5mm
    expected_spacing = wavelength / ppw  # 0.5mm
    expected_points_x = int(np.ceil(dimensions[0] / expected_spacing))
    expected_points_y = int(np.ceil(dimensions[1] / expected_spacing))

    assert grid.dim == 2
    assert np.isclose(grid.dx, expected_spacing)
    assert np.isclose(grid.dy, expected_spacing)
    assert grid.Nx == expected_points_x
    assert grid.Ny == expected_points_y

    # Test error cases
    with pytest.raises(ValueError):
        kWaveGrid.from_domain([-0.1], 1e6, 1500)  # Negative dimension
    with pytest.raises(ValueError):
        kWaveGrid.from_domain([0.1], -1e6, 1500)  # Negative frequency
    with pytest.raises(ValueError):
        kWaveGrid.from_domain([0.1], 1e6, -1500)  # Negative sound speed


def test_total_grid_points():
    # Test 1D grid
    grid = kWaveGrid([10], [0.1])
    assert grid.total_grid_points == 10

    # Test 2D grid
    grid = kWaveGrid([10, 20], [0.1, 0.1])
    assert grid.total_grid_points == 200

    # Test 3D grid
    grid = kWaveGrid([10, 20, 30], [0.1, 0.1, 0.1])
    assert grid.total_grid_points == 6000


def test_kx_ky_kz_properties():
    # Test 1D grid
    grid = kWaveGrid([10], [0.1])
    assert np.array_equal(grid.kx, grid.k_vec.x)
    assert np.isnan(grid.ky)
    assert np.isnan(grid.kz)

    # Test 2D grid
    grid = kWaveGrid([10, 20], [0.1, 0.1])
    expected_kx = np.tile(grid.k_vec.x, (1, 20))
    expected_ky = np.tile(grid.k_vec.y.T, (10, 1))
    assert np.array_equal(grid.kx, expected_kx)
    assert np.array_equal(grid.ky, expected_ky)
    assert np.isnan(grid.kz)

    # Test 3D grid
    grid = kWaveGrid([10, 20, 30], [0.1, 0.1, 0.1])
    expected_kx = np.tile(grid.k_vec.x[:, :, None], (1, 20, 30))
    expected_ky = np.tile(grid.k_vec.y[None, :, :], (10, 1, 30))
    expected_kz = np.tile(grid.k_vec.z.T[None, :, :], (10, 20, 1))
    assert np.array_equal(grid.kx, expected_kx)
    assert np.array_equal(grid.ky, expected_ky)
    assert np.array_equal(grid.kz, expected_kz)


def test_size_properties():
    # Test 1D grid
    grid = kWaveGrid([10], [0.1])
    assert grid.x_size == 1.0  # 10 * 0.1
    assert grid.y_size == 0.0  # Not applicable in 1D
    assert grid.z_size == 0.0  # Not applicable in 1D

    # Test 2D grid
    grid = kWaveGrid([10, 20], [0.1, 0.1])
    assert grid.x_size == 1.0  # 10 * 0.1
    assert grid.y_size == 2.0  # 20 * 0.1
    assert grid.z_size == 0.0  # Not applicable in 2D

    # Test 3D grid
    grid = kWaveGrid([10, 20, 30], [0.1, 0.1, 0.1])
    assert grid.x_size == 1.0  # 10 * 0.1
    assert grid.y_size == 2.0  # 20 * 0.1
    assert grid.z_size == 3.0  # 30 * 0.1
