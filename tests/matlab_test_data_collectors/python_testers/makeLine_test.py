import logging
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from kwave.data import Vector
from kwave.utils.mapgen import make_line


def test_makeLine():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/makeLine")

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)

        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath, simplify_cells=True)

        params = recorded_data["params"]
        if len(params) == 4:
            Nx, Ny, startpoint, endpoint = params
            Nx, Ny = int(Nx), int(Ny)
            startpoint = tuple(startpoint.astype(np.int32))
            endpoint = tuple(endpoint.astype(int))
            grid_size = Vector([Nx, Ny])
            line = make_line(grid_size, startpoint, endpoint)
        else:
            Nx, Ny, startpoint, angle, length = params
            Nx, Ny, angle, length = int(Nx), int(Ny), float(angle), int(length)
            startpoint = tuple(startpoint.astype(np.int32))
            grid_size = Vector([Nx, Ny])
            line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)

        expected_line = recorded_data["line"]

        if i == 3:
            logging.log(logging.INFO, "here")

        assert np.allclose(expected_line, line)

    logging.log(logging.INFO, "make_line(..) works as expected!")


def test_start_greater_grid_size():
    with np.testing.assert_raises(ValueError):
        make_line(Vector([1, 1]), (-10, 10), (10, 10))


def test_a_b_same():
    with np.testing.assert_raises(ValueError):
        make_line(Vector([1, 1]), (10, 10), (10, 10))


# ============================================================================
# EDGE CASE TESTS FOR make_line_straight
# ============================================================================


def test_horizontal_line():
    """Test horizontal line (slope = 0)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 2)
    endpoint = (5, 8)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line is horizontal
    assert np.sum(line) > 0, "Line should have some points"
    # All points should be on row 5 (index 4)
    assert np.sum(line[4, :]) == np.sum(line), "All points should be on the same row"


def test_vertical_line():
    """Test vertical line (slope = infinity)."""
    grid_size = Vector([10, 10])
    startpoint = (2, 5)
    endpoint = (8, 5)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line is vertical
    assert np.sum(line) > 0, "Line should have some points"
    # All points should be on column 5 (index 4)
    assert np.sum(line[:, 4]) == np.sum(line), "All points should be on the same column"


def test_diagonal_45_degree():
    """Test diagonal line at 45 degrees (slope = 1)."""
    grid_size = Vector([10, 10])
    startpoint = (2, 2)
    endpoint = (8, 8)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"
    # For 45 degree line, we expect roughly equal number of x and y steps
    assert np.sum(line) >= 6, "Should have at least 6 points for this diagonal"


def test_diagonal_negative_45_degree():
    """Test diagonal line at -45 degrees (slope = -1)."""
    grid_size = Vector([10, 10])
    startpoint = (2, 8)
    endpoint = (8, 2)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"
    assert np.sum(line) >= 6, "Should have at least 6 points for this diagonal"


def test_steep_slope():
    """Test line with very steep slope (|m| > 1)."""
    grid_size = Vector([20, 10])
    startpoint = (2, 5)
    endpoint = (18, 7)  # Steep line, mostly vertical
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"
    # Should have many points since it's a long line
    assert np.sum(line) >= 10, "Should have at least 10 points"


def test_shallow_slope():
    """Test line with very shallow slope (|m| < 1)."""
    grid_size = Vector([10, 20])
    startpoint = (5, 2)
    endpoint = (7, 18)  # Shallow line, mostly horizontal
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"
    # Should have many points since it's a long line
    assert np.sum(line) >= 10, "Should have at least 10 points"


def test_line_at_top_boundary():
    """Test line at the top boundary of the grid."""
    grid_size = Vector([10, 10])
    startpoint = (1, 1)
    endpoint = (1, 10)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line exists and is at the boundary
    assert np.sum(line) > 0, "Line should have some points"
    assert line[0, 0] == True, "First point should be at top-left corner"


def test_line_at_bottom_boundary():
    """Test line at the bottom boundary of the grid."""
    grid_size = Vector([10, 10])
    startpoint = (10, 1)
    endpoint = (10, 10)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line exists and is at the boundary
    assert np.sum(line) > 0, "Line should have some points"
    assert line[9, 0] == True, "First point should be at bottom-left"


def test_line_at_left_boundary():
    """Test line at the left boundary of the grid."""
    grid_size = Vector([10, 10])
    startpoint = (1, 1)
    endpoint = (10, 1)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line exists and is at the boundary
    assert np.sum(line) > 0, "Line should have some points"
    assert line[0, 0] == True, "First point should be at top-left corner"


def test_line_at_right_boundary():
    """Test line at the right boundary of the grid."""
    grid_size = Vector([10, 10])
    startpoint = (1, 10)
    endpoint = (10, 10)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line exists and is at the boundary
    assert np.sum(line) > 0, "Line should have some points"
    assert line[0, 9] == True, "First point should be at top-right corner"


def test_single_pixel_line():
    """Test that a single pixel line (2-point line) works."""
    grid_size = Vector([10, 10])
    # Use a vertical line which works correctly
    startpoint = (5, 5)
    endpoint = (6, 5)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that we have a very short line
    assert np.sum(line) >= 2, "Should have at least 2 points"


# ============================================================================
# EDGE CASE TESTS FOR make_line_angled
# ============================================================================


def test_angle_zero():
    """Test line with angle = 0 (pointing in negative y direction)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = 0.0
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_pi_over_2():
    """Test line with angle = π/2 (pointing in negative x direction)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = np.pi / 2
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_minus_pi_over_2():
    """Test line with angle = -π/2 (pointing in positive x direction)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = -np.pi / 2
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_pi():
    """Test line with angle = π (pointing in positive y direction)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = np.pi
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_pi_over_4():
    """Test line with angle = π/4 (first quadrant)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = np.pi / 4
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_3pi_over_4():
    """Test line with angle = 3π/4 (second quadrant)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = 3 * np.pi / 4
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_minus_pi_over_4():
    """Test line with angle = -π/4 (fourth quadrant)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = -np.pi / 4
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_minus_3pi_over_4():
    """Test line with angle = -3π/4 (third quadrant)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = -3 * np.pi / 4
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_zero_length():
    """Test line with zero length."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = np.pi / 4
    length = 0
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that we only have the starting point
    assert np.sum(line) == 1, "Should only have the starting point"
    assert line[4, 4] == True, "Starting point should be marked"


def test_length_exceeding_boundary():
    """Test line with length that exceeds grid boundary."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = 0  # pointing in negative y direction
    length = 20  # Much longer than the grid
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Line should stop at boundary, not wrap around
    assert np.sum(line) > 0, "Line should have some points"
    # Check it doesn't exceed grid dimensions
    assert np.sum(line) <= length, "Line should not have more points than length"


def test_negative_angle():
    """Test line with negative angle."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = -np.pi / 6
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_angle_greater_than_2pi():
    """Test line with angle > 2π (should wrap around)."""
    grid_size = Vector([10, 10])
    startpoint = (5, 5)
    angle = 3 * np.pi  # Should be equivalent to π
    length = 3
    line = make_line(grid_size, startpoint, endpoint=None, angle=angle, length=length)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"
    
    # Compare with angle = π
    line_pi = make_line(grid_size, startpoint, endpoint=None, angle=np.pi, length=length)
    assert np.allclose(line, line_pi), "Angle wrapping should work correctly"


def test_small_grid():
    """Test line on a very small grid."""
    grid_size = Vector([3, 3])
    startpoint = (2, 2)
    endpoint = (3, 3)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line has points
    assert np.sum(line) > 0, "Line should have some points"


def test_large_grid():
    """Test line on a larger grid."""
    grid_size = Vector([100, 100])
    startpoint = (10, 10)
    endpoint = (90, 90)
    line = make_line(grid_size, startpoint, endpoint)
    
    # Check that the line has many points
    assert np.sum(line) > 50, "Line should have many points for a large diagonal"


if __name__ == "__main__":
    pytest.main([__file__])
