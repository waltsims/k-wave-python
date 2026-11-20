"""Regression tests for issue #469: make_multi_bowl rejected integer radii and list inputs."""

import numpy as np
import pytest

from kwave.data import Vector
from kwave.utils.mapgen import make_bowl, make_multi_bowl

GRID_SIZE = Vector([64, 64, 64])
BOWL_POS = [(19, 19, 31), (48, 48, 31)]
RADIUS = [20, 15]
DIAMETER = [15, 21]
FOCUS_POS = [(31, 31, 31)]


def _expected_bowls():
    expected = np.zeros(tuple(GRID_SIZE))
    for pos, radius, diameter in zip(BOWL_POS, RADIUS, DIAMETER):
        expected += make_bowl(GRID_SIZE, Vector(pos), float(radius), diameter, Vector(FOCUS_POS[0]))
    return expected


@pytest.mark.parametrize("as_array", [True, False], ids=["ndarray", "list"])
def test_make_multi_bowl_integer_inputs(as_array):
    """Integer radii and list / ndarray positions (the reproducer from the issue) are accepted."""
    convert = (lambda x: np.array(x, dtype=int)) if as_array else (lambda x: x)

    bowls, bowls_labelled = make_multi_bowl(GRID_SIZE, convert(BOWL_POS), convert(RADIUS), convert(DIAMETER), convert(FOCUS_POS))

    assert bowls.shape == tuple(GRID_SIZE)
    assert bowls_labelled.shape == tuple(GRID_SIZE)
    assert np.array_equal(bowls, _expected_bowls())


def test_make_multi_bowl_scalar_radius_and_diameter():
    """A single radius / diameter / focus position is shared by all bowls."""
    bowl_pos = [(20, 20, 31), (20, 44, 31)]

    bowls, _ = make_multi_bowl(GRID_SIZE, bowl_pos, 20, 15, (31, 31, 31))

    expected = sum(make_bowl(GRID_SIZE, Vector(pos), 20.0, 15, Vector([31, 31, 31])) for pos in bowl_pos)
    assert np.array_equal(bowls, expected)


def test_make_multi_bowl_single_bowl():
    bowls, _ = make_multi_bowl(GRID_SIZE, [(20, 20, 31)], [20], [15], [(31, 31, 31)])

    assert np.array_equal(bowls, make_bowl(GRID_SIZE, Vector([20, 20, 31]), 20.0, 15, Vector([31, 31, 31])))


def test_make_bowl_bounding_box_ends_at_grid_edge():
    """A bowl whose bounding box ends exactly on the grid boundary is placed without truncation."""
    # the bounding box is 32 grid points wide and starts at 48 - 16 = 32, so it ends at 64
    bowl = make_bowl(GRID_SIZE, Vector([48, 48, 31]), 15.0, 21, Vector([31, 31, 31]))
    # the same bowl shifted away from the boundary by one grid point
    shifted = make_bowl(GRID_SIZE, Vector([47, 47, 31]), 15.0, 21, Vector([30, 30, 31]))

    assert bowl.sum() > 0
    assert np.array_equal(bowl[1:, 1:, :], shifted[:-1, :-1, :])


def test_make_multi_bowl_input_validation():
    with pytest.raises(ValueError, match="3 columns"):
        make_multi_bowl(GRID_SIZE, [(19, 19), (48, 48)], RADIUS, DIAMETER, FOCUS_POS)
    with pytest.raises(ValueError, match="radius"):
        make_multi_bowl(GRID_SIZE, BOWL_POS, [20, 15, 10], DIAMETER, FOCUS_POS)
    with pytest.raises(ValueError, match="diameter"):
        make_multi_bowl(GRID_SIZE, BOWL_POS, RADIUS, [15, 21, 11], FOCUS_POS)
