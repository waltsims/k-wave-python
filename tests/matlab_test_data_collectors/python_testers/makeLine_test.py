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


if __name__ == "__main__":
    pytest.main([__file__])
