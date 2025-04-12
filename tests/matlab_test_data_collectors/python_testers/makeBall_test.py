import logging
import os
from pathlib import Path

import numpy as np
import pytest

from kwave.data import Vector
from kwave.utils.mapgen import make_ball
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeBall():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/makeBall.mat"))

    for i in range(len(reader)):
        logging.log(logging.INFO, i)

        Nx, Ny, Nz, cx, cy, cz, radius, plot_ball, binary = reader.expected_value_of("params")
        grid_size = Vector([int(Nx), int(Ny), int(Nz)])
        ball_center = Vector([int(cx), int(cy), int(cz)])
        radius, plot_ball, binary = int(radius), bool(plot_ball), bool(binary)

        expected_ball = reader.expected_value_of("ball")

        ball = make_ball(grid_size, ball_center, radius, plot_ball, binary)

        reader.increment()

        assert np.allclose(expected_ball, ball)

    logging.log(logging.INFO, "make_ball(..) works as expected!")


if __name__ == "__main__":
    pytest.main([__file__])
