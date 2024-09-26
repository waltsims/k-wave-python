import logging
import os
from pathlib import Path

import numpy as np
import pytest

from kwave.data import Vector
from kwave.utils.mapgen import make_circle
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeCircle():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/makeCircle.mat"))

    for i in range(len(reader)):
        logging.log(logging.INFO, i)

        Nx, Ny, cx, cy, radius, arc_angle = reader.expected_value_of("param")

        grid_size = Vector([Nx, Ny])
        center = Vector([cx, cy])
        circle = make_circle(grid_size, center, radius, arc_angle)

        expected_circle = reader.expected_value_of("circle")

        assert np.allclose(expected_circle, circle)

        reader.increment()

    logging.log(logging.INFO, "make_circle(..) works as expected!")


if __name__ == "__main__":
    pytest.main([__file__])
