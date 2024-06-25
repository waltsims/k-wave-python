import os
from pathlib import Path
import pytest
from kwave.data import Vector
from kwave.utils.mapgen import make_arc

import logging
import numpy as np

from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeArc():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/makeArc.mat"))

    for i in range(len(reader)):
        logging.log(logging.INFO, i)

        grid_size, arc_pos, radius, diameter, focus_pos = reader.expected_value_of("params")
        grid_size, arc_pos, diameter, focus_pos = grid_size, arc_pos, int(diameter), focus_pos
        try:
            radius = int(radius)
        except OverflowError:
            radius = float(radius)
        expected_arc = reader.expected_value_of("arc")

        grid_size = Vector(grid_size)
        arc_pos = Vector(arc_pos)
        focus_pos = Vector(focus_pos)
        arc = make_arc(grid_size, arc_pos, radius, diameter, focus_pos)

        assert np.allclose(expected_arc, arc)

    logging.log(logging.INFO, "make_arc(..) works as expected!")


if __name__ == "__main__":
    pytest.main([__file__])
