import pytest
from kwave.data import Vector
from kwave.utils.mapgen import make_bowl

import logging
import numpy as np
import os
from pathlib import Path

from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeBowl():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/makeBowl.mat")
    reader = TestRecordReader(collected_values_file)

    for i in range(len(reader)):
        params = reader.expected_value_of("params")
        grid_size, bowl_pos, radius, diameter, focus_pos = params[:5]
        grid_size, bowl_pos, diameter, focus_pos = grid_size, bowl_pos, int(diameter), focus_pos
        grid_size = Vector(grid_size)
        bowl_pos = Vector(bowl_pos)
        focus_pos = Vector(focus_pos)

        try:
            radius = int(radius)
        except OverflowError:
            radius = float(radius)

        binary = bool(params[6])
        remove_overlap = bool(params[8])
        expected_bowl = reader.expected_value_of("bowl")

        bowl = make_bowl(grid_size, bowl_pos, radius, diameter, focus_pos, binary=binary, remove_overlap=remove_overlap)

        assert np.allclose(expected_bowl, bowl)
        reader.increment()

    logging.log(logging.INFO, "make_bowl(..) works as expected!")


if __name__ == "__main__":
    pytest.main([__file__])
