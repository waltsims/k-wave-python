import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.mapgen import make_cart_bowl
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_make_cart_bowl():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/makeCartBowl.mat")
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        params = reader.expected_value_of("params")
        bowl_pos, radius, diameter, focus_pos, num_points, plot_bowl = params
        focus_pos = focus_pos.astype(np.float64)
        bowl_pos = bowl_pos.astype(np.float64)
        radius = float(radius)
        diameter = float(diameter)
        plot_bowl = plot_bowl == 1
        coordinates = make_cart_bowl(bowl_pos, radius, diameter, focus_pos, num_points, plot_bowl)
        assert np.allclose(coordinates, reader.expected_value_of("coordinates"), equal_nan=True)
        reader.increment()

    logging.log(logging.INFO, "make_cart_bowl(..) works as expected!")
