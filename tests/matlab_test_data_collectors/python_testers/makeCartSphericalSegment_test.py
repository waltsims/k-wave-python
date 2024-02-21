import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.mapgen import make_cart_spherical_segment
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_make_cart_spherical_segments():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/makeCartSphericalSegment.mat")
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        params = reader.expected_value_of("params")
        bowl_pos, radius, inner_diameter, outer_diameter, focus_pos, num_points, plot_bowl, num_points_inner = params
        plot_bowl = isinstance(plot_bowl, bool) or (isinstance(plot_bowl, int) and plot_bowl == 1)

        coordinates = make_cart_spherical_segment(
            bowl_pos, radius, inner_diameter, outer_diameter, focus_pos, num_points, plot_bowl, num_points_inner
        )
        assert np.allclose(coordinates, reader.expected_value_of("coordinates"), equal_nan=True)
        reader.increment()

    logging.log(logging.INFO, "make_cart_spherical_segment(..) works as expected!")
