import logging
import os
from pathlib import Path

import numpy as np
import pytest

from kwave.data import Vector
from kwave.utils.mapgen import make_cart_arc
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeCartArc():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/make_cart_arc.mat")

    record_reader = TestRecordReader(collected_values_file)

    for i in range(len(record_reader)):
        params = record_reader.expected_value_of("params")

        arc_pos, radius, diameter, focus_pos, num_points = params
        arc_pos = Vector(arc_pos.astype(float))
        focus_pos = Vector(focus_pos.astype(float))
        expected_value = record_reader.expected_value_of("cart_arc")

        cart_arc = make_cart_arc(arc_pos, radius, diameter, focus_pos, num_points)
        record_reader.increment()

        # if radius of curvature is negative
        with pytest.raises(ValueError):
            _ = make_cart_arc(arc_pos, -radius, diameter, focus_pos, num_points)
        # if diameter of arc is negative
        with pytest.raises(ValueError):
            _ = make_cart_arc(arc_pos, radius, -diameter, focus_pos, num_points)
        # if diameter of arc is unphysical
        with pytest.raises(ValueError):
            _ = make_cart_arc(arc_pos, radius, 2.1 * radius, focus_pos, num_points)
        # if focus is at same place as middle of arc
        with pytest.raises(ValueError):
            _ = make_cart_arc(arc_pos, radius, diameter, arc_pos, num_points)

        assert np.allclose(expected_value, cart_arc), "Step {} of {} failed!".format(i, collected_values_file)

    logging.log(logging.INFO, "makeCartArc(..) works as expected!")
