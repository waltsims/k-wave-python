import logging
from pathlib import Path

import numpy as np
import os

from kwave.utils.dotdictionary import dotdict
from kwave.utils.mapgen import trim_cart_points
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_trim_cart_points():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/trimCartPoints.mat")
    reader = TestRecordReader(test_record_path)

    kgrids = [reader.expected_value_of(f"kgrid{i}") for i in range(1, 4)]
    kgrids = [dotdict(kgrid) for kgrid in kgrids]
    point_sets = [reader.expected_value_of(f"pointsSet{i}") for i in range(1, 4)]
    reader.increment()

    for i in range(len(kgrids)):
        for j in range(len(point_sets)):
            trimmed_points = trim_cart_points(kgrids[i], point_sets[j])
            expected_trimmed_points = reader.expected_value_of("trimmed_points")

            assert np.allclose(expected_trimmed_points, trimmed_points)

    logging.log(logging.INFO, "trim_cart_points(..) works as expected!")
