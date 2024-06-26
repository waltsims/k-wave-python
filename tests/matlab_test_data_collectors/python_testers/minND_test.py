import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.matrix import min_nd
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_minND():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/minND.mat")
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        matrix = reader.expected_value_of("matrix")
        expected_min_val = reader.expected_value_of("min_val")
        expected_ind = reader.expected_value_of("ind")

        min_val, ind = min_nd(matrix)

        assert np.allclose(expected_min_val, min_val, equal_nan=True)
        assert np.allclose(expected_ind, ind)

    logging.log(logging.INFO, "min_nd(..) works as expected!")
