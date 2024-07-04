import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.matrix import max_nd
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_maxND():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/maxND.mat"))

    for _ in range(len(reader)):
        matrix = reader.expected_value_of("matrix")
        expected_max_val = reader.expected_value_of("max_val")
        expected_ind = reader.expected_value_of("ind")

        max_val, ind = max_nd(matrix)

        assert np.allclose(expected_max_val, max_val, equal_nan=True)
        assert np.allclose(expected_ind, ind)

        reader.increment()

    logging.log(logging.INFO, "max_nd(..) works as expected!")


if __name__ == "__main__":
    test_maxND()
