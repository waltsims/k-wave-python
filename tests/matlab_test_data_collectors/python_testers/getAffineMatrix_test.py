import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.math import make_affine
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_get_affine_matrix():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/getAffineMatrix.mat")
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        params = reader.expected_value_of("params")
        translation, rotation = params
        affine_matrix = make_affine(translation, rotation)
        assert np.allclose(affine_matrix, reader.expected_value_of("affine_matrix"))
        reader.increment()

    logging.log(logging.INFO, "make_affine(..) works as expected!")
