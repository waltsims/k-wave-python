import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.matrix import resize
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_resize():
    test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/resize.mat')
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        volume = reader.expected_value_of('volume')
        expected_resized_volume = reader.expected_value_of('resized_volume')
        new_size = reader.expected_value_of('new_size')
        method = reader.expected_value_of('method')  # TODO: does not work for spline cases

        resized_volume = resize(volume, new_size, interp_mode=method)

        assert np.allclose(expected_resized_volume,
                           resized_volume), f"Results do not match for {i + 1} dimensional case."

    logging.log(logging.INFO,  'revolve2d(..) works as expected!')
