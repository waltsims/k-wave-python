import os
from pathlib import Path

import numpy as np

from kwave.utils.filters import gaussian
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_gaussian():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/gaussian.mat'))
    reader = TestRecordReader(test_record_path)
    x = reader.expected_value_of('x')

    y = gaussian(x)
    y_prime = reader.expected_value_of('y')
    assert np.allclose(y, y_prime), "Gaussian distribution did not match expected distribution"
