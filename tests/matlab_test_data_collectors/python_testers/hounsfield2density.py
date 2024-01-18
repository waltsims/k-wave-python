import os
from pathlib import Path

import numpy as np

from kwave.utils.conversion import hounsfield2density
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_hounsfield2density():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/hounsfield2density.mat'))
    reader = TestRecordReader(test_record_path)
    p = reader.expected_value_of('p')

    out = hounsfield2density(p)

    out_prime = reader.expected_value_of('out')
    assert np.allclose(out, out_prime), "hounsfield2density did not match expected hounsfield2density"

test_hounsfield2density()
