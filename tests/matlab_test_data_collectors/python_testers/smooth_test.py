import os
from pathlib import Path

import numpy as np

from kwave.utils.filters import smooth
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_smooth():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/smooth.mat'))
    reader = TestRecordReader(test_record_path)
    img = reader.expected_value_of('img')

    out = smooth(img)

    out_prime = reader.expected_value_of('out')
    assert np.allclose(out, out_prime), "Smooth did not match expected smooth"
