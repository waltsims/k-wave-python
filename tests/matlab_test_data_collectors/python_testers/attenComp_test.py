import os
from pathlib import Path

import numpy as np

from kwave.utils.atten_comp import atten_comp
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_angular_spectrum():
    # test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/attenComp.mat')
    test_record_path = os.path.join('/Users/farid/workspace/black_box_testing', 'collectedValues/attenComp.mat')
    reader = TestRecordReader(test_record_path)

    inp_signal = reader.expected_value_of('inp_signal')
    dt = reader.expected_value_of('dt')
    c = reader.expected_value_of('c')
    alpha_0 = reader.expected_value_of('alpha_0')
    y = reader.expected_value_of('y')

    expected_out_signal = reader.expected_value_of('out_signal')
    expected_tfd = reader.expected_value_of('tfd')
    expected_cutoff_freq = reader.expected_value_of('cutoff_freq')
    fit_type = reader.expected_value_of('fit_type')

    out_signal, tfd, cutoff_freq = atten_comp(inp_signal, dt, c, alpha_0, y, fit_type=fit_type)

    assert np.allclose(out_signal, expected_out_signal)
    assert np.allclose(tfd, expected_tfd)
    assert np.allclose(cutoff_freq, expected_cutoff_freq)
