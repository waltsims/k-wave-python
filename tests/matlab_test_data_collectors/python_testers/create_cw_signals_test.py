import logging
import os
from pathlib import Path

import numpy as np

from kwave.utils.signals import create_cw_signals
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_create_cw_signals():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/createCWSignals.mat'))
    reader = TestRecordReader(test_record_path)
    amp = reader.expected_value_of('amp')
    phase = reader.expected_value_of('phase')
    f = reader.expected_value_of('f')
    t_array = reader.expected_value_of('t_array')
    signal_prime = reader.expected_value_of('signal')

    signal = create_cw_signals(t_array=t_array, amp=amp, phase=phase, freq=f)

    # compare signal and signal_prime element-wise
    difference = signal - signal_prime
    tolerance = 1e-6
    is_close = np.allclose(signal, signal_prime, atol=tolerance, rtol=tolerance)

    if not is_close:
        logging.log(logging.INFO,  "signal and signal_prime are not equal")
        logging.log(logging.INFO,  "difference =", difference)
    else:
        logging.log(logging.INFO,  "signal and signal_prime are equal within the specified tolerance")
    assert is_close, "signal did not match expected signal"
