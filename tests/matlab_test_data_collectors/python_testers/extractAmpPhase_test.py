import os
from pathlib import Path

import numpy as np
from kwave.utils.filters import extract_amp_phase
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_extract_amp_phase():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/extract_amp_phase.mat"))

    for _ in range(len(reader)):
        data, Fs, source_freq, dim, fft_padding, window = reader.expected_value_of("params")

        amp, phase, freq = extract_amp_phase(data, Fs, source_freq, dim, fft_padding, window)

        assert np.allclose(amp, reader.expected_value_of("amp")), "amp did not match expected lin_ind"
        assert np.allclose(phase, reader.expected_value_of("phase")), "phase not match expected is"
        assert np.allclose(freq, reader.expected_value_of), "freq did not match expected ks"
        reader.increment()
