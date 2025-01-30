import os
from pathlib import Path

import numpy as np

from kwave.utils.filters import extract_amp_phase
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_extract_amp_phase():
    # Load the test data
    test_record_path = os.path.join(Path(__file__).parent.parent, "collectedValues/extract_amp_phase.mat")
    reader = TestRecordReader(test_record_path)

    # Get input parameters
    data = reader.expected_value_of("data")
    data_2d = reader.expected_value_of("data_2d")
    data_3d = reader.expected_value_of("data_3d")
    Fs = float(reader.expected_value_of("Fs"))  # Ensure float
    source_freq = float(reader.expected_value_of("source_freq"))
    fft_padding = int(reader.expected_value_of("fft_padding"))
    window = str(reader.expected_value_of("window"))

    # Define comparison tolerance
    rtol = 1e-10
    atol = 1e-15

    # Get expected outputs and run tests for each case
    for test_case in [
        ("1d", data, "auto"),
        ("2d", data_2d, "auto"),
        ("3d", data_3d, "auto"),
        ("dim2", data_2d, 2)
    ]:
        suffix, test_data, dim = test_case
        
        # Get expected values
        expected_amp = reader.expected_value_of(f"amp_{suffix}")
        expected_phase = reader.expected_value_of(f"phase_{suffix}")
        expected_f = reader.expected_value_of(f"f_{suffix}")

        # Run test
        amp, phase, f = extract_amp_phase(
            test_data, Fs, source_freq,
            dim=dim, fft_padding=fft_padding,
            window=window
        )

        # Compare results
        assert np.allclose(amp, expected_amp, rtol=rtol, atol=atol), f"Amplitude mismatch for {suffix}"
        assert np.allclose(phase, expected_phase, rtol=rtol, atol=atol), f"Phase mismatch for {suffix}"
        assert np.allclose(f, expected_f, rtol=rtol, atol=atol), f"Frequency mismatch for {suffix}" 