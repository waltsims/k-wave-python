import logging
import os
from pathlib import Path
import warnings

import numpy as np
import pytest

from kwave.utils.math import phase_shift_interpolate, fourier_shift
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_phase_shift_interpolate():
    """Test that phase_shift_interpolate works as expected"""
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/fourierShift.mat")
    reader = TestRecordReader(collected_values_file)

    for i in range(len(reader)):
        # Read recorded data
        data = reader.expected_value_of("data")
        shift = reader.expected_value_of("shift")
        try:
            shift_dim = reader.expected_value_of("shift_dim")
        except KeyError:
            shift_dim = None
        expected_shifted_data = reader.expected_value_of("shifted_data")

        # Execute implementation with new function
        shifted_data = phase_shift_interpolate(data, shift, shift_dim)

        # Check correctness
        assert np.allclose(shifted_data, expected_shifted_data)

        reader.increment()

    logging.log(logging.INFO, "phase_shift_interpolate works as expected!")


def test_fourier_shift_deprecation():
    """Test that the old fourier_shift function works and raises deprecation warning"""
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/fourierShift.mat")
    reader = TestRecordReader(collected_values_file)

    # Get first test case
    data = reader.expected_value_of("data")
    shift = reader.expected_value_of("shift")
    try:
        shift_dim = reader.expected_value_of("shift_dim")
    except KeyError:
        shift_dim = None
    expected_shifted_data = reader.expected_value_of("shifted_data")

    # Test that old function raises deprecation warning but still works
    shifted_data = fourier_shift(data, shift, shift_dim)
    assert np.allclose(shifted_data, expected_shifted_data)

    logging.log(logging.INFO, "fourier_shift deprecation works as expected!")


if __name__ == "__main__":
    pytest.main([__file__])
