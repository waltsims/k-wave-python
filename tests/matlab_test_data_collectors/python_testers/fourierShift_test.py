import logging
import os
from pathlib import Path

import numpy as np
import pytest

from kwave.utils.math import fourier_shift
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_fourier_shift():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/fourierShift.mat")
    reader = TestRecordReader(collected_values_folder)

    for i in range(len(reader)):
        # Read recorded data

        data = reader.expected_value_of("data")
        shift = reader.expected_value_of("shift")
        try:
            # - 1 TODO: subtract 1 from dimension here to make fourier_shift use python dimensions
            shift_dim = reader.expected_value_of("shift_dim")
        except KeyError:
            shift_dim = None
        expected_shifted_data = reader.expected_value_of("shifted_data")

        # Execute implementation
        shifted_data = fourier_shift(data, shift, shift_dim)

        # Check correctness
        assert np.allclose(shifted_data, expected_shifted_data)

        reader.increment()

    logging.log(logging.INFO, "fourier_shift(..) works as expected!")


if __name__ == "__main__":
    pytest.main([__file__])
