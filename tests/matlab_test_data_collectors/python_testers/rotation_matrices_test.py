import warnings
import os
from pathlib import Path
import numpy as np
from kwave.utils.math import Rx, Ry, Rz
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_rotation_matrices():
    """Test that Python rotation matrices match MATLAB output"""
    # Ignore deprecation warnings for these functions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Test 3D rotations against MATLAB output
        reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/rotation_matrices.mat"))
        for _ in range(len(reader)):
            theta = reader.expected_value_of("theta")

            # Test Rx
            result = Rx(theta)  # Default 3D
            expected_matrix = reader.expected_value_of("Rx_matrix")
            np.testing.assert_allclose(result, expected_matrix, rtol=1e-15, atol=1e-15, err_msg=f"Rx matrix mismatch for angle {theta}")

            # Test Ry
            result = Ry(theta)  # Default 3D
            expected_matrix = reader.expected_value_of("Ry_matrix")
            np.testing.assert_allclose(result, expected_matrix, rtol=1e-15, atol=1e-15, err_msg=f"Ry matrix mismatch for angle {theta}")

            # Test Rz
            result = Rz(theta)  # Default 3D
            expected_matrix = reader.expected_value_of("Rz_matrix")
            np.testing.assert_allclose(result, expected_matrix, rtol=1e-15, atol=1e-15, err_msg=f"Rz matrix mismatch for angle {theta}")

            reader.increment()
