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


def test_rotation_matrices_other_dims():
    """Test rotation matrices in other dimensionalities"""
    # Ignore deprecation warnings for these functions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Test 2D rotations
        theta = 45
        expected_2d = np.array([[cosd(theta), -sind(theta)], [sind(theta), cosd(theta)]])

        result = Rx(theta, dim=2)
        np.testing.assert_allclose(result[0:2, 0:2], np.eye(2), rtol=1e-15, atol=1e-15, err_msg="2D Rx should be identity in xy-plane")

        result = Rz(theta, dim=2)
        np.testing.assert_allclose(result, expected_2d, rtol=1e-15, atol=1e-15, err_msg="2D Rz mismatch")

        # Test 4D rotations
        result = Rx(theta, dim=4)
        np.testing.assert_allclose(result[0, :], [1, 0, 0, 0], rtol=1e-15, atol=1e-15, err_msg="4D Rx should preserve first dimension")
        np.testing.assert_allclose(result[3, :], [0, 0, 0, 1], rtol=1e-15, atol=1e-15, err_msg="4D Rx should preserve fourth dimension")

        result = Ry(theta, dim=4)
        np.testing.assert_allclose(result[1, :], [0, 1, 0, 0], rtol=1e-15, atol=1e-15, err_msg="4D Ry should preserve second dimension")
        np.testing.assert_allclose(result[3, :], [0, 0, 0, 1], rtol=1e-15, atol=1e-15, err_msg="4D Ry should preserve fourth dimension")

        result = Rz(theta, dim=4)
        np.testing.assert_allclose(
            result[2:, 2:], np.eye(2), rtol=1e-15, atol=1e-15, err_msg="4D Rz should preserve third and fourth dimensions"
        )
