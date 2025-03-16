"""Tests for math utility functions."""

import numpy as np
import pytest
from deprecation import fail_if_not_removed
from scipy.spatial.transform import Rotation

from kwave import __version__
from kwave.utils.math import (
    Rx,
    Ry,
    Rz,
    fourier_shift,
    get_affine_matrix,
    make_affine,
    phase_shift_interpolate,
)


@fail_if_not_removed
def test_make_affine_2d():
    """Test 2D affine transformation."""
    translation = [1, 2]
    rotation = 45
    matrix = make_affine(translation, rotation)

    # Check matrix shape
    assert matrix.shape == (3, 3)

    # Check translation part
    assert np.allclose(matrix[:2, 2], translation)

    # Check rotation part (45 degrees)
    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    expected_rotation = np.array([[cos45, -sin45], [sin45, cos45]])
    assert np.allclose(matrix[:2, :2], expected_rotation)


@fail_if_not_removed
def test_make_affine_3d():
    """Test 3D affine transformation."""
    translation = [1, 2, 3]
    rotation = [30, 45, 60]
    matrix = make_affine(translation, rotation)

    # Check matrix shape
    assert matrix.shape == (4, 4)

    # Check translation part
    assert np.allclose(matrix[:3, 3], translation)

    # Check that it's a valid rotation matrix
    rotation_part = matrix[:3, :3]
    assert np.allclose(np.dot(rotation_part, rotation_part.T), np.eye(3))
    assert np.allclose(np.linalg.det(rotation_part), 1.0)


# @pytest.mark.skipif(__version__ >= "0.5.0", reason="These functions should be removed in 0.5.0")
class TestDeprecatedFunctionsBehavior:
    """Test that deprecated functions work correctly during deprecation period and are removed when they should be."""

    @fail_if_not_removed
    def test_rotation_functions_equivalent(self):
        """Test that Rx/Ry/Rz give same results as Rotation.from_euler and are removed when they should be."""
        angle = 45.0
        # Test each rotation function
        assert np.allclose(Rx(angle), Rotation.from_euler("x", angle, degrees=True).as_matrix())
        assert np.allclose(Ry(angle), Rotation.from_euler("y", angle, degrees=True).as_matrix())
        assert np.allclose(Rz(angle), Rotation.from_euler("z", angle, degrees=True).as_matrix())

    @fail_if_not_removed
    def test_affine_functions_equivalent(self):
        """Test that get_affine_matrix and make_affine are equivalent and get_affine_matrix is removed when it should be."""
        translation = [1, 2, 3]
        rotation = [30, 45, 60]

        old_result = get_affine_matrix(translation, rotation)
        new_result = make_affine(translation, rotation)
        assert np.allclose(old_result, new_result)

    @fail_if_not_removed
    def test_shift_functions_equivalent(self):
        """Test that fourier_shift and phase_shift_interpolate are equivalent and fourier_shift is removed when it should be."""
        # Create test signal
        x = np.linspace(0, 10, 100)
        signal = np.sin(x)
        shift = 0.5

        # Compare results
        old_result = fourier_shift(signal, shift)
        new_result = phase_shift_interpolate(signal, shift)
        assert np.allclose(old_result, new_result)

        # Test with explicit dimension
        old_result_dim = fourier_shift(signal, shift, shift_dim=1)
        new_result_dim = phase_shift_interpolate(signal, shift, shift_dim=1)
        assert np.allclose(old_result_dim, new_result_dim)


def test_phase_shift_interpolate():
    """Test phase shift interpolation functionality."""
    # Create a simple signal
    x = np.linspace(0, 2 * np.pi, 100)
    signal = np.sin(x)

    # Test with default parameters
    shifted = phase_shift_interpolate(signal, 0.5)
    assert shifted.shape == signal.shape

    # Test with explicit dimension
    shifted_dim = phase_shift_interpolate(signal, 0.5, shift_dim=1)
    assert shifted_dim.shape == signal.shape

    # Test 2D array
    signal_2d = np.tile(signal, (3, 1))
    shifted_2d = phase_shift_interpolate(signal_2d, 0.5)
    assert shifted_2d.shape == signal_2d.shape

    # Test that the shift preserves signal properties
    # 1. Shape preservation
    assert shifted.shape == signal.shape

    # 2. Amplitude preservation (max absolute value should be similar)
    assert np.allclose(np.max(np.abs(signal)), np.max(np.abs(shifted)), rtol=1e-3)

    # 3. Frequency preservation (check main frequency component magnitude)
    signal_fft = np.fft.fft(signal)
    shifted_fft = np.fft.fft(shifted)

    # Get the magnitude of the main frequency component (first harmonic for sine wave)
    signal_main_freq_mag = np.abs(signal_fft[1])
    shifted_main_freq_mag = np.abs(shifted_fft[1])

    # The magnitude should be preserved within reasonable tolerance
    assert np.allclose(signal_main_freq_mag, shifted_main_freq_mag, rtol=1e-3)

    # 4. Phase shift verification
    # For a sine wave, a phase shift should result in a similar waveform
    # but shifted in time/space. We can verify this by checking the
    # correlation between the signals is high
    correlation = np.corrcoef(signal, shifted)[0, 1]
    assert abs(correlation) > 0.9  # Strong correlation should exist
