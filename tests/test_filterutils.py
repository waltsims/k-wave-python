from math import pi

import numpy as np

from kwave.reconstruction.beamform import envelope_detection
from kwave.utils.filters import fwhm, single_sided_correction


def test_envelope_detection():
    fs = 512  # [Hz]
    dt = 1 / fs  # [s]
    duration = 0.25  # [s]
    t = np.arange(0, duration, dt)  # [s]
    F = 60
    data = np.sin(2 * pi * F * t)
    data = envelope_detection(data)
    assert np.allclose(data, 1)


def test_fwhm():
    # Define a function that returns a peak at a given center point
    def peak(x, c):
        return np.exp(-np.power(x - c, 2) / 16.0)

    # Create an array of x values from 0 to 20 with 21 elements
    x = np.linspace(0, 20, 21)
    # Get the y values for the peak centered at x=10
    y = peak(x, 10)
    # Assert that the full width at half maximum (fwhm) of the peak is approximately 6.691
    val, positions = fwhm(y, x)

    assert np.isclose(val, 6.691, rtol=1e-3)
    assert np.isclose((positions[1] - positions[0]) / 2 + positions[0], 10, rtol=1e-3)


class TestSingleSidedCorrection:
    """Tests for the single_sided_correction function."""

    def test_odd_fft_length_dim0(self):
        """Test single_sided_correction with odd FFT length and dim=0."""
        func_fft = np.ones((5, 3))
        fft_len = 5
        dim = 0
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first row is unchanged (DC component)
        assert np.array_equal(func_fft[0, :], np.ones(3))
        # Check that all other rows are multiplied by 2
        assert np.array_equal(func_fft[1:, :], 2 * np.ones((4, 3)))

    def test_odd_fft_length_dim1(self):
        """Test single_sided_correction with odd FFT length and dim=1."""
        func_fft = np.ones((3, 5))
        fft_len = 5
        dim = 1
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first column is unchanged (DC component)
        assert np.array_equal(func_fft[:, 0], np.ones(3))
        # Check that all other columns are multiplied by 2
        assert np.array_equal(func_fft[:, 1:], 2 * np.ones((3, 4)))

    def test_odd_fft_length_dim2(self):
        """Test single_sided_correction with odd FFT length and dim=2."""
        func_fft = np.ones((2, 3, 5))
        fft_len = 5
        dim = 2
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first slice is unchanged (DC component)
        assert np.array_equal(func_fft[:, :, 0], np.ones((2, 3)))
        # Check that all other slices are multiplied by 2
        assert np.array_equal(func_fft[:, :, 1:], 2 * np.ones((2, 3, 4)))

    def test_odd_fft_length_dim3(self):
        """Test single_sided_correction with odd FFT length and dim=3."""
        func_fft = np.ones((2, 2, 2, 5))
        fft_len = 5
        dim = 3
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first hyperslice is unchanged (DC component)
        assert np.array_equal(func_fft[:, :, :, 0], np.ones((2, 2, 2)))
        # Check that all other hyperslices are multiplied by 2
        assert np.array_equal(func_fft[:, :, :, 1:], 2 * np.ones((2, 2, 2, 4)))

    def test_even_fft_length_dim0(self):
        """Test single_sided_correction with even FFT length and dim=0."""
        func_fft = np.ones((6, 3))
        fft_len = 6
        dim = 0
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first and last rows are unchanged (DC and Nyquist components)
        assert np.array_equal(func_fft[0, :], np.ones(3))
        assert np.array_equal(func_fft[-1, :], np.ones(3))
        # Check that all other rows are multiplied by 2
        assert np.array_equal(func_fft[1:-1, :], 2 * np.ones((4, 3)))

    def test_even_fft_length_dim1(self):
        """Test single_sided_correction with even FFT length and dim=1."""
        func_fft = np.ones((3, 6))
        fft_len = 6
        dim = 1
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first and last columns are unchanged (DC and Nyquist components)
        assert np.array_equal(func_fft[:, 0], np.ones(3))
        assert np.array_equal(func_fft[:, -1], np.ones(3))
        # Check that all other columns are multiplied by 2
        assert np.array_equal(func_fft[:, 1:-1], 2 * np.ones((3, 4)))

    def test_even_fft_length_dim2(self):
        """Test single_sided_correction with even FFT length and dim=2."""
        func_fft = np.ones((2, 3, 6))
        fft_len = 6
        dim = 2
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first and last slices are unchanged (DC and Nyquist components)
        assert np.array_equal(func_fft[:, :, 0], np.ones((2, 3)))
        assert np.array_equal(func_fft[:, :, -1], np.ones((2, 3)))
        # Check that all other slices are multiplied by 2
        assert np.array_equal(func_fft[:, :, 1:-1], 2 * np.ones((2, 3, 4)))

    def test_even_fft_length_dim3(self):
        """Test single_sided_correction with even FFT length and dim=3."""
        func_fft = np.ones((2, 2, 2, 6))
        fft_len = 6
        dim = 3
        single_sided_correction(func_fft, fft_len, dim)

        # Check that first and last hyperslices are unchanged (DC and Nyquist components)
        assert np.array_equal(func_fft[:, :, :, 0], np.ones((2, 2, 2)))
        assert np.array_equal(func_fft[:, :, :, -1], np.ones((2, 2, 2)))
        # Check that all other hyperslices are multiplied by 2
        assert np.array_equal(func_fft[:, :, :, 1:-1], 2 * np.ones((2, 2, 2, 4)))

    def test_non_uniform_input(self):
        """Test single_sided_correction with non-uniform input values."""
        # Create an array with non-uniform values
        func_fft = np.ones((6, 3)) * np.arange(1, 7)[:, np.newaxis]
        func_fft_input = func_fft.copy()
        fft_len = 6
        dim = 0
        single_sided_correction(func_fft, fft_len, dim)

        # Check that the first and last rows are unchanged
        assert np.array_equal(func_fft[0, :], func_fft_input[0, :])
        assert np.array_equal(func_fft[-1, :], func_fft_input[-1, :])

        # Check that middle rows are multiplied by 2
        assert np.array_equal(func_fft[1:-1, :], 2 * func_fft_input[1:-1, :])

    def test_input_preservation(self):
        """Test that the input array is modified in-place."""
        func_fft = np.ones((5, 3))
        fft_len = 5
        dim = 0

        # Store a reference to the original array
        original_id = id(func_fft)

        # Apply correction
        single_sided_correction(func_fft, fft_len, dim)

        # Check that the returned array is the same object
        assert id(func_fft) == original_id
