from math import pi

import numpy as np

from kwave.reconstruction.beamform import envelope_detection
from kwave.utils.filters import fwhm


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
