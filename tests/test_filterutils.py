from math import pi

import numpy as np

from kwave.reconstruction.beamform import envelope_detection
from kwave.utils.filters import fwhm
from kwave.utils.signals import create_cw_signals
from kwave.utils.signals import get_win


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
    assert np.isclose(fwhm(y, x), 6.691, rtol=1e-3)


# TODO:
def test_create_cw_signals():
    # define sampling parameters
    f = 5e6
    T = 1 / f
    Fs = 100e6
    dt = 1 / Fs
    t_array = np.arange(0, 10 * T, dt)

    # define amplitude and phase
    amp, cg = get_win(9, type_='Gaussian')
    phase = np.arange(0, 2 * pi, 9).T

    # create signals
    cw_signal = create_cw_signals(t_array, f, amp, phase)
