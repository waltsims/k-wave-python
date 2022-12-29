import os
from math import pi
from pathlib import Path

import numpy as np

from kwave.reconstruction.beamform import envelope_detection
from kwave.utils.filters import gaussian, fwhm, sharpness, \
    smooth
from kwave.utils.signals import create_cw_signals
from kwave.utils.signals import get_win
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


# todo:
def test_tannenbaum_sharpness2():
    image = np.ones((5, 5))
    image[:2, :] = 0.0
    out = sharpness(image, 'Tenenbaum')


def test_tannenbaum_sharpness3():
    image = np.ones((5, 5, 5))
    image[:2, :] = 0.0
    out = sharpness(image, 'Tenenbaum')
    out = sharpness(image)


def test_brenner_sharpness2():
    image = np.ones((5, 5))
    image[:2, :] = 0.0
    out = sharpness(image)


def test_brenner_sharpness3():
    image = np.ones((5, 5, 5))
    image[:2, :] = 0.0
    out = sharpness(image)


def test_norm_var_sharpness():
    image = np.ones((5, 5))
    image[:2, :] = 0.0
    out = sharpness(image, 'NormVariance')


def test_envelope_detection():
    fs = 512  # [Hz]
    dt = 1 / fs  # [s]
    duration = 0.25  # [s]
    t = np.arange(0, duration, dt)  # [s]
    F = 60
    data = np.sin(2 * pi * F * t)
    data = envelope_detection(data)
    assert (abs(data - 1) < 1e-4).all()


def test_fwhm():
    def peak(x, c):
        return np.exp(-np.power(x - c, 2) / 16.0)

    x = np.linspace(0, 20, 21)
    y = peak(x, 10)
    assert abs(fwhm(y, x) - 6.691) < 1e-3


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


def test_smooth():
    A = np.diag(np.ones(6))
    A_sm = np.array(smooth(A))
    # TODO: refactor
    expected_A_sm = np.array([[0.3150659, 0.23343146, 0.09246705, 0.03313708, 0.09246705, 0.23343146],
                     [0.23343146, 0.3150659, 0.23343146, 0.09246705, 0.03313708, 0.09246705],
                     [0.09246705, 0.23343146, 0.3150659, 0.23343146, 0.09246705, 0.03313708],
                     [0.03313708, 0.09246705, 0.23343146, 0.3150659, 0.23343146, 0.09246705],
                     [0.09246705, 0.03313708, 0.09246705, 0.23343146, 0.3150659, 0.23343146],
                     [0.23343146, 0.09246705, 0.03313708, 0.09246705, 0.23343146, 0.3150659]])
    assert (A_sm - expected_A_sm < 0.001).all()
