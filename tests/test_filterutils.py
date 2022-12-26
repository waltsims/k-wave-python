from math import pi

import numpy as np

from kwave.reconstruction.beamform import envelope_detection
from kwave.utils.filters import fwhm, sharpness, \
    smooth
from kwave.utils.math import gaussian
from kwave.utils.signals import create_cw_signals
from kwave.utils.signals import get_win


def test_gaussian():
    x = np.arange(-3, 3.05, 0.05)

    gauss_distr = gaussian(x)
    expected_gaussian = np.array([[0.00443185, 0.00514264, 0.00595253, 0.00687277, 0.00791545,
                                   0.00909356, 0.01042093, 0.01191224, 0.01358297, 0.01544935,
                                   0.0175283, 0.01983735, 0.02239453, 0.02521822, 0.02832704,
                                   0.03173965, 0.03547459, 0.03955004, 0.0439836, 0.04879202,
                                   0.05399097, 0.05959471, 0.06561581, 0.07206487, 0.07895016,
                                   0.08627732, 0.09404908, 0.10226492, 0.11092083, 0.120009,
                                   0.1295176, 0.13943057, 0.14972747, 0.16038333, 0.17136859,
                                   0.18264909, 0.19418605, 0.20593627, 0.21785218, 0.22988214,
                                   0.24197072, 0.25405906, 0.26608525, 0.27798489, 0.28969155,
                                   0.30113743, 0.31225393, 0.32297236, 0.3332246, 0.34294386,
                                   0.35206533, 0.36052696, 0.36827014, 0.37524035, 0.38138782,
                                   0.38666812, 0.39104269, 0.39447933, 0.39695255, 0.39844391,
                                   0.39894228, 0.39844391, 0.39695255, 0.39447933, 0.39104269,
                                   0.38666812, 0.38138782, 0.37524035, 0.36827014, 0.36052696,
                                   0.35206533, 0.34294386, 0.3332246, 0.32297236, 0.31225393,
                                   0.30113743, 0.28969155, 0.27798489, 0.26608525, 0.25405906,
                                   0.24197072, 0.22988214, 0.21785218, 0.20593627, 0.19418605,
                                   0.18264909, 0.17136859, 0.16038333, 0.14972747, 0.13943057,
                                   0.1295176, 0.120009, 0.11092083, 0.10226492, 0.09404908,
                                   0.08627732, 0.07895016, 0.07206487, 0.06561581, 0.05959471,
                                   0.05399097, 0.04879202, 0.0439836, 0.03955004, 0.03547459,
                                   0.03173965, 0.02832704, 0.02521822, 0.02239453, 0.01983735,
                                   0.0175283, 0.01544935, 0.01358297, 0.01191224, 0.01042093,
                                   0.00909356, 0.00791545, 0.00687277, 0.00595253, 0.00514264,
                                   0.00443185]])
    assert (expected_gaussian - gauss_distr < 1e-6).all(), "Gaussian distribution did not match expected distribution"


def generate_test_signal(fs):
    dt = 1 / fs  # [s]
    duration = 0.25  # [s]
    t = np.arange(0, duration, dt)  # [s]
    F = 15
    return np.sin(2 * pi * F * t)

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
    expected_A_sm = np.array([[0.3150659, 0.23343146, 0.09246705, 0.03313708, 0.09246705, 0.23343146],
                     [0.23343146, 0.3150659, 0.23343146, 0.09246705, 0.03313708, 0.09246705],
                     [0.09246705, 0.23343146, 0.3150659, 0.23343146, 0.09246705, 0.03313708],
                     [0.03313708, 0.09246705, 0.23343146, 0.3150659, 0.23343146, 0.09246705],
                     [0.09246705, 0.03313708, 0.09246705, 0.23343146, 0.3150659, 0.23343146],
                     [0.23343146, 0.09246705, 0.03313708, 0.09246705, 0.23343146, 0.3150659]])
    assert (A_sm - expected_A_sm < 0.001).all()
