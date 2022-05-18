import numpy
import numpy as np
import scipy
from scipy.signal import lfilter
from scipy.fftpack import fft, ifft, ifftshift, fftshift
# from scipy.stats import norm
import math
from math import pi

from .misc import sinc
from .conversionutils import scale_SI
from .checkutils import num_dim2


def create_cw_signals(t_array, freq, amp, phase, ramp_length=4):
    """
   create_cw_signals generates a series of continuous wave (CW) signals
   based on the 1D or 2D input matrices amp and phase, where each signal
   is given by:

       amp(i, j) .* sin(2 .* pi .* freq .* t_array + phase(i, j));

   To avoid startup transients, a cosine tapered up-ramp is applied to
   the beginning of the signal. By default, the length of this ramp is
   four periods of the wave. The up-ramp can be turned off by setting
   the ramp_length to 0.

    Example:

        # define sampling parameters
        f = 5e6
        T = 1/f
        Fs = 100e6
        dt = 1/Fs
        t_array = np.arange(0, 10*T, dt)

        # define amplitude and phase
        amp = getWin(9, 'Gaussian')
        phase = np.arange(0, 2*pi, 9).T

        # create signals and plot
        cw_signal = create_cw_signals(t_array, f, amp, phase)

    Args:
        t_array:
        freq:
        amp:
        phase:
        ramp_length:

    Returns:
        cw_signals:

    """
    if len(phase) == 1:
        phase = phase * np.ones(amp.shape)

    N1, N2 = amp.T.shape

    cw_signals = np.zeros([N1, N2, len(t_array)])

    for idx1 in range(N1 - 1):
        for idx2 in range(N2 - 1):
            cw_signals[idx1, idx2, :] = amp[idx1, idx2] * np.sin(2 * pi * freq * t_array + phase[idx1, idx2])

    if ramp_length != 0:
        # get period and time-step
        period = 1 / freq
        dt = t_array[1] - t_array[0]

        # create ramp x-axis between 0 and pi
        ramp_length_points = int(np.round(ramp_length * period / dt))
        ramp_axis = np.arange(0, pi, pi / (ramp_length_points))

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp = np.reshape(ramp, (1, 1, -1))

        # apply ramp to all signals simultaneously

        cw_signals[:, :, :ramp_length_points] *= ramp

    return np.squeeze(cw_signals)


def envelope_detection(signal):
    """
    envelopeDetection applies the Hilbert transform to extract the
    envelope from an input vector x. If x is a matrix, the envelope along
    the last axis.

    Args:
        signal:

    Returns:
        signal_envelope:

    """

    return np.abs(scipy.signal.hilbert(signal))


def brenner_sharpness(im):
    ndim = len(np.squeeze(im).shape)

    if ndim == 2:
        bren_x = (im[:-2, :] - im[2:, :]) ** 2
        bren_y = (im[:, :-2] - im[:, 2:]) ** 2
        s = bren_x.sum() + bren_y.sum()
    elif ndim == 3:
        bren_x = (im[:-2, :, :] - im[2:, :, :]) ** 2
        bren_y = (im[:, :-2, :] - im[:, 2:, :]) ** 2
        bren_z = (im[:, :, :-2] - im[:, :, 2:]) ** 2
        s = bren_x.sum() + bren_y.sum() + bren_z.sum()
    else:
        raise ValueError("Invalid number of dimensions in im")
    return s


def norm_var(im):
    mu = np.mean(im)
    s = np.sum((im - mu) ** 2) / mu
    return s


def tenenbaum_sharpness(im):
    ndim = len(np.squeeze(im).shape)
    if ndim == 2:
        sobel = scipy.ndimage.sobel(im)
    elif ndim ==3:
        sobel = scipy.ndimage.sobel(im)

    else:
        raise ValueError("Invalid number of dimensions in im")
    return sobel.sum()


def sharpness(im, metric="Brenner"):
    """
    sharpness returns a scalar metric related to the sharpness of the 2D
    or 3D image matrix defined by im. By default, the metric is based on
    the Brenner gradient which returns the sum of the centered
    finite-difference at each matrix element in each Cartesian direction.
    Metrics calculated using the Sobel operator or the normalised
    variance can also be returned by setting the input paramater metric.

    For further details, see B. E. Treeby, T. K. Varslot, E. Z. Zhang,
    J. G. Laufer, and P. C. Beard, "Automatic sound speed selection in
    photoacoustic image reconstruction using an autofocus approach," J.
    Biomed. Opt., vol. 16, no. 9, p. 090501, 2011.

    Args:
        im:
        metric (str):   Defaults "Brenner"

    Returns:
        sharp_met

    """
    assert isinstance(im, np.ndarray), "Argument im must be of type numpy array"

    if metric == "Brenner":
        sharp_met = brenner_sharpness(im)
    elif metric == "Tenenbaum":
        sharp_met = tenenbaum_sharpness(im)
    elif metric == "NormVariance":
        sharp_met = norm_var(im)
    else:
        raise ValueError(
            "Unrecognized sharpness metric passed. Valid values are ['Brenner', 'Tanenbaum', 'NormVariance']")

    return sharp_met


def fwhm(f, x):
    """
    fwhm calculates the Full Width at Half Maximum (FWHM) of a positive
    1D input function f(x) with spacing given by x.


    Args:
        f:       f(x)
        x:       x

    Returns:
        fwhm_val:   FWHM of f(x)

    """

    # ensure f is numpy array
    f = np.array(f)
    if len(f.squeeze().shape) != 1:
        raise ValueError("Input function must be 1-dimensional.")

    def lin_interp(x, y, i, half):
        return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))

    def half_max_x(x, y):
        half = max(y) / 2.0
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [lin_interp(x, y, zero_crossings_i[0], half),
                lin_interp(x, y, zero_crossings_i[1], half)]

    hmx = half_max_x(x, f)
    fwhm_val = hmx[1] - hmx[0]

    return fwhm_val


def gaussian(x, magnitude=None, mean=0, variance=1):
    """
    gaussian returns a Gaussian distribution f(x) with the specified
    magnitude, mean, and variance. If these values are not specified, the
    magnitude is normalised and values of variance = 1 and mean = 0 are
    used. For example running

        import matplotlib.pyplot as plt
        x = np.arrange(-3,0.05,3)
        plt.plot(x, gaussian(x))

    will plot a normalised Gaussian distribution.

    Note, the full width at half maximum of the resulting distribution
    can be calculated by FWHM = 2 * sqrt(2 * log(2) * variance).


    Args:
        x:
        magnitude:          Bell height. Defaults to normalized.
        mean (float):       mean or expected value. Defaults to 0.
        variance (float):   variance ~ bell width. Defaults to 1.

    Returns:
        gauss_distr: Gaussian distribution

    """
    if magnitude is None:
        magnitude = (2 * math.pi * variance) ** -0.5

    gauss_distr = magnitude * np.exp(-(x - mean) ** 2 / (2 * variance))

    return gauss_distr
    # return magnitude * norm.pdf(x, loc=mean, scale=variance)


def gaussian_filter(signal, Fs, frequency, bandwidth):
    """
    gaussian_filter applies a frequency domain Gaussian filter with the
    specified center frequency and percentage bandwidth to the input
    signal. If the input signal is given as a matrix, the filter is
    applied to each matrix row.

    Args:
        signal:         signal to filter
        Fs:             sampling frequency [Hz]
        frequency:      center frequency of filter [Hz]
        bandwidth:      bandwidth of filter

    Returns:
        signal:         filtered signal

    """
    N = len(signal)
    if N % 2 == 0:
        f = np.arange(-N / 2, N / 2) * Fs / N
    else:
        f = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1) * Fs / N

    mean = frequency
    variance = (bandwidth / 100 * frequency / (2 * np.sqrt(2 * np.log(2)))) ** 2
    magnitude = 1

    # create double-sided Gaussain filter
    gfilter = np.fmax(gaussian(f, magnitude, mean, variance), gaussian(f, magnitude, -mean, variance))

    # apply filter
    signal = np.real(ifft(ifftshift(gfilter * fftshift(fft(signal)))))

    return signal


def filterTimeSeries(kgrid, medium, signal, *args):
    """
        Filter signal using the Kaiser windowing method
        filterTimeSeries filters an input time domain signal using a low pass
        filter applied by applyFilter with a specified cut-off frequency,
        stop-band attenuation, and transition bandwidth. It uses the Kaiser
        Windowing method to design the FIR filter, which can be implemented
        as either a zero phase or linear phase filter. The cutoff frequency
        is defined by a minimum number of points per wavelength. A smoothing
        ramp can also be applied to the beginning of the signal to reduce
        high frequency transients.
    Args:
        kgrid:
        medium:
        signal:
        *args:

    Returns:

    """
    # default filter cut-off frequency
    points_per_wavelength = 3

    # default ramp length
    ramp_points_per_wavelength = 0

    # default settings for the Kaiser window
    stop_band_atten = 60
    transition_width = 0.1
    zero_phase = False

    # default plot settings
    plot_signals = False
    plot_spectrums = False

    # replace with user defined values if provided
    for input_index in range(0, len(args), 2):
        if args[input_index] == 'PlotSignals':
            plot_signals = args[input_index + 1]
        elif args[input_index] == 'PlotSpectrums':
            plot_spectrums = args[input_index + 1]
        elif args[input_index] == 'PPW':
            points_per_wavelength = args[input_index + 1]
        elif args[input_index] == 'RampPPW':
            ramp_points_per_wavelength = args[input_index + 1]
            if isinstance(ramp_points_per_wavelength, bool) and ramp_points_per_wavelength:
                ramp_points_per_wavelength = points_per_wavelength
        elif args[input_index] == 'StopBandAtten':
            stop_band_atten = args[input_index + 1]
        elif args[input_index] == 'TransitionWidth':
            transition_width = args[input_index + 1]
        elif args[input_index] == 'ZeroPhase':
            zero_phase = args[input_index + 1]
        else:
            raise ValueError('Unknown optional input.')

    # check the input is a row vector
    if num_dim2(signal) == 1:
        m, n = signal.shape
        if n == 1:
            signal = signal.T
            rotate_signal = True
        else:
            rotate_signal = False
    else:
        raise TypeError('Input signal must be a vector.')

    # update the command line status
    print('Filtering input signal...')

    # extract the time step
    assert not isinstance(kgrid.t_array, str) or kgrid.t_array != 'auto', 'kgrid.t_array must be explicitly defined.'

    # compute the sampling frequency
    Fs = 1 / kgrid.dt

    # extract the minium sound speed
    if medium.sound_speed is not None:

        # for the fluid code, use medium.sound_speed
        c0 = medium.sound_speed.min()

    elif all(medium.is_defined('sound_speed_compression', 'sound_speed_shear')):  # pragma: no cover

        # for the elastic code, combine the shear and compression sound speeds and remove zeros values
        ss = np.hstack([medium.sound_speed_compression, medium.sound_speed_shear])
        ss[ss == 0] = np.nan
        c0 = np.nanmin(ss)

        # cleanup unused variables
        del ss

    else:
        raise ValueError(
            'The input fields medium.sound_speed or medium.sound_speed_compression and medium.sound_speed_shear must be defined.')

    # extract the maximum supported frequency (two points per wavelength)
    f_max = kgrid.k_max_all * c0 / (2 * np.pi)

    # calculate the filter cut-off frequency
    filter_cutoff_f = 2 * f_max / points_per_wavelength

    # calculate the wavelength of the filter cut-off frequency as a number of time steps
    filter_wavelength = ((2 * np.pi / filter_cutoff_f) / kgrid.dt)

    # filter the signal if required
    if points_per_wavelength != 0:
        filtered_signal = apply_filter(signal, Fs, float(filter_cutoff_f), 'LowPass',
                                      'ZeroPhase', zero_phase, 'StopBandAtten', float(stop_band_atten),
                                      'TransitionWidth', transition_width, 'Plot', plot_spectrums)

    # add a start-up ramp if required
    if ramp_points_per_wavelength != 0:
        # calculate the length of the ramp in time steps
        ramp_length = round(ramp_points_per_wavelength * filter_wavelength / (2 * points_per_wavelength))

        # create the ramp
        ramp = (-np.cos(np.arange(0, ramp_length - 1 + 1) * np.pi / ramp_length) + 1) / 2

        # apply the ramp
        filtered_signal[1:ramp_length] = filtered_signal[1:ramp_length] * ramp

    # restore the original vector orientation if modified
    if rotate_signal:
        filtered_signal = filtered_signal.T

    # update the command line status
    print(f'  maximum frequency supported by kgrid: {scale_SI(f_max)}Hz (2 PPW)')
    if points_per_wavelength != 0:
        print(f'  filter cutoff frequency: {scale_SI(filter_cutoff_f)}Hz ({points_per_wavelength} PPW)')
    if ramp_points_per_wavelength != 0:
        print(
            f'  ramp frequency: {scale_SI(2 * np.pi / (2 * ramp_length * kgrid.dt))}Hz (ramp_points_per_wavelength PPW)')
    print('  computation complete.')

    # plot signals if required
    if plot_signals:
        raise NotImplementedError

    return filtered_signal


def apply_filter(signal, Fs, cutoff_f, filter_type, zero_phase=False, transition_width=0.1, stop_band_atten=60):
    """
    applyFilter filters an input signal using filter. The FIR filter
    coefficients are based on a Kaiser window with the specified cut-off
    frequency and filter type ('HighPass', 'LowPass' or 'BandPass'). Both
    causal and zero phase filters can be applied.

    Args:
        signal:
        Fs:
        cutoff_f:
        filter_type:
        zero_phase:
        transition_width: as proportion of sampling frequency
        stop_band_atten: [dB]

    Returns:

    """

    # for a bandpass filter, use applyFilter recursively
    if filter_type == 'BandPass':

        # apply the low pass filter
        func_filt_lp = apply_filter(signal, Fs, cutoff_f(2), 'LowPass', 'StopBandAtten', stop_band_atten,
                                   'TransitionWidth', transition_width, 'ZeroPhase', zero_phase);

        # apply the high pass filter
        filtered_signal = apply_filter(func_filt_lp, Fs, cutoff_f(1), 'HighPass', 'StopBandAtten', stop_band_atten,
                                      'TransitionWidth', transition_width, 'ZeroPhase', zero_phase);

    else:

        # check filter type
        if filter_type == 'LowPass':
            high_pass = False
        elif filter_type == 'HighPass':
            high_pass = True
            cutoff_f = (Fs / 2 - cutoff_f)
        else:
            ValueError(f'Unknown filter type {filter_type}')

        # make sure input is the correct way around
        m, n = signal.shape
        if m > n:
            signal = signal.T

        # correct the stopband attenuation if a zero phase filter is being used
        if zero_phase:
            stop_band_atten = stop_band_atten / 2

        # decide the filter order
        N = np.ceil((stop_band_atten - 7.95) / (2.285 * (transition_width * np.pi)))
        N = int(N)

        # construct impulse response of ideal bandpass filter h(n), a sinc function
        fc = cutoff_f / Fs  # normalised cut-off
        n = np.arange(-N / 2, N / 2)
        h = 2 * fc * sinc(2 * np.pi * fc * n)

        # if no window is given, use a Kaiser window
        if 'w' not in locals():

            # compute Kaiser window parameter beta
            if stop_band_atten > 50:
                beta = 0.1102 * (stop_band_atten - 8.7)
            elif stop_band_atten >= 21:
                beta = 0.5842 * (stop_band_atten - 21) ^ 0.4 + 0.07886 * (stop_band_atten - 21)
            else:
                beta = 0

            # construct the Kaiser smoothing window w(n)
            m = np.arange(0, N)
            w = np.real(scipy.special.iv(0, np.pi * beta * np.sqrt(1 - (2 * m / N - 1) ** 2))) / np.real(
                scipy.special.iv(0, np.pi * beta))

        # window the ideal impulse response with Kaiser window to obtain the FIR filter coefficients hw(n)
        hw = w * h

        # modify to make a high_pass filter
        if high_pass:
            hw = (-1 * np.ones((1, len(hw)))) ** (np.arange(1, len(hw))) * hw

        # add some zeros to allow the reverse (zero phase) filtering room to work
        L = signal.size  # length of original input signal
        filtered_signal = np.hstack([np.zeros((1, N)), signal])

        # apply the filter
        filtered_signal = lfilter(hw, 1, filtered_signal);
        if zero_phase:
            filtered_signal = np.fliplr(lfilter(hw, 1, filtered_signal(np.arange(L + N, 1, -1))))

        # remove the part of the signal corresponding to the added zeros
        filtered_signal = filtered_signal[:, N:]

    return filtered_signal
