import math

import numpy as np
import scipy
from scipy.fftpack import fft, ifft, ifftshift, fftshift
from scipy.signal import lfilter

from .checks import num_dim, num_dim2, is_number
from .conversion import scale_SI
from .math import find_closest, sinc
from .signals import get_win


# Compute the next highest power of 2 of a 32–bit number `n`
def next_pow2(n):
    """
    Calculate the next power of 2 that is greater than or equal to `n`.

    This function takes a positive integer `n` and returns the smallest power of 2 that is greater
    than or equal to `n`.

    Args:
        n: The number to find the next power of 2 for.

    Returns:
        The smallest power of 2 that is greater than or equal to `n`.
    """

    # decrement `n` (to handle cases when `n` itself is a power of 2)
    n = n - 1

    # set all bits after the last set bit
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16

    # increment `n` and return
    return np.log2(n + 1)


def single_sided_correction(func_fft, fft_len, dim):
    """Correct the single-sided magnitude by multiplying the symmetric points by 2.

    The DC and Nyquist components are unique and are not multiplied by 2.
    The Nyquist component only exists for even numbered FFT lengths.

    Args:
        func_fft: The FFT of the function to be corrected.
        fft_len: The length of the FFT.
        dim: The number of dimensions of `func_fft`.

    Returns:
        The corrected FFT of the function.
    """
    if fft_len % 2:

        # odd FFT length switch dim case
        if dim == 0:
            func_fft[1:, :] = func_fft[1:, :] * 2
        elif dim == 1:
            func_fft[:, 1:] = func_fft[:, 1:] * 2
        elif dim == 2:
            func_fft[:, :, 1:] = func_fft[:, :, 1:] * 2
        elif dim == 3:
            func_fft[:, :, :, 1:] = func_fft[:, :, :, 1:] * 2
    else:

        # even FFT length
        if dim == 0:
            func_fft[1: -1, :, :, :] = func_fft[1: -1, :, :, :] * 2
        elif dim == 2:
            func_fft[:, 1: -1, :, :] = func_fft[:, 1: -1, :, :] * 2
        elif dim == 3:
            func_fft[:, :, 1: -1, :] = func_fft[:, :, 1: -1, :] * 2
        elif dim == 4:
            func_fft[:, :, :, 1: -1] = func_fft[:, :, :, 1: -1] * 2

    return func_fft


def spect(func, Fs, dim='auto', fft_len=0, power_two=False, unwrap=False, window='Rectangular'):
    """
    Args:
        func:          signal to analyse
        Fs:            sampling frequency [Hz]
        dim:           dimension over which the spectrum is calculated
        fft_len:       length of FFT. If the set
                       length is smaller than the signal length, the default
                       value is used instead (default = signal length).
        power_two:     Boolean controlling whether the FFT length is forced to
                       be the next highest power of 2 (default = false).
        unwrap (bool):
        window:        parameter string controlling the window type used to
                       filter the signal before the FFT is taken (default =
                       'Rectangular'). Any valid input types for get_win may be
                       used.

    Returns:
        f:             frequency array
        func_as:       single-sided amplitude spectrum
        func_ps:       single-sided phase spectrum

    Raises:
        ValueError: if input signal is scalar or has more than 4 dimensions.

    """

    # check the size of the input
    sz = func.shape

    # check input isn't scalar
    if np.size(func) == 1:
        raise ValueError('Input signal cannot be scalar.')

    # check input doesn't have more than 4 dimensions
    if len(sz) > 4:
        raise ValueError('Input signal must have 1, 2, 3, or 4 dimensions.')

    # automatically set dimension to first non - singleton dimension
    if dim == 'auto':
        dim_index = 0
        while dim_index <= len(sz):
            if sz[dim_index] > 1:
                dim = dim_index
                break
            dim_index = dim_index + 1

    # assign the number of points being analysed
    func_length = sz[dim]

    # set the length of the FFT
    if not fft_len > func_length:
        if power_two:
            # find an appropriate FFT length of the form 2 ^ N that is equal to or
            # larger than the length of the input signal
            fft_len = 2 ** (next_pow2(func_length))
        else:
            # set the FFT length to the function length
            fft_len = func_length

    # window the signal, reshaping the window to be in the correct direction
    win, coherent_gain = get_win(func_length, window, symmetric=False)
    win = np.reshape(win, tuple(([1] * dim + [func_length] + [1] * (len(sz) - 2))))
    func = win * func

    # compute the fft using the defined FFT length, if fft_len >
    # func_length, the input signal is padded with zeros
    func_fft = fft(func, fft_len, dim)

    # correct for the magnitude scaling of the FFT and the coherent gain of the
    # window(note that the correction is equal to func_length NOT fft_len)
    func_fft = func_fft / (func_length * coherent_gain)

    # reduce to a single sided spectrum where the number of unique points for
    # even numbered FFT lengths is given by N / 2 + 1, and for odd(N + 1) / 2
    num_unique_pts = int(np.ceil((fft_len + 1) / 2))
    if dim == 0:
        func_fft = func_fft[0:num_unique_pts]
    elif dim == 1:
        func_fft = func_fft[:, 0: num_unique_pts]
    elif dim == 2:
        func_fft = func_fft[:, :, 0: num_unique_pts]
    elif dim == 3:
        func_fft = func_fft[:, :, :, 0: num_unique_pts]

    func_fft = single_sided_correction(func_fft, fft_len, dim)

    # create the frequency axis variable
    f = np.arange(0, func_fft.shape[dim]) * Fs / fft_len

    # calculate the amplitude spectrum
    func_as = np.abs(func_fft)

    # calculate the phase spectrum
    func_ps = np.angle(func_fft)

    # unwrap the phase spectrum if required
    if unwrap:
        func_ps = unwrap(func_ps, [], dim)

    return f, func_as, func_ps


def extract_amp_phase(data, Fs, source_freq, dim='auto', fft_padding=3, window='Hanning'):
    """Extract the amplitude and phase information at a specified frequency from a vector or matrix of time series data.

    The amplitude and phase are extracted from the frequency spectrum, which is calculated using a windowed and zero
    padded FFT. The values are extracted at the frequency closest to source_freq. By default, the time dimension is set
    to the highest non-singleton dimension.

    Args:
        data: matrix of time signals [s]
        Fs: sampling frequency [Hz]
        source_freq: frequency at which the amplitude and phase should be extracted [Hz]
        dim: the time dimension of the input data. If 'auto', the highest non-singleton dimension is used.
        fft_padding: the amount of zero padding to apply to the FFT.
        window: the windowing function to use for the FFT.

    Returns:
        amp: the extracted amplitude values
        phase: the extracted phase values
        f: the frequencies of the FFT spectrum
    """

    # check for the dim input
    if dim == 'auto':
        dim = num_dim(data)
        if dim == 2 and data.shape[1] == 1:
            dim = 1

    # create 1D window and reshape to be oriented in the time dimension of the
    # input data
    win, coherent_gain = get_win(data.shape[dim], window)
    # this list magic in Python comes from the use of ones in MATLAB
    # TODO: simplify this
    win = np.reshape(win, [1] * (dim - 1) + [len(win)])

    # apply window to time dimension of input data
    data = win * data

    # compute amplitude and phase spectra
    f, func_as, func_ps = spect(data, Fs, fft_len=fft_padding * data.shape[dim], dim=dim)

    # correct for coherent gain
    func_as = func_as / coherent_gain

    # find the index of the frequency component closest to source_freq
    _, f_index = find_closest(f, source_freq)

    # get size of output variable, collapsing the time dimension
    sz = list(data.shape)
    sz[dim - 1] = 1

    # extract amplitude and relative phase at freq_index
    if dim == 0:
        amp = func_as[f_index]
        phase = func_ps[f_index]
    elif dim == 1:
        amp = func_as[:, f_index]
        phase = func_ps[:, f_index]
    elif dim == 2:
        amp = func_as[:, :, f_index]
        phase = func_ps[:, :, f_index]
    elif dim == 3:
        amp = func_as[:, :, :, f_index]
        phase = func_ps[:, :, :, f_index]
    else:
        raise ValueError('dim must be 0, 1, 2, or 3')

    return amp.squeeze(), phase.squeeze(), f[f_index]


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
    elif ndim == 3:
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
    """ # Former impl. form Farid
        if magnitude is None:
        magnitude = np.sqrt(2 * np.pi * variance)
    return magnitude * np.exp(-(x - mean) ** 2 / (2 * variance))
    """


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


def filter_time_series(kgrid, medium, signal, ppw=3, rppw=0, stop_band_atten=60, transition_width=0.1, zerophase=False,
                       plot_spectrums=False, plot_signals=False):
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
        ppw:        Points-Per-Wavelength (default 3)
        rppw:       Ramp-Points-Per-Wavelength (default 0)
        stop_band_atten:        Stop-Band-Attenuation (default 60)
        transition_width:         Transition-width (default 0.1)
        zero-phase: (default False)


    Returns:

    """

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
            'The input fields medium.sound_speed or medium.sound_speed_compression and medium.sound_speed_shear must '
            'be defined.')

    # extract the maximum supported frequency (two points per wavelength)
    f_max = kgrid.k_max_all * c0 / (2 * np.pi)

    # calculate the filter cut-off frequency
    filter_cutoff_f = 2 * f_max / ppw

    # calculate the wavelength of the filter cut-off frequency as a number of time steps
    filter_wavelength = ((2 * np.pi / filter_cutoff_f) / kgrid.dt)

    # filter the signal if required
    if ppw != 0:
        filtered_signal = apply_filter(signal, Fs, float(filter_cutoff_f), 'LowPass',
                                       zero_phase=zerophase, stop_band_atten=float(stop_band_atten),
                                       transition_width=transition_width)

    # add a start-up ramp if required
    if rppw != 0:
        # calculate the length of the ramp in time steps
        ramp_length = round(rppw * filter_wavelength / (2 * ppw))

        # create the ramp
        ramp = (-np.cos(np.arange(0, ramp_length - 1 + 1) * np.pi / ramp_length) + 1) / 2

        # apply the ramp
        filtered_signal[1:ramp_length] = filtered_signal[1:ramp_length] * ramp

    # restore the original vector orientation if modified
    if rotate_signal:
        filtered_signal = filtered_signal.T

    # update the command line status
    print(f'  maximum frequency supported by kgrid: {scale_SI(f_max)}Hz (2 PPW)')
    if ppw != 0:
        print(f'  filter cutoff frequency: {scale_SI(filter_cutoff_f)}Hz ({ppw} PPW)')
    if rppw != 0:
        print(
            f'  ramp frequency: {scale_SI(2 * np.pi / (2 * ramp_length * kgrid.dt))}Hz (ramp_points_per_wavelength PPW)')
    print('  computation complete.')

    # plot signals if required
    if plot_signals or plot_spectrums:
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
        assert isinstance(cutoff_f, list), "List of two frequencies required as for filter type 'BandPass'"
        assert len(cutoff_f) == 2, "List of two frequencies required as for filter type 'BandPass'"

        # apply the low pass filter
        func_filt_lp = apply_filter(signal, Fs, cutoff_f[1], 'LowPass', stop_band_atten=stop_band_atten,
                                    transition_width=transition_width, zero_phase=zero_phase)

        # apply the high pass filter
        filtered_signal = apply_filter(func_filt_lp, Fs, cutoff_f[0], 'HighPass', stop_band_atten=stop_band_atten,
                                       transition_width=transition_width, zero_phase=zero_phase)

    else:

        # check filter type
        if filter_type == 'LowPass':
            high_pass = False
        elif filter_type == 'HighPass':
            high_pass = True
            cutoff_f = (Fs / 2 - cutoff_f)
        else:
            raise ValueError(f'Unknown filter type {filter_type}. Options are "LowPass, HighPass, BandPass"')

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
        # TODO: there is no window argument
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
            hw = (-1 * np.ones((1, len(hw))) ** (np.arange(1, len(hw) + 1))) * hw

        # add some zeros to allow the reverse (zero phase) filtering room to work
        L = signal.size  # length of original input signal
        filtered_signal = np.hstack([np.zeros((1, N)), signal]).squeeze()

        # apply the filter
        filtered_signal = lfilter(hw.squeeze(), 1, filtered_signal)
        if zero_phase:
            filtered_signal = np.fliplr(lfilter(hw.squeeze(), 1, filtered_signal[np.arange(L + N, 1, -1)]))

        # remove the part of the signal corresponding to the added zeros
        filtered_signal = filtered_signal[N:]

    return filtered_signal[np.newaxis]


def smooth(a, restore_max=False, window_type='Blackman'):
    """
    Smooth a matrix.

    DESCRIPTION:
    smooth filters an input matrix using an n - dimensional frequency
    domain window created using get_win. If no window type is specified, a
    Blackman window is used.

    Args:
        a: spatial distribution to smooth
        restore_max:  Boolean controlling whether the maximum value is restored after smoothing(default=false).
        window_type:  shape of the smoothing window; any valid inputs to get_win are supported(default='Blackman').

    OUTPUTS:
    A_sm - smoothed
    """
    DEF_USE_ROTATION = True

    assert is_number(a) and np.all(~np.isinf(a))
    assert isinstance(restore_max, bool)
    assert isinstance(window_type, str)

    # get the grid size
    grid_size = a.shape

    # remove singleton dimensions
    if num_dim2(a) != len(grid_size):
        grid_size = np.squeeze(grid_size)

    # use a symmetric filter for odd grid sizes, and a non-symmetric filter for
    # even grid sizes to ensure the DC component of the window has a value of
    # unity
    window_symmetry = (np.array(grid_size) % 2).astype(bool)

    # get the window, taking the absolute value to discard machine precision
    # negative values
    from .signals import get_win
    win, _ = get_win(grid_size, type_=window_type,
                     rotation=DEF_USE_ROTATION, symmetric=window_symmetry)
    win = np.abs(win)

    # rotate window if input mat is (1, N)
    if a.shape[0] == 1:  # is row?
        win = win.T

    # apply the filter
    a_sm = np.real(np.fft.ifftn(np.fft.fftn(a) * np.fft.ifftshift(win)))

    # restore magnitude if required
    if restore_max:
        a_sm = (np.abs(a).max() / np.abs(a_sm).max()) * a_sm
    return a_sm
