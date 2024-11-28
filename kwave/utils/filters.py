import logging
from typing import Optional, Union, Tuple, List

import numpy as np
import scipy
from scipy.fftpack import fft, ifft, ifftshift, fftshift
from scipy.signal import lfilter, convolve

from .checks import is_number
from .data import scale_SI
from .math import find_closest, sinc, next_pow2, norm_var, gaussian
from .matrix import num_dim, num_dim2
from .signals import get_win
from ..kgrid import kWaveGrid
from ..kmedium import kWaveMedium


def single_sided_correction(func_fft: np.ndarray, fft_len: int, dim: int) -> np.ndarray:
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
            func_fft[1:-1] = func_fft[1:-1] * 2
        elif dim == 1:
            func_fft[:, 1:-1] = func_fft[:, 1:-1] * 2
        elif dim == 2:
            func_fft[:, :, 1:-1] = func_fft[:, :, 1:-1] * 2
        elif dim == 3:
            func_fft[:, :, :, 1:-1] = func_fft[:, :, :, 1:-1] * 2

    return func_fft


def spect(
    func: np.ndarray,
    Fs: float,
    dim: Optional[Union[int, str]] = "auto",
    fft_len: Optional[int] = 0,
    power_two: Optional[bool] = False,
    unwrap_phase: Optional[bool] = False,
    window: Optional[str] = "Rectangular",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the spectrum of a signal.

    Args:
        func: The signal to analyse.
        Fs: The sampling frequency in Hz.
        dim: The dimension over which the spectrum is calculated. Defaults to 'auto'.
        fft_len: The length of the FFT. If the set length is smaller than the signal length, the default value is used
                 instead (default = signal length).
        power_two: Whether the FFT length is forced to be the next highest power of 2 (default = False).
        unwrap_phase: Whether to unwrap the phase spectrum (default = False).
        window: (str) The window type used to filter the signal before the FFT is taken (default = 'Rectangular'). Any valid
                input types for get_win may be used.

    Returns:
        f: Frequency array
        func_as: Single-sided amplitude spectrum
        func_ps: Single-sided phase spectrum

    Raises:
        ValueError: If the input signal is scalar or has more than 4 dimensions.
    """

    # check the size of the input
    sz = func.shape

    # check input isn't scalar
    if np.size(func) == 1:
        raise ValueError("Input signal cannot be scalar.")

    # check input doesn't have more than 4 dimensions
    if len(sz) > 4:
        raise ValueError("Input signal must have 1, 2, 3, or 4 dimensions.")

    # automatically set dimension to first non - singleton dimension
    if dim == "auto":
        dim = np.argmax(np.array(sz) > 1)
        if sz[dim] <= 1:
            raise ValueError("All dimensions are singleton; unable to determine valid dimension.")

    # assign the number of points being analysed
    func_length = sz[dim]

    # set the length of the FFT
    if fft_len <= 0 or fft_len < func_length:
        if power_two:
            # find an appropriate FFT length of the form 2 ^ N that is equal to or
            # larger than the length of the input signal
            fft_len = 2 ** (next_pow2(func_length))
        else:
            # set the FFT length to the function length
            fft_len = func_length

    # window the signal, reshaping the window to be in the correct direction
    win, coherent_gain = get_win(func_length, type_=window, symmetric=False)
    win_shape = [1] * len(sz)
    win_shape[dim] = func_length
    win = np.reshape(win, tuple(win_shape))
    func = win * func

    # compute the fft using the defined FFT length, if fft_len >
    # func_length, the input signal is padded with zeros
    func_fft = np.fft.fft(func, n=fft_len, axis=dim)

    # correct for the magnitude scaling of the FFT and the coherent gain of the
    # window(note that the correction is equal to func_length NOT fft_len)
    epsilon = 1e-10  # Small value to prevent division by zero
    func_fft = func_fft / (func_length * coherent_gain + epsilon)

    # reduce to a single sided spectrum where the number of unique points for
    # even numbered FFT lengths is given by N / 2 + 1, and for odd(N + 1) / 2
    num_unique_pts = int(np.ceil((fft_len + 1) / 2))
    slicing = [slice(None)] * len(sz)
    slicing[dim] = slice(0, num_unique_pts)
    func_fft = func_fft[tuple(slicing)]

    func_fft = single_sided_correction(func_fft, fft_len, dim)

    # create the frequency axis variable
    f = np.arange(0, num_unique_pts) * Fs / fft_len

    # calculate the amplitude spectrum
    func_as = np.abs(func_fft)

    # calculate the phase spectrum
    func_ps = np.angle(func_fft)

    # unwrap the phase spectrum if required
    if unwrap_phase:
        func_ps = np.unwrap(func_ps, axis=dim)

    return f, func_as, func_ps


def extract_amp_phase(
    data: np.ndarray, Fs: float, source_freq: float, dim: Tuple[str, int] = "auto", fft_padding: int = 3, window: str = "Hanning"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the amplitude and phase information at a specified frequency from a vector or matrix of time series data.

    The amplitude and phase are extracted from the frequency spectrum, which is calculated using a windowed and zero
    padded FFT. The values are extracted at the frequency closest to source_freq. By default, the time dimension is set
    to the highest non-singleton dimension.

    Args:
        data: Matrix of time signals [s]
        Fs: Sampling frequency [Hz]
        source_freq: Frequency at which the amplitude and phase should be extracted [Hz]
        dim: The time dimension of the input data. If 'auto', the highest non-singleton dimension is used.
        fft_padding: The amount of zero padding to apply to the FFT.
        window: The windowing function to use for the FFT.

    Returns:
        A tuple of the amplitude, phase and frequency of the extracted signal.

    """

    # check for the dim input
    if dim == "auto":
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
        raise ValueError("dim must be 0, 1, 2, or 3")

    return amp.squeeze(), phase.squeeze(), f[f_index]


def brenner_sharpness(im):
    num_dim = im.ndim
    if num_dim == 2:
        # compute metric
        bren_x = (im[:-2, :] - im[2:, :]) ** 2
        bren_y = (im[:, :-2] - im[:, 2:]) ** 2
        s = np.sum(bren_x) + np.sum(bren_y)
    elif num_dim == 3:
        # compute metric
        bren_x = (im[:-2, :, :] - im[2:, :, :]) ** 2
        bren_y = (im[:, :-2, :] - im[:, 2:, :]) ** 2
        bren_z = (im[:, :, :-2] - im[:, :, 2:]) ** 2
        s = np.sum(bren_x) + np.sum(bren_y) + np.sum(bren_z)
    return s


def tenenbaum_sharpness(im):
    num_dim = im.ndim
    if num_dim == 2:
        # define the 2D sobel gradient operator
        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # compute metric
        s = (convolve(sobel, im) ** 2 + convolve(sobel.T, im) ** 2).sum()
    elif num_dim == 3:
        # define the 3D sobel gradient operator
        sobel3D = np.zeros((3, 3, 3))
        sobel3D[:, :, 0] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        sobel3D[:, :, 2] = -sobel3D[:, :, 0]

        # compute metric
        s = (
            convolve(im, sobel3D) ** 2
            + convolve(im, np.transpose(sobel3D, (2, 0, 1))) ** 2
            + convolve(im, np.transpose(sobel3D, (1, 2, 0))) ** 2
        ).sum()
    return s

    # TODO: get this passing the tests
    # NOTE: Walter thinks this is the proper way to do this, but it doesn't match the MATLAB version
    # num_dim = im.ndim
    # if num_dim == 2:
    #     # compute metric
    #     sx = sobel(im, axis=0, mode='constant')
    #     sy = sobel(im, axis=1, mode='constant')
    #     s = (sx ** 2) + (sy ** 2)
    #     s = np.sum(s)
    #
    # elif num_dim == 3:
    #     # compute metric
    #     sx = sobel(im, axis=0, mode='constant')
    #     sy = sobel(im, axis=1, mode='constant')
    #     sz = sobel(im, axis=2, mode='constant')
    #     s = (sx ** 2) + (sy ** 2) + (sz ** 2)
    #     s = np.sum(s)
    # else:
    #     raise ValueError("Invalid number of dimensions in im")


def sharpness(im: np.ndarray, mode: Optional[str] = "Brenner") -> float:
    """
    Returns a scalar metric related to the sharpness of a 2D or 3D image matrix.

    Args:
        im: The image matrix.
        metric: The metric to use. Defaults to "Brenner".

    Returns:
        A scalar sharpness metric.

    Raises:
        AssertionError: If `im` is not a NumPy array.

    References:
        B. E. Treeby, T. K. Varslot, E. Z. Zhang, J. G. Laufer, and P. C. Beard, "Automatic sound speed selection in
        photoacoustic image reconstruction using an autofocus approach," J. Biomed. Opt., vol. 16, no. 9, p. 090501, 2011.

    """

    assert isinstance(im, np.ndarray), "Argument im must be of type numpy array"

    if mode == "Brenner":
        metric = brenner_sharpness(im)
    elif mode == "Tenenbaum":
        metric = tenenbaum_sharpness(im)
    elif mode == "NormVariance":
        metric = norm_var(im)
    else:
        raise ValueError("Unrecognized sharpness metric passed. Valid values are ['Brenner', 'Tanenbaum', 'NormVariance']")

    return metric


def fwhm(f, x):
    """
    fwhm calculates the Full Width at Half Maximum (FWHM) of a positive
    1D input function f(x) with spacing given by x.


    Args:
        f:        f(x)
        x:        x

    Returns:
        FWHM of f(x) along with the position of the leading and trailing edges as a tuple

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
        zero_crossings = signs[0:-2] != signs[1:-1]
        zero_crossings_i = np.where(zero_crossings)[0]
        return [lin_interp(x, y, zero_crossings_i[0], half), lin_interp(x, y, zero_crossings_i[1], half)]

    hmx = half_max_x(x, f)
    fwhm_val = hmx[1] - hmx[0]

    return fwhm_val, tuple(hmx)


def gaussian_filter(
    signal: Union[np.ndarray, List[float]], Fs: float, frequency: float, bandwidth: float
) -> Union[np.ndarray, List[float]]:
    """
    Applies a frequency domain Gaussian filter with the
    specified center frequency and percentage bandwidth to the input
    signal. If the input signal is given as a matrix, the filter is
    applied to each matrix row.

    Args:
        signal:         Signal to filter [channel, samples]
        Fs:             Sampling frequency [Hz]
        frequency:      Center frequency of filter [Hz]
        bandwidth:      Bandwidth of filter in percentage

    Returns:
        The filtered signal

    """

    N = signal.shape[-1]
    if N % 2 == 0:
        f = np.arange(-N / 2, N / 2) * Fs / N
    else:
        f = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1) * Fs / N

    mean = frequency
    variance = (bandwidth / 100 * frequency / (2 * np.sqrt(2 * np.log(2)))) ** 2
    magnitude = 1

    # create double-sided Gaussain filter
    gfilter = np.fmax(gaussian(f, magnitude, mean, variance), gaussian(f, magnitude, -mean, variance))

    # add dimensions to filter to be broadcastable to signal shape
    if len(signal.shape) == 2:
        gfilter = gfilter[np.newaxis, :]

    # apply filter
    signal = np.real(ifft(ifftshift(gfilter * fftshift(fft(signal)))))

    return signal


def filter_time_series(
    kgrid: "kWaveGrid",
    medium: "kWaveMedium",
    signal: np.ndarray,
    ppw: Optional[int] = 3,
    rppw: Optional[int] = 0,
    stop_band_atten: Optional[int] = 60,
    transition_width: Optional[float] = 0.1,
    zerophase: Optional[bool] = False,
    plot_spectrums: Optional[bool] = False,
    plot_signals: Optional[bool] = False,
) -> np.ndarray:
    """
    Filters a time-domain signal using the Kaiser windowing method.

    The filter is designed to attenuate high-frequency noise in the signal while preserving
    the signal's important features. The filter design parameters can be adjusted to trade off
    between the amount of noise reduction and the amount of signal distortion.

    Args:
        kgrid: The kWaveGrid grid.
        medium: The kWavemedium.
        signal: The time-domain signal to filter.
        ppw: The minimum number of points per wavelength in the signal. This determines the
            minimum frequency that will be passed through the filter. Higher values of ppw
            result in a lower cut-off frequency and more noise reduction, but may also result
            in more signal distortion. Defaults to 3.
        rppw:  The number of points per wavelength in the smoothing ramp applied to the beginning
            of the signal. This can be used to reduce ringing artifacts caused by the sudden
            transition from the filtered signal to the unfiltered signal. Defaults to 0.
        stop_band_atten: The stop-band attenuation in dB. This determines the steepness of the
            filter's transition from the pass-band to the stop-band. Higher values result in a
            steeper transition and more noise reduction, but may also result in more signal
            distortion. Defaults to 60.
        transition_width: The transition width as a proportion of the sampling frequency. This
            determines the width of the transition region between the pass-band and the stop-band.
            Smaller values result in a narrower transition and more noise reduction, but may also
            result in more signal distortion. Defaults to 0.1.
        zerophase: Whether to implement the filter as a zero-phase filter. Zero-phase filtering
            can be used to preserve the phase information in the signal, which can be important
            for some applications. However, it may also result in more signal distortion.
            Defaults to False.
        plot_spectrums: Whether to plot the spectrums of the input and filtered signals.
            Defaults to False.
        plot_signals: Whether to plot the input and filtered signals. Defaults to False.

    Raises:
        ValueError: Checks correctness of passed arguments.
        NotImplementedError: Cannot currently plot anything.

    Returns:
        The filtered signal.

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
        raise TypeError("Input signal must be a vector.")

    # update the command line status
    logging.log(logging.INFO, "Filtering input signal...")

    # extract the time step
    assert not isinstance(kgrid.t_array, str) or kgrid.t_array != "auto", "kgrid.t_array must be explicitly defined."

    # compute the sampling frequency
    Fs = 1 / kgrid.dt

    # extract the minium sound speed
    if medium.sound_speed is not None:
        # for the fluid code, use medium.sound_speed
        c0 = medium.sound_speed.min()

    elif all(medium.is_defined("sound_speed_compression", "sound_speed_shear")):  # pragma: no cover
        # for the elastic code, combine the shear and compression sound speeds and remove zeros values
        ss = np.hstack([medium.sound_speed_compression, medium.sound_speed_shear])
        ss[ss == 0] = np.nan
        c0 = np.nanmin(ss)

        # cleanup unused variables
        del ss

    else:
        raise ValueError(
            "The input fields medium.sound_speed or medium.sound_speed_compression and medium.sound_speed_shear must " "be defined."
        )

    # extract the maximum supported frequency (two points per wavelength)
    f_max = kgrid.k_max_all * c0 / (2 * np.pi)

    # calculate the filter cut-off frequency
    filter_cutoff_f = 2 * f_max / ppw

    # calculate the wavelength of the filter cut-off frequency as a number of time steps
    filter_wavelength = (2 * np.pi / filter_cutoff_f) / kgrid.dt

    # filter the signal if required
    if ppw != 0:
        filtered_signal = apply_filter(
            signal,
            Fs,
            float(filter_cutoff_f),
            "LowPass",
            zero_phase=zerophase,
            stop_band_atten=float(stop_band_atten),
            transition_width=transition_width,
        )

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
    logging.log(logging.INFO, f"  maximum frequency supported by kgrid: {scale_SI(f_max)}Hz (2 PPW)")
    if ppw != 0:
        logging.log(logging.INFO, f"  filter cutoff frequency: {scale_SI(filter_cutoff_f)}Hz ({ppw} PPW)")
    if rppw != 0:
        logging.log(
            logging.INFO, f"  ramp frequency: {scale_SI(2 * np.pi / (2 * ramp_length * kgrid.dt))}Hz (ramp_points_per_wavelength PPW)"
        )
    logging.log(logging.INFO, "  computation complete.")

    # plot signals if required
    if plot_signals or plot_spectrums:
        raise NotImplementedError

    return filtered_signal


def apply_filter(
    signal: np.ndarray,
    Fs: float,
    cutoff_f: float,
    filter_type: str,
    zero_phase: Optional[bool] = False,
    transition_width: Optional[float] = 0.1,
    stop_band_atten: Optional[int] = 60,
) -> np.ndarray:
    """
    Filters an input signal using a FIR filter with Kaiser window coefficients based on the specified cut-off frequency and filter type.
    Both causal and zero phase filters can be applied.

    Args:
        signal: The input signal.
        Fs: The sampling frequency of the signal.
        cutoff_f: The cut-off frequency of the filter.
        filter_type: The type of filter to apply, either 'HighPass', 'LowPass' or 'BandPass'.
        zero_phase: Whether to apply a zero-phase filter. Defaults to False.
        transition_width: The transition width of the filter, as a proportion of the sampling frequency. Defaults to 0.1.
        stop_band_atten: The stop-band attenuation of the filter in dB. Defaults to 60.

    Returns:
        The filtered signal.

    """

    # for a bandpass filter, use applyFilter recursively
    if filter_type == "BandPass":
        assert isinstance(cutoff_f, list), "List of two frequencies required as for filter type 'BandPass'"
        assert len(cutoff_f) == 2, "List of two frequencies required as for filter type 'BandPass'"

        # apply the low pass filter
        func_filt_lp = apply_filter(
            signal, Fs, cutoff_f[1], "LowPass", stop_band_atten=stop_band_atten, transition_width=transition_width, zero_phase=zero_phase
        )

        # apply the high pass filter
        filtered_signal = apply_filter(
            func_filt_lp,
            Fs,
            cutoff_f[0],
            "HighPass",
            stop_band_atten=stop_band_atten,
            transition_width=transition_width,
            zero_phase=zero_phase,
        )

    else:
        # check filter type
        if filter_type == "LowPass":
            high_pass = False
        elif filter_type == "HighPass":
            high_pass = True
            cutoff_f = Fs / 2 - cutoff_f
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
        if "w" not in locals():
            # compute Kaiser window parameter beta
            if stop_band_atten > 50:
                beta = 0.1102 * (stop_band_atten - 8.7)
            elif stop_band_atten >= 21:
                beta = 0.5842 * (stop_band_atten - 21) ** 0.4 + 0.07886 * (stop_band_atten - 21)
            else:
                beta = 0

            # construct the Kaiser smoothing window w(n)
            m = np.arange(0, N)
            w = np.real(scipy.special.iv(0, np.pi * beta * np.sqrt(1 - (2 * m / N - 1) ** 2))) / np.real(scipy.special.iv(0, np.pi * beta))

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


def smooth(a: np.ndarray, restore_max: Optional[bool] = False, window_type: Optional[str] = "Blackman") -> np.ndarray:
    """
    Smooths a matrix.

    Args:
        a: The spatial distribution to smooth.
        restore_max: Boolean controlling whether the maximum value is restored after smoothing. Defaults to False.
        window_type: Shape of the smoothing window. Any valid inputs to get_win are supported. Defaults to 'Blackman'.

    Returns:
        a_sm: The smoothed matrix.

    """

    DEF_USE_ROTATION = True

    if a.dtype == bool:
        a = a.astype(int)

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

    win, _ = get_win(grid_size, type_=window_type, rotation=DEF_USE_ROTATION, symmetric=window_symmetry)
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
