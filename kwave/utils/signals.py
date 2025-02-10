import logging
from math import floor

import numpy as np
import scipy
from scipy.signal import get_window, windows as signal_windows
from numpy.fft import ifftshift, fft, ifft

from beartype import beartype as typechecker
from beartype.typing import Union, List, Optional, Tuple
from jaxtyping import Int, Bool

from .conversion import freq2wavenumber
from .data import scale_SI
from .mapgen import ndgrid
from .math import sinc, gaussian
from .matlab import matlab_mask, unflatten_matlab_mask, rem
from .matrix import broadcast_axis, num_dim

import kwave.utils.typing as kt


def add_noise(signal: np.ndarray, snr: float, mode="rms"):
    """
    Add Gaussian noise to a signal.

    Args:
        signal:      input signal
        snr:         desired signal snr (signal-to-noise ratio) in decibels after adding noise
        mode:        'rms' (default) or 'peak'

    Returns:
        Signal with augmented with noise. This behaviour differs from the k-Wave MATLAB implementation in that the SNR is nor returned.

    """
    if mode == "rms":
        reference = np.sqrt(np.mean(signal**2))
    elif mode == "peak":
        reference = np.max(signal)
    else:
        raise ValueError(f"Unknown parameter '{mode}' for input mode.")

    # calculate the standard deviation of the Gaussian noise
    std_dev = reference / (10 ** (snr / 20))

    # calculate noise
    noise = std_dev * np.random.randn(*signal.shape)

    # check the snr
    noise_rms = np.sqrt(np.mean(noise**2))
    snr = 20.0 * np.log10(reference / noise_rms)

    # add noise to the recorded sensor data
    signal = signal + noise

    return signal

def create_window(N: Union[int, List[int]], 
                 window: str,
                 param: Optional[float] = None,
                 rotation: bool = False,
                 symmetric: bool = True,
                 square: bool = False) -> Tuple[np.ndarray, float]:
    """
    Create a window using scipy.signal.windows with support for multi-dimensional windows.
    
    Args:
        N: Number of samples. For 1D windows, this is an integer. For multi-dimensional windows,
           this should be a list of integers specifying the size in each dimension.
        window: Window type (e.g., 'hann', 'hamming', 'blackman', etc.)
        param: Optional parameter for windows that support it (e.g., beta for kaiser)
        rotation: If True, create multi-dimensional windows via rotation instead of outer product
        symmetric: Make the window symmetrical
        square: Force multi-dimensional windows to be square
        
    Returns:
        Tuple of (window array, coherent gain)
    """
    # Map k-wave window names to scipy names
    window = {
        'Bartlett': 'bartlett',
        'Blackman': 'blackman',
        'Hamming': 'hamming',
        'Hanning': 'hann',
        'Kaiser': 'kaiser',
        'Rectangular': 'boxcar',
        'Triangular': 'triang',
        'Tukey': 'tukey',
        'Gaussian': 'gaussian',
        'Blackman-Harris': 'blackmanharris',
        'Flattop': 'flattop'
    }.get(window, window.lower())

    # Convert inputs to numpy arrays
    N = np.array(N, dtype=int)
    if square and N.size > 1:
        N = np.full_like(N, N.min())

    # Create base 1D window
    def get_1d_window(size):
        if window == 'gaussian':
            return signal_windows.gaussian(size, (param or 0.5) * size)
        elif window in ['kaiser', 'tukey']:
            return getattr(signal_windows, window)(size, param or (3 if window == 'kaiser' else 0.5))
        return getattr(signal_windows, window)(size)

    # Handle 1D case
    if N.size == 1:
        win = get_1d_window(int(N))
        return win[:, np.newaxis], win.sum() / N

    # Handle multi-dimensional case
    if rotation:
        # Create radially symmetric window
        max_size = N.max()
        radius = (max_size - 1) / 2
        window_1d = get_1d_window(max_size)
        
        # Create coordinate grid
        coords = [np.linspace(-radius, radius, n) for n in N]
        grid = np.meshgrid(*coords, indexing='ij')
        
        # Calculate radial distances
        r = np.sqrt(sum(x**2 for x in grid))
        r_norm = r / radius
        
        # Interpolate window values
        win = np.interp(r_norm.flatten(), np.linspace(0, 1, max_size), window_1d).reshape(r.shape)
    else:
        # Create separable window using outer products
        windows = [get_1d_window(n) for n in N]
        win = windows[0]
        for w in windows[1:]:
            win = win[..., np.newaxis] * w

    return win, win.sum() / win.size


def tone_burst(sample_freq, signal_freq, num_cycles, envelope="Gaussian", plot_signal=False, signal_length=0, signal_offset=0):
    """
    Create an enveloped single frequency tone burst.

    Args:
        sample_freq: sampling frequency in Hz
        signal_freq: frequency of the tone burst signal in Hz
        num_cycles: number of sinusoidal oscillations
        envelope: Envelope used to taper the tone burst. Valid inputs are:
            - 'Gaussian' (the default)
            - 'Rectangular'
            - [num_ring_up_cycles, num_ring_down_cycles]
                The last option generates a continuous wave signal with a cosine taper of the specified length at the beginning and end.
        plot: Boolean controlling whether the created tone burst is plotted.
        signal_length: Signal length in number of samples. If longer than the tone burst length, the signal is appended with zeros.
        signal_offset: Signal offset before the tone burst starts in number of samples.
                        If an array is given, a matrix of tone bursts is created where each row corresponds to
                        a tone burst for each value of the 'SignalOffset'.

    Returns:
        created tone burst

    """
    assert isinstance(signal_offset, int) or isinstance(signal_offset, np.ndarray), "signal_offset must be integer or array of integers"
    assert isinstance(signal_length, int), "signal_length must be integer"

    # calculate the temporal spacing
    dt = 1 / sample_freq  # [s]

    # create the tone burst
    tone_length = num_cycles / signal_freq  # [s]
    # We want to include the endpoint but only if it's divisible by the step-size.
    # Modulo operator is not stable, so multiple conditions included.
    # if ( (tone_length % dt) < 1e-18 or (np.abs(tone_length % dt - dt) < 1e-18) ):
    if rem(tone_length, dt) < 1e-18:
        tone_t = np.linspace(0, tone_length, int(tone_length / dt) + 1)
    else:
        tone_t = np.arange(0, tone_length, dt)

    tone_burst = np.sin(2 * np.pi * signal_freq * tone_t)
    tone_index = np.round(signal_offset)

    # check for ring up and ring down input
    if isinstance(envelope, list) or isinstance(envelope, np.ndarray):
        num_ring_up_cycles, num_ring_down_cycles = envelope

        # check signal is long enough for ring up and down
        assert num_cycles >= (
            num_ring_up_cycles + num_ring_down_cycles
        ), "Input num_cycles must be longer than num_ring_up_cycles + num_ring_down_cycles."

        # get period
        period = 1 / signal_freq

        # create x-axis for ramp between 0 and pi
        up_ramp_length_points = round(num_ring_up_cycles * period / dt)
        down_ramp_length_points = round(num_ring_down_cycles * period / dt)
        up_ramp_axis = np.arange(0, np.pi + 1e-8, np.pi / (up_ramp_length_points - 1))
        down_ramp_axis = np.arange(0, np.pi + 1e-8, np.pi / (down_ramp_length_points - 1))

        # create ramp using a shifted cosine
        up_ramp = (-np.cos(up_ramp_axis) + 1) * 0.5
        down_ramp = (np.cos(down_ramp_axis) + 1) * 0.5

        # apply the ramps
        tone_burst[0:up_ramp_length_points] = tone_burst[0:up_ramp_length_points] * up_ramp
        tone_burst[-down_ramp_length_points:] = tone_burst[-down_ramp_length_points:] * down_ramp

    else:
        # create the envelope
        if envelope == "Gaussian":
            x_lim = 3
            window_x = np.arange(-x_lim, x_lim + 1e-8, 2 * x_lim / (len(tone_burst) - 1))
            window = gaussian(window_x, 1, 0, 1)
        elif envelope == "Rectangular":
            window = np.ones_like(tone_burst)
        elif envelope == "RingUpDown":
            raise NotImplementedError("RingUpDown not yet implemented")
        else:
            raise ValueError(f"Unknown envelope {envelope}.")

        # apply the envelope
        tone_burst = tone_burst * window

        # force the ends to be zero by applying a second window
        if envelope == "Gaussian":
            tone_burst = tone_burst * np.squeeze(create_window(len(tone_burst), type_="Tukey", param=0.05)[0])

    # Convert tone_index and signal_offset to numpy arrays
    signal_offset = np.array(signal_offset)

    # Determine the length of the signal array
    signal_length = max(signal_length, signal_offset.max() + len(tone_burst))

    # Create the signal array with the correct size
    signal = np.zeros((np.atleast_1d(signal_offset).size, signal_length))

    # Add the tone burst to the signal array
    tone_index = np.atleast_1d(tone_index)

    if tone_index.size == 1:
        tone_index = int(np.squeeze(tone_index))
        signal[:, tone_index : tone_index + len(tone_burst)] = tone_burst.T
    else:
        for i, idx in enumerate(tone_index):
            signal[i, int(idx) : int(idx) + len(tone_burst)] = tone_burst

    # plot the signal if required
    if plot_signal:
        raise NotImplementedError

    return signal


def reorder_sensor_data(kgrid, sensor, sensor_data: np.ndarray) -> np.ndarray:
    """
    Reorders the sensor data based on the coordinates of the sensor points.

    Args:
        kgrid: The k-Wave grid object.
        sensor: The k-Wave sensor object.
        sensor_data: The sensor data to be reordered.

    Returns:
        np.ndarray of the reordered sensor data.

    Raises:
        ValueError: If the simulation is not 2D or the sensor is not defined as a binary mask.
    """
    # check simulation is 2D
    if kgrid.dim != 2:
        raise ValueError("The simulation must be 2D.")

    # check sensor.mask is a binary mask
    if sensor.mask.dtype != bool and set(np.unique(sensor.mask).tolist()) != {0, 1}:
        raise ValueError("The sensor must be defined as a binary mask.")

    # find the coordinates of the sensor points
    x_sensor = matlab_mask(kgrid.x, sensor.mask == 1)
    x_sensor = np.squeeze(x_sensor)
    y_sensor = matlab_mask(kgrid.y, sensor.mask == 1)
    y_sensor = np.squeeze(y_sensor)

    # find the angle of each sensor point (from the centre)
    angle = np.arctan2(-x_sensor, -y_sensor)
    angle[angle < 0] = 2 * np.pi + angle[angle < 0]

    # sort the sensor points in order of increasing angle
    indices_new = np.argsort(angle, kind="stable")

    # reorder the measure time series so that adjacent time series correspond
    # to adjacent sensor points.
    reordered_sensor_data = sensor_data[indices_new]
    return reordered_sensor_data


def reorder_binary_sensor_data(sensor_data: np.ndarray, reorder_index: np.ndarray):
    """
    Args:
        sensor_data: N x K
        reorder_index: N

    Returns:
        reordered sensor data

    """
    reorder_index = np.squeeze(reorder_index)
    assert sensor_data.ndim == 2
    assert reorder_index.ndim == 1

    return sensor_data[reorder_index.argsort()]


def calc_max_freq(max_spat_freq, c):
    filter_cutoff_freq = max_spat_freq * c / (2 * np.pi)
    return filter_cutoff_freq


def get_alpha_filter(kgrid, medium, filter_cutoff, taper_ratio=0.5):
    """
     get_alpha_filter uses get_win to create a Tukey window via rotation to
     pass to the medium.alpha_filter. This parameter is used to regularise time
     reversal image reconstruction when absorption compensation is included.

    Args:
        kgrid: simulation grid
        medium: simulation medium
        filter_cutoff: Any of the filter_cutoff inputs may be set to 'max' to set the cutoff frequency to
                       the maximum frequency supported by the grid
        taper_ratio: The taper_ratio input is used to control the width of the transition region between
                     the passband and stopband. The default value is 0.5, which corresponds to
                     a transition region of 50% of the filter width.

    Returns:
        alpha_filter

    """

    dim = num_dim(kgrid.k)
    logging.log(logging.INFO, f"    taper ratio: {taper_ratio}")
    # extract the maximum sound speed
    c = max(medium.sound_speed)

    assert len(filter_cutoff) == dim, f"Input filter_cutoff must have {dim} elements for a {dim}D grid"

    # parse cutoff freqs
    filter_size = []
    for idx, freq in enumerate(filter_cutoff):
        if freq == "max":
            filter_cutoff[idx] = calc_max_freq(kgrid.k_max[idx], c)
            filter_size_local = kgrid.N[idx]
        else:
            filter_size_local, filter_cutoff[idx] = freq2wavenumber(kgrid.N[idx], kgrid.k_max[idx], filter_cutoff[idx], c, kgrid.k[idx])
        filter_size.append(filter_size_local)

    # create the alpha_filter
    filter_sec, _ = create_window(filter_size, "Tukey", param=taper_ratio, rotation=True)

    # enlarge the alpha_filter to the size of the grid
    alpha_filter = np.zeros(kgrid.N)
    indexes = [round((kgrid.N[idx] - filter_size[idx]) / 2) for idx in range(len(filter_size))]

    if dim == 1:
        alpha_filter[indexes[0] : indexes[0] + filter_size[0]] = np.squeeze(filter_sec)
    elif dim == 2:
        alpha_filter[indexes[0] : indexes[0] + filter_size[0], indexes[1] : indexes[1] + filter_size[1]] = filter_sec
    elif dim == 3:
        alpha_filter[
            indexes[0] : indexes[0] + filter_size[0], indexes[1] : indexes[1] + filter_size[1], indexes[2] : indexes[2] + filter_size[2]
        ] = filter_sec

    def dim_string(cutoff_vals):
        return "".join([(str(scale_SI(co)[0]) + " Hz by ") for co in cutoff_vals])

    # update the command line status
    logging.log(logging.INFO, "  filter cutoff: " + dim_string(filter_cutoff)[:-4] + ".")

    return alpha_filter


def get_wave_number(Nx, dx, dim):
    if Nx % 2 == 0:
        # even
        nx = np.arange(start=-Nx / 2, stop=Nx / 2) / Nx
    else:
        nx = np.arange(start=-(Nx - 1) / 2, stop=(Nx - 1) / 2 + 1) / Nx

    kx = ifftshift((2 * np.pi / dx) * nx)

    return kx


def gradient_spect(f: np.ndarray, dn: List[float], dim: Optional[Union[int, List[int]]] = None, deriv_order: int = 1) -> np.ndarray:
    """
    gradient_spect calculates the gradient of an n-dimensional input matrix using the Fourier collocation spectral method.
    The gradient for singleton dimensions is returned as 0.

    Args:
        f: A numpy ndarray representing the input matrix.
        dn: A list of floats representing the grid spacings in each dimension.
        dim: An optional integer or list of integers representing the dimensions along which to calculate the gradient.
        deriv_order: An integer representing the order of the derivative to calculate. Default is 1.

    Returns:
        A numpy ndarray containing the calculated gradient.

    """

    # get size of the input function
    sz = f.shape

    # check if input is 1D or user defined input dimension is given
    if dim or len(sz) == 1:
        # check if a single dn value is given, if not, extract the required value
        if not (isinstance(dn, int) or isinstance(dn, float)):
            dn = dn[dim]

        # get the grid size along the specified dimension, or the longest dimension if 1D
        if max(sz) == np.prod(sz):
            dim = np.argmax(sz)
            Nx = sz[dim]
        else:
            Nx = sz[dim]

        # get the wave number
        kx = get_wave_number(Nx, dn, dim)

        # calculate derivative and assign output
        grads = np.real(ifft((1j * kx) ** deriv_order * fft(f, axis=dim), axis=dim))
    else:
        # logging.log(logging.WARN, "This implementation is not tested.")
        # get the wave number
        # kx = get_wave_number(sz(dim), dn[dim], dim)

        assert len(dn) == len(sz), ValueError(f"{len(sz)} values for dn must be specified for a {len(sz)}-dimensional input matrix.")

        grads = []
        # calculate the gradient for each non-singleton dimension
        for dim in range(num_dim(f)):
            # get the wave number
            kx = get_wave_number(sz[dim], dn[dim], dim)
            # calculate derivative and assign output
            # TODO: replace this with numpy broadcasting
            kx = broadcast_axis(kx, num_dim(f), dim)
            grads.append(np.real(ifft((1j * kx) ** deriv_order * fft(f, axis=dim), axis=dim)))

    return grads


def unmask_sensor_data(kgrid, sensor, sensor_data: np.ndarray) -> np.ndarray:
    # create an empty matrix
    if kgrid.k == 1:
        unmasked_sensor_data = np.zeros((kgrid.Nx, 1))
    elif kgrid.k == 2:
        unmasked_sensor_data = np.zeros((kgrid.Nx, kgrid.Ny))
    elif kgrid.k == 3:
        unmasked_sensor_data = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz))
    else:
        raise NotImplementedError

    # reorder input data
    flat_sensor_mask = (sensor.mask != 0).flatten("F")
    assignment_mask = unflatten_matlab_mask(unmasked_sensor_data, np.where(flat_sensor_mask)[0])
    # unmasked_sensor_data.flatten('F')[flat_sensor_mask] = sensor_data.flatten()
    unmasked_sensor_data[assignment_mask] = sensor_data.flatten()
    # unmasked_sensor_data[unflatten_matlab_mask(unmasked_sensor_data, sensor.mask != 0)] = sensor_data
    return unmasked_sensor_data


def create_cw_signals(t_array: np.ndarray, freq: float, amp: np.ndarray, phase: np.ndarray, ramp_length: int = 4) -> np.ndarray:
    """
    Generate a series of continuous wave (CW) signals based on the 1D or 2D input matrices `amp` and `phase`, where each signal
    is given by:

        amp[i, j] .* sin(2 * pi * freq * t_array + phase[i, j]);

    To avoid startup transients, a cosine tapered up-ramp is applied to the beginning of the signal. By default, the length
    of this ramp is four periods of the wave. The up-ramp can be turned off by setting the `ramp_length` to 0.

    Examples:

        # define sampling parameters
        f = 5e6
        T = 1/f
        Fs = 100e6
        dt = 1/Fs
        t_array = np.arange(0, 10*T, dt)

        # define amplitude and phase
        amp = get_win(9, 'Gaussian')
        phase = np.arange(0, 2*pi, 9).T

        # create signals and plot
        cw_signal = create_cw_signals(t_array, f, amp, phase)

    Args:
        t_array: A numpy ndarray representing the time values.
        freq: A float representing the frequency of the signals.
        amp: A numpy ndarray representing the amplitudes of the signals.
        phase: A numpy ndarray representing the phases of the signals.
        ramp_length: An optional integer representing the length of the cosine up-ramp in periods of the wave. Default is 4.

    Returns:
        A numpy ndarray containing the generated CW signals.
    """

    if len(phase) == 1:
        phase = phase * np.ones(amp.shape)

    if amp.ndim > 1:
        N1, N2 = amp.shape
    else:
        N1, N2 = amp.shape[0], 1

    # create input signals
    cw_signal = np.zeros((N1, N2, len(t_array)))

    # create signal
    for index1 in range(N1):
        for index2 in range(N2):
            if amp.ndim > 1:
                cw_signal[index1, index2, :] = amp[index1, index2] * np.sin(2 * np.pi * freq * t_array + phase[index1, index2])
            else:
                cw_signal[index1, index2, :] = amp[index1] * np.sin(2 * np.pi * freq * t_array + phase[index1])

    # apply ramp to avoid startup transients
    if ramp_length != 0:
        # get period and time step (assuming dt is constant)
        period = 1 / freq
        dt = t_array[1] - t_array[0]

        # create x-axis for ramp between 0 and pi
        ramp_length_points = round(ramp_length * period / dt)
        ramp_axis = np.linspace(0, np.pi, ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp = np.expand_dims(ramp, axis=(0, 1))

        # apply ramp to all signals simultaneously
        cw_signal[:, :, :ramp_length_points] = ramp * cw_signal[:, :, :ramp_length_points]

    # remove singleton dimensions if cw_signal has more than two dimensions
    if cw_signal.ndim > 2:
        cw_signal = np.squeeze(cw_signal)

    # if only a single amplitude and phase is given, force time to be the
    # second dimensions
    if amp.ndim == 1:
        cw_signal = np.reshape(cw_signal, (N1, -1))

    return cw_signal
