import logging
from math import floor

import numpy as np
import scipy
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


@typechecker
def get_win(
    N: Union[int, np.ndarray, Tuple[int, int], Tuple[int, int, int], List[Int[kt.ScalarLike, ""]]],
    # TODO: replace and refactor for scipy.signal.get_window
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    type_: str,  # TODO change this to enum in the future
    plot_win: bool = False,
    param: Optional[float] = None,
    rotation: bool = False,
    symmetric: Union[bool, Bool[np.ndarray, "N"]] = True,
    square: bool = False,
):
    """

    A frequency domain windowing function of specified type and dimensions.

     Args:
         N: Number of samples, [Nx] for 1D, [Nx, Ny] for 2D, [Nx, Ny, Nz] for 3D.
         type_: Window type. Supported values: 'Bartlett', 'Bartlett-Hanning', 'Blackman', 'Blackman-Harris',
                                               'Blackman-Nuttall', 'Cosine', 'Flattop', 'Gaussian', 'HalfBand',
                                               'Hamming', 'Hanning', 'Kaiser', 'Lanczos', 'Nuttall',
                                               'Rectangular', 'Triangular', 'Tukey'.
         plot_win: Boolean to display the window (default = False).
         param: Control parameter for Tukey, Blackman, Gaussian, and Kaiser windows: taper ratio (Tukey),
                                      alpha (Blackman, Kaiser), standard deviation (Gaussian)
                                      (default = 0.5, 0.16, 3 respectively).
         rotation: Boolean to create windows via rotation or outer product (default = False).
         symmetric: Boolean to make the window symmetrical (default = True).
                    Can also be a vector defining the symmetry in each matrix dimension.
         square: Boolean to force the window to be square (default = False).

     Returns:
         A tuple of (win, cg) where win is the window and cg is the coherent gain of the window.
    """

    def cosine_series(n: int, N: int, coeffs: List[float]) -> np.ndarray:
        """

        Sub-function to calculate a summed filter cosine series.

        Args:
            n: An integer representing the current index in the series.
            N: An integer representing the total number of terms in the series.
            coeffs: A list of floats representing the coefficients of the cosine terms.

        Returns:
            A numpy ndarray containing the calculated series.

        """
        series = coeffs[0]
        for index in range(1, len(coeffs)):
            series = series + (-1) ** index * coeffs[index] * np.cos(index * 2 * np.pi * n / (N - 1))
        return series.T

    # Check if N is either `int` or `list of ints`
    # assert isinstance(N, int) or isinstance(N, list) or isinstance(N, np.ndarray)
    N = np.array(N, dtype=int)
    N = N if np.size(N) > 1 else N.item()

    # Check if symmetric is either `bool` or `list of bools`
    # assert isinstance(symmetric, int) or isinstance(symmetric, list)
    symmetric = np.array(symmetric, dtype=bool)

    # Set default value for `param` if type is one of the special ones
    assert not plot_win, NotImplementedError("Plotting is not implemented.")
    if type_ == "Tukey":
        if param is None:
            param = 0.5
        param = np.clip(param, a_min=0, a_max=1)
    elif type_ == "Blackman":
        if param is None:
            param = 0.16
        param = np.clip(param, a_min=0, a_max=1)
    elif type_ == "Gaussian":
        if param is None:
            param = 0.5
        param = np.clip(param, a_min=0, a_max=0.5)
    elif type_ == "Kaiser":
        if param is None:
            param = 3
        param = np.clip(param, a_min=0, a_max=100)

    # if a non-symmetrical window is required, enlarge the window size (note,
    # this expands each dimension individually if symmetric is a vector)
    N = N + 1 * (1 - symmetric.astype(int))

    # if a square window is required, replace grid sizes with smallest size and
    # store a copy of the original size
    if square and (N.size != 1):
        N_orig = np.copy(N)
        L = min(N)
        N[:] = L

    # create the window
    if N.size == 1:
        # TODO: what should this behaviour be if N is a list of ints? make windows of multiple lengths?
        n = np.arange(0, N)

        # TODO: find failure cases in test suite when N is zero.
        # assert np.all(N) > 1, 'Signal length N must be greater than 1'

        if type_ == "Bartlett":
            win = (2 / (N - 1) * ((N - 1) / 2 - abs(n - (N - 1) / 2))).T
        elif type_ == "Bartlett-Hanning":
            win = (0.62 - 0.48 * abs(n / (N - 1) - 1 / 2) - 0.38 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == "Blackman":
            win = cosine_series(n, N, [(1 - param) / 2, 0.5, param / 2])
        elif type_ == "Blackman-Harris":
            win = cosine_series(n, N, [0.35875, 0.48829, 0.14128, 0.01168])
        elif type_ == "Blackman-Nuttall":
            win = cosine_series(n, N, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
        elif type_ == "Cosine":
            win = (np.cos(np.pi * n / (N - 1) - np.pi / 2)).T
        elif type_ == "Flattop":
            win = cosine_series(n, N, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
        elif type_ == "Gaussian":
            win = (np.exp(-0.5 * ((n - (N - 1) / 2) / (param * (N - 1) / 2)) ** 2)).T
        elif type_ == "HalfBand":
            win = np.ones(N)
            # why not to just round? => because rounding 0.5 introduces unexpected behaviour
            # round(0.5) should be 1 but it is 0
            ramp_length = round(N / 4 + 1e-8)
            ramp = (
                1 / 2
                + 9 / 16 * np.cos(np.pi * np.arange(1, ramp_length + 1) / (2 * ramp_length))
                - 1 / 16 * np.cos(3 * np.pi * np.arange(1, ramp_length + 1) / (2 * ramp_length))
            )
            if ramp_length > 0:
                win[0:ramp_length] = np.flip(ramp)
                win[-ramp_length:] = ramp
        elif type_ == "Hamming":
            win = (0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == "Hanning":
            win = (0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == "Kaiser":
            part_1 = scipy.special.iv(0, np.pi * param * np.sqrt(1 - (2 * n / (N - 1) - 1) ** 2))
            part_2 = scipy.special.iv(0, np.pi * param)
            win = part_1 / part_2
        elif type_ == "Lanczos":
            win = 2 * np.pi * n / (N - 1) - np.pi
            win = sinc(win + 1e-12).T
        elif type_ == "Nuttall":
            win = cosine_series(n, N, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
        elif type_ == "Rectangular":
            win = np.ones(N)
        elif type_ == "Triangular":
            win = (2 / N * (N / 2 - abs(n - (N - 1) / 2))).T
        elif type_ == "Tukey":
            win = np.ones((N, 1))
            index = np.arange(0, (N - 1) * param / 2 + 1e-8)
            param = param * N
            win[0 : len(index)] = 0.5 * (1 + np.cos(2 * np.pi / param * (index - param / 2)))[:, None]
            win[np.arange(-1, -len(index) - 1, -1)] = win[0 : len(index)]
            win = win.squeeze(axis=-1)
        else:
            raise ValueError(f"Unknown window type: {type_}")

        # trim the window if required
        if not symmetric:
            N -= 1
        win = win[0:N]
        win = np.expand_dims(win, axis=-1)

        # calculate the coherent gain
        cg = win.sum() / N
    elif N.size == 2:
        # create the 2D window
        if rotation:
            # create the window in one dimension using getWin recursively
            L = max(N)
            win_lin, _ = get_win(int(L), type_, param=param)
            win_lin = np.squeeze(win_lin)

            # create the reference axis
            radius = (L - 1) / 2
            ll = np.linspace(-radius, radius, L)

            # create the 2D window using rotation
            xx = np.linspace(-radius, radius, N[0])
            yy = np.linspace(-radius, radius, N[1])
            [x, y] = ndgrid(xx, yy)
            r = np.sqrt(x**2 + y**2)
            r[r > radius] = radius
            interp_func = scipy.interpolate.interp1d(ll, win_lin)
            win = interp_func(r)
            win[r <= radius] = interp_func(r[r <= radius])

        else:
            # create the window in each dimension using getWin recursively
            win_x, _ = get_win(int(N[0]), type_, param=param)
            win_y, _ = get_win(int(N[1]), type_, param=param)

            # create the 2D window using the outer product
            win = (win_y * win_x.T).T

        # trim the window if required
        N = N - 1 * (1 - np.array(symmetric).astype(int))
        win = win[0 : N[0], 0 : N[1]]

        # calculate the coherent gain
        cg = win.sum() / np.prod(N)
    elif N.size == 3:
        # create the 3D window
        if rotation:
            # create the window in one dimension using getWin recursively
            L = N.max()
            win_lin, _ = get_win(int(L), type_, param=param)

            # create the reference axis
            radius = (L - 1) / 2
            ll = np.linspace(-radius, radius, L)

            # create the 3D window using rotation
            xx = np.linspace(-radius, radius, N[0])
            yy = np.linspace(-radius, radius, N[1])
            zz = np.linspace(-radius, radius, N[2])
            [x, y, z] = ndgrid(xx, yy, zz)
            r = np.sqrt(x**2 + y**2 + z**2)
            r[r > radius] = radius

            win_lin = np.squeeze(win_lin)
            interp_func = scipy.interpolate.interp1d(ll, win_lin)
            win = interp_func(r)
            win[r <= radius] = interp_func(r[r <= radius])

        else:
            # create the window in each dimension using getWin recursively
            win_x, _ = get_win(int(N[0]), type_, param=param)
            win_y, _ = get_win(int(N[1]), type_, param=param)
            win_z, _ = get_win(int(N[2]), type_, param=param)

            # create the 2D window using the outer product
            win_2D = win_x * win_z.T

            # create the 3D window
            win = np.zeros((N[0], N[1], N[2]))
            for index in range(0, N[1]):
                win[:, index, :] = win_2D[:, :] * win_y[index]

        # trim the window if required
        N = N - 1 * (1 - np.array(symmetric).astype(int))
        win = win[0 : N[0], 0 : N[1], 0 : N[2]]

        # calculate the coherent gain
        cg = win.sum() / np.prod(N)
    else:
        raise ValueError("Invalid input for N, only 1-, 2-, and 3-D windows are supported.")

    # enlarge the window if required
    if square and (N.size != 1):
        L = N[0]
        win_sq = win
        win = np.zeros(N_orig)
        if N.size == 2:
            index1 = round((N[0] - L) / 2)
            index2 = round((N[1] - L) / 2)
            win[index1 : (index1 + L), index2 : (index2 + L)] = win_sq
        elif N.size == 3:
            index1 = floor((N_orig[0] - L) / 2)
            index2 = floor((N_orig[1] - L) / 2)
            index3 = floor((N_orig[2] - L) / 2)
            win[index1 : index1 + L, index2 : index2 + L, index3 : index3 + L] = win_sq

    return win, cg


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
            tone_burst = tone_burst * np.squeeze(get_win(len(tone_burst), type_="Tukey", param=0.05)[0])

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
    filter_sec, _ = get_win(filter_size, "Tukey", param=taper_ratio, rotation=True)

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
