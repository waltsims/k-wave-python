from copy import deepcopy
from math import floor
from typing import Union, List, Optional
from kwave.utils.checkutils import num_dim
from kwave.utils.conversionutils import scale_SI

import numpy as np
import scipy

from .misc import sinc, ndgrid, gaussian
from .conversionutils import db2neper

import math


def primefactors(n):
    # even number divisible
    factors = []
    while n % 2 == 0:
        factors.append(2),
        n = n / 2

    # n became odd
    for i in range(3, int(math.sqrt(n)) + 1, 2):

        while (n % i == 0):
            factors.append(i)
            n = n / i

    if n > 2:
        factors.append(n)

    return factors


def check_factors(min_number, max_number):
    """
        Return the maximum prime factor for a range of numbers.

        checkFactors loops through the given range of numbers and finds the
        numbers with the smallest maximum prime factors. This allows suitable
        grid sizes to be selected to maximise the speed of the FFT (this is
        fastest for FFT lengths with small prime factors). The output is
        printed to the command line, and a plot of the factors is generated.

    Args:
        min_number: integer specifying the lower bound of values to test
        max_number: integer specifying the upper bound of values to test

    Returns:

    """

    # extract factors
    facs = np.zeros(1, max_number - min_number)
    fac_max = facs
    for index in range(min_number, max_number):
        facs[index - min_number + 1] = len(primefactors(index))
        fac_max[index - min_number + 1] = max(primefactors(index))

    # compute best factors in range
    print('Numbers with a maximum prime factor of 2')
    ind = min_number + np.argwhere(fac_max == 2)
    print(ind)
    print('Numbers with a maximum prime factor of 3')
    ind = min_number + np.argwhere(fac_max == 3)
    print(ind)
    print('Numbers with a maximum prime factor of 5')
    ind = min_number + np.argwhere(fac_max == 5)
    print(ind)
    print('Numbers with a maximum prime factor of 7')
    ind = min_number + np.argwhere(fac_max == 7)
    print(ind)
    print('Numbers to avoid (prime numbers)')
    nums = np.arange(min_number, max_number)
    print(nums[fac_max == nums])


def check_stability(kgrid, medium):
    """
          checkStability calculates the maximum time step for which the k-space
          propagation models kspaceFirstOrder1D, kspaceFirstOrder2D and
          kspaceFirstOrder3D are stable. These models are unconditionally
          stable when the reference sound speed is equal to or greater than the
          maximum sound speed in the medium and there is no absorption.
          However, when the reference sound speed is less than the maximum
          sound speed the model is only stable for sufficiently small time
          steps. The criterion is more stringent (the time step is smaller) in
          the absorbing case.

          The time steps given are accurate when the medium properties are
          homogeneous. For a heterogeneous media they give a useful, but not
          exact, estimate.
    Args:
        kgrid: k-Wave grid object return by kWaveGrid
        medium: structure containing the medium properties

    Returns: the maximum time step for which the models are stable.
            This is set to Inf when the model is unconditionally stable.
    """
    # why? : this function was migrated from Matlab.
    # Matlab would treat the 'medium' as a "pass by value" argument.
    # In python argument is passed by reference and changes in this function will cause original data to be changed.
    # Instead of making significant changes to the function, we make a deep copy of the argument
    medium = deepcopy(medium)

    # define literals
    FIXED_POINT_ACCURACY = 1e-12

    # find the maximum wavenumber
    kmax = kgrid.k.max()

    # calculate the reference sound speed for the fluid code, using the
    # maximum by default which ensures the model is unconditionally stable
    reductions = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean
    }

    if medium.sound_speed_ref is not None:
        ss_ref = medium.sound_speed_ref
        if np.isscalar(ss_ref):
            c_ref = ss_ref
        else:
            try:
                c_ref = reductions[ss_ref](medium.sound_speed)
            except KeyError:
                raise NotImplementedError('Unknown input for medium.sound_speed_ref.')
    else:
        c_ref = reductions['max'](medium.sound_speed)

    # calculate the timesteps required for stability
    if medium.alpha_coeff is None or np.all(medium.alpha_coeff == 0):

        # =====================================================================
        # NON-ABSORBING CASE
        # =====================================================================

        medium.sound_speed = np.atleast_1d(medium.sound_speed)
        if c_ref >= medium.sound_speed.max():
            # set the timestep to Inf when the model is unconditionally stable
            dt_stability_limit = float('inf')

        else:
            # set the timestep required for stability when c_ref~=max(medium.sound_speed(:))
            dt_stability_limit = 2 / (c_ref * kmax) * np.asin(c_ref / medium.sound_speed.max())

    else:

        # =====================================================================
        # ABSORBING CASE
        # =====================================================================

        # convert the absorption coefficient to nepers.(rad/s)^-y.m^-1
        medium.alpha_coeff = db2neper(medium.alpha_coeff, medium.alpha_power)

        # calculate the absorption constant
        if medium.alpha_mode == 'no_absorption':
            absorb_tau = -2 * medium.alpha_coeff * medium.sound_speed ** (medium.alpha_power - 1)
        else:
            absorb_tau = np.array([0])

        # calculate the dispersion constant
        if medium.alpha_mode == 'no_dispersion':
            absorb_eta = 2 * medium.alpha_coeff * medium.sound_speed ** medium.alpha_power * np.tan(
                np.pi * medium.alpha_power / 2)
        else:
            absorb_eta = np.array([0])

        # estimate the timestep required for stability in the absorbing case by
        # assuming the k-space correction factor, kappa = 1 (note that
        # absorb_tau and absorb_eta are negative quantities)
        medium.sound_speed = np.atleast_1d(medium.sound_speed)

        temp1 = medium.sound_speed.max() * absorb_tau.min() * kmax ** (medium.alpha_power - 1)
        temp2 = 1 - absorb_eta.min() * kmax ** (medium.alpha_power - 1)
        dt_estimate = (temp1 + np.sqrt(temp1 ** 2 + 4 * temp2)) / (temp2 * kmax * medium.sound_speed.max())

        # use a fixed point iteration to find the correct timestep, assuming
        # now that kappa = kappa(dt), using the previous estimate as a starting
        # point

        # first define the function to iterate
        def kappa(dt):
            return sinc(c_ref * kmax * dt / 2)

        def temp3(dt):
            return medium.sound_speed.max() * absorb_tau.min() * kappa(dt) * kmax ** (medium.alpha_power - 1)

        def func_to_solve(dt):
            return (temp3(dt) + np.sqrt((temp3(dt)) ** 2 + 4 * temp2)) / (
                    temp2 * kmax * kappa(dt) * medium.sound_speed.max())

        # run the fixed point iteration
        dt_stability_limit = dt_estimate
        dt_old = 0
        while abs(dt_stability_limit - dt_old) > FIXED_POINT_ACCURACY:
            dt_old = dt_stability_limit
            dt_stability_limit = func_to_solve(dt_stability_limit)

    return dt_stability_limit


def add_noise(signal, snr, mode="rms"):
    """

    Args:
        signal (np.array):      input signal
        snr (float):            desired signal snr (signal-to-noise ratio) in decibels after adding noise
        mode (str):             'rms' (default) or 'peak'

    Returns:
        signal (np.array):      signal with augmented with noise. This behaviour differs from the k-Wave MATLAB implementation in that the SNR is nor returned.

    """
    if mode == "rms":
        reference = np.sqrt(np.mean(signal ** 2))
    elif mode == "peak":
        reference = max(signal)
    else:
        raise ValueError(f"Unknown parameter '{mode}' for input mode.")

    # calculate the standard deviation of the Gaussian noise
    std_dev = reference / (10 ** (snr / 20))

    # calculate noise
    noise = std_dev * np.random.randn(signal.size)

    # check the snr
    noise_rms = np.sqrt(np.mean(noise ** 2))
    snr = 20. * np.log10(reference / noise_rms)

    # add noise to the recorded sensor data
    signal = signal + noise

    return signal


def get_win(N: Union[int, List[int]],
            type_: str,  # TODO change this to enum in the future
            plot_win: bool = False,
            param: Optional[float] = None,
            rotation: bool = False,
            symmetric: bool = True,
            square: bool = False):
    """
        Return a frequency domain windowing function
        getWin returns a 1D, 2D, or 3D frequency domain window of the
        specified type of the given dimensions. By default, higher
        dimensional windows are created using the outer product. The windows
        can alternatively be created using rotation by setting the optional
        input 'Rotation' to true. The coherent gain of the window can also be
        returned.
    Args:
        N: - number of samples, use. N = Nx for 1D | N = [Nx Ny] for 2D | N = [Nx Ny Nz] for 3D
        type_: - window type. Supported values are
                           'Bartlett'
                           'Bartlett-Hanning'
                           'Blackman'
                           'Blackman-Harris'
                           'Blackman-Nuttall'
                           'Cosine'
                           'Flattop'
                           'Gaussian'
                           'HalfBand'
                           'Hamming'
                           'Hanning'
                           'Kaiser'
                           'Lanczos'
                           'Nuttall'
                           'Rectangular'
                           'Triangular'
                           'Tukey'
             plot_win:      - Boolean controlling whether the window is displayed
                           (default = false).

             param:     Control parameter for the Tukey, Blackman, Gaussian,
                           and Kaiser windows:

                           Tukey: taper ratio (default = 0.5)
                           Blackman: alpha (default = 0.16)
                           Gaussian: standard deviation (default = 0.5)
                           Kaiser: alpha (default = 3)

             rotation:  - Boolean controlling whether 2D and 3D windows are
                           created via rotation or the outer product (default =
                           false). Windows created via rotation will have edge
                           values outside the window radius set to the first
                           window value.

             symmetric: - Boolean controlling whether the window is symmetrical
                           (default = true). If set to false, a window of length N
                           + 1 is created and the first N points are returned. For
                           2D and 3D windows, 'Symmetric' can be defined as a
                           vector defining the symmetry in each matrix dimension.

             square:    - Boolean controlling whether the window is forced to
                           be square (default = false). If set to true and Nx
                           and Nz are not equal, the window is created using the
                           smaller variable, and then padded with zeros.

    Returns:
        win: the window
        cg: the coherent gain of the window

    """

    def cosineSeries(n, N, coeffs):
        """
            Sub-function to calculate a summed filter cosine series.
        Args:
            n:
            N:
            coeffs:

        Returns:

        """
        series = coeffs[0]
        for index in range(1, len(coeffs)):
            series = series + (-1) ** (index) * coeffs[index] * np.cos(index * 2 * np.pi * n / (N - 1))
        return series.T

    # Check if N is either `int` or `list of ints`
    # assert isinstance(N, int) or isinstance(N, list) or isinstance(N, np.ndarray)
    N = np.array(N, dtype=int)
    N = N if np.size(N) > 1 else int(N)

    # Check if symmetric is either `bool` or `list of bools`
    # assert isinstance(symmetric, int) or isinstance(symmetric, list)
    symmetric = np.array(symmetric, dtype=bool)

    # Set default value for `param` if type is one of the special ones
    assert not plot_win, NotImplementedError('Plotting is not implemented.')
    if type_ == 'Tukey':
        if param is None:
            param = 0.5
        param = np.clip(param, a_min=0, a_max=1)
    elif type_ == 'Blackman':
        if param is None:
            param = 0.16
        param = np.clip(param, a_min=0, a_max=1)
    elif type_ == 'Gaussian':
        if param is None:
            param = 0.5
        param = np.clip(param, a_min=0, a_max=0.5)
    elif type_ == 'Kaiser':
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
        n = np.arange(0, N)

        if type_ == 'Bartlett':
            win = (2 / (N - 1) * ((N - 1) / 2 - abs(n - (N - 1) / 2))).T
        elif type_ == 'Bartlett-Hanning':
            win = (0.62 - 0.48 * abs(n / (N - 1) - 1 / 2) - 0.38 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == 'Blackman':
            win = cosineSeries(n, N, [(1 - param) / 2, 0.5, param / 2])
        elif type_ == 'Blackman-Harris':
            win = cosineSeries(n, N, [0.35875, 0.48829, 0.14128, 0.01168])
        elif type_ == 'Blackman-Nuttall':
            win = cosineSeries(n, N, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
        elif type_ == 'Cosine':
            win = (np.cos(np.pi * n / (N - 1) - np.pi / 2)).T
        elif type_ == 'Flattop':
            win = cosineSeries(n, N, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
            ylim = [-0.2, 1]
        elif type_ == 'Gaussian':
            win = (np.exp(-0.5 * ((n - (N - 1) / 2) / (param * (N - 1) / 2)) ** 2)).T
        elif type_ == 'HalfBand':
            win = np.ones(N)
            # why not to just round? => because rounding 0.5 introduces unexpected behaviour
            # round(0.5) should be 1 but it is 0
            ramp_length = round(N / 4 + 1e-8)
            ramp = 1 / 2 + 9 / 16 * np.cos(np.pi * np.arange(1, ramp_length + 1) / (2 * ramp_length)) - 1 / 16 * np.cos(
                3 * np.pi * np.arange(1, ramp_length + 1) / (2 * ramp_length))
            if ramp_length > 0:
                win[0:ramp_length] = np.flip(ramp)
                win[-ramp_length:] = ramp
        elif type_ == 'Hamming':
            win = (0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == 'Hanning':
            win = (0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == 'Kaiser':
            part_1 = scipy.special.iv(0, np.pi * param * np.sqrt(1 - (2 * n / (N - 1) - 1) ** 2))
            part_2 = scipy.special.iv(0, np.pi * param)
            win = part_1 / part_2
        elif type_ == 'Lanczos':
            win = 2 * np.pi * n / (N - 1) - np.pi
            win = sinc(win + 1e-12).T
        elif type_ == 'Nuttall':
            win = cosineSeries(n, N, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
        elif type_ == 'Rectangular':
            win = np.ones(N)
        elif type_ == 'Triangular':
            win = (2 / N * (N / 2 - abs(n - (N - 1) / 2))).T
        elif type_ == 'Tukey':
            win = np.ones((N, 1))
            index = np.arange(0, (N - 1) * param / 2 + 1e-8)
            param = param * N
            win[0: len(index)] = 0.5 * (1 + np.cos(2 * np.pi / param * (index - param / 2)))[:, None]
            win[np.arange(-1, -len(index) - 1, -1)] = win[0:len(index)]
            win = win.squeeze(axis=-1)
        else:
            raise ValueError(f'Unknown window type: {type_}')

        # trim the window if required
        if not symmetric:
            N -= 1
        win = win[0:N]
        win = np.expand_dims(win, axis=-1)

        # calculate the coherent gain
        cg = win.sum() / N
    elif N.size == 2:
        input_options = {
            "param": param,
            "rotation": rotation,
            "symmetric": symmetric,
            "square": square
        }

        # create the 2D window
        if rotation:

            # create the window in one dimension using getWin recursively
            L = max(N)
            win_lin, _ = get_win(L, type_, param=param)
            win_lin = np.squeeze(win_lin)

            # create the reference axis
            radius = (L - 1) / 2
            ll = np.linspace(-radius, radius, L)

            # create the 2D window using rotation
            xx = np.linspace(-radius, radius, N[0])
            yy = np.linspace(-radius, radius, N[1])
            [x, y] = ndgrid(xx, yy)
            r = np.sqrt(x ** 2 + y ** 2)
            r[r > radius] = radius
            interp_func = scipy.interpolate.interp1d(ll, win_lin)
            win = interp_func(r)
            win[r <= radius] = interp_func(r[r <= radius])

        else:
            # create the window in each dimension using getWin recursively
            win_x, _ = get_win(N[0], type_, param=param)
            win_y, _ = get_win(N[1], type_, param=param)

            # create the 2D window using the outer product
            win = (win_y * win_x.T).T

        # trim the window if required
        N = N - 1 * (1 - np.array(symmetric).astype(int))
        win = win[0:N[0], 0:N[1]]

        # calculate the coherent gain
        cg = win.sum() / np.prod(N)
    elif N.size == 3:
        # create the 3D window
        if rotation:

            # create the window in one dimension using getWin recursively
            L = N.max()
            win_lin, _ = get_win(L, type_, param=param)

            # create the reference axis
            radius = (L - 1) / 2
            ll = np.linspace(-radius, radius, L)

            # create the 3D window using rotation
            xx = np.linspace(-radius, radius, N[0])
            yy = np.linspace(-radius, radius, N[1])
            zz = np.linspace(-radius, radius, N[2])
            [x, y, z] = ndgrid(xx, yy, zz)
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            r[r > radius] = radius

            win_lin = np.squeeze(win_lin)
            interp_func = scipy.interpolate.interp1d(ll, win_lin)
            win = interp_func(r)
            win[r <= radius] = interp_func(r[r <= radius])

        else:

            # create the window in each dimension using getWin recursively
            win_x, _ = get_win(N[0], type_, param=param)
            win_y, _ = get_win(N[1], type_, param=param)
            win_z, _ = get_win(N[2], type_, param=param)

            # create the 2D window using the outer product
            win_2D = (win_x * win_z.T)

            # create the 3D window
            win = np.zeros((N[0], N[1], N[2]))
            for index in range(0, N[1]):
                win[:, index, :] = win_2D[:, :] * win_y[index]

        # trim the window if required
        N = N - 1 * (1 - np.array(symmetric).astype(int))
        win = win[0:N[0], 0:N[1], 0:N[2]]

        # calculate the coherent gain
        cg = win.sum() / np.prod(N)
    else:
        raise ValueError('Invalid input for N, only 1-, 2-, and 3-D windows are supported.')

    # enlarge the window if required
    if square and (N.size != 1):
        L = N[0]
        win_sq = win
        win = np.zeros(N_orig)
        if N.size == 2:
            index1 = round((N[0] - L) / 2)
            index2 = round((N[1] - L) / 2)
            win[index1:(index1 + L), index2:(index2 + L)] = win_sq
        elif N.size == 3:
            index1 = floor((N_orig[0] - L) / 2)
            index2 = floor((N_orig[1] - L) / 2)
            index3 = floor((N_orig[2] - L) / 2)
            win[index1:index1 + L, index2:index2 + L, index3:index3 + L] = win_sq

    return win, cg


def toneBurst(sample_freq, signal_freq, num_cycles, envelope='Gaussian', plot_signal=False, signal_length=[],
              signal_offset=0):
    """
        Create an enveloped single frequency tone burst.
        toneBurst creates an enveloped single frequency tone burst for use in
        ultrasound simulations. If an array is given for the optional input
        'SignalOffset', a matrix of tone bursts is created where each row
        corresponds to a tone burst for each value of the 'SignalOffset'. If
        a value for the optional input 'SignalLength' is  given, the tone
        burst/s are zero padded to this length (in samples).
    Args:
        plot_signal:
        signal_offset:
        signal_length:
        plot:
        sample_freq: sampling frequency [Hz]
        signal_freq: frequency of the tone burst signal [Hz]
        num_cycles: number of sinusoidal oscillations
        envelope:
        OPTIONAL INPUTS:
            Optional 'string', value pairs that may be used to modify the default
            computational settings.

            'Envelope'      - Envelope used to taper the tone burst. Valid inputs
                              are:

                                  'Gaussian' (the default)
                                  'Rectangular'
                                  [num_ring_up_cycles, num_ring_down_cycles]

                              The last option generates a continuous wave signal
                              with a cosine taper of the specified length at the
                              beginning and end.

            'Plot'          - Boolean controlling whether the created tone
                              burst is plotted.
            'SignalLength'  - Signal length in number of samples, if longer
                              than the tone burst length, the signal is
                              appended with zeros.
            'SignalOffset'  - Signal offset before the tone burst starts in
                              number of samples.

    Returns: created tone burst

    """
    assert isinstance(signal_offset, int), "signal_offset must be integer"
    # TODO: make this more consistent. Only one type.
    assert isinstance(signal_length, list) or isinstance(signal_length, int), "signal_length must be integer"

    # calculate the temporal spacing
    dt = 1 / sample_freq  # [s]

    # create the tone burst
    tone_length = num_cycles / signal_freq  # [s]
    # We want to include the endpoint but only if it's divisible by the stepsize
    if tone_length % dt < 1e-18:
        tone_t = np.linspace(0, tone_length, int(tone_length / dt) + 1)
    else:
        tone_t = np.arange(0, tone_length, dt)

    tone_burst = np.sin(2 * np.pi * signal_freq * tone_t)
    tone_index = round(signal_offset)

    # check for ring up and ring down input
    if isinstance(envelope, list) or isinstance(envelope, np.ndarray):  # and envelope.size == 2:

        # assign the inputs
        num_ring_up_cycles, num_ring_down_cycles = envelope

        # check signal is long enough for ring up and down
        assert num_cycles >= (num_ring_up_cycles + num_ring_down_cycles), \
            'Input num_cycles must be longer than num_ring_up_cycles + num_ring_down_cycles.'

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
        tone_burst[0:up_ramp_length_points] = tone_burst[1:up_ramp_length_points] * up_ramp
        tone_burst[-down_ramp_length_points + 1:] = tone_burst[-down_ramp_length_points + 1:] * down_ramp

    else:

        # create the envelope
        if envelope == 'Gaussian':
            x_lim = 3
            window_x = np.arange(-x_lim, x_lim + 1e-8, 2 * x_lim / (len(tone_burst) - 1))
            window = gaussian(window_x, 1, 0, 1)
        elif envelope == 'Rectangular':
            window = np.ones_like(tone_burst)
        elif envelope == 'RingUpDown':
            raise NotImplementedError("RingUpDown not yet implemented")
        else:
            raise ValueError(f'Unknown envelope {envelope}.')

        # apply the envelope
        tone_burst = tone_burst * window

        # force the ends to be zero by applying a second window
        if envelope == 'Gaussian':
            tone_burst = tone_burst * np.squeeze(get_win(len(tone_burst), type_='Tukey', param=0.05)[0])

    # calculate the expected FWHM in the frequency domain
    # t_var = tone_length/(2*x_lim)
    # w_var = 1/(4*pi^2*t_var)
    # fw = 2 * sqrt(2 * log(2) * w_var)

    # create the signal with the offset tone burst
    tone_index = np.array([tone_index])
    signal_offset = np.array(signal_offset)
    if len(signal_length) == 0:
        signal = np.zeros((tone_index.size, signal_offset.max() + len(tone_burst)))
    else:
        signal = np.zeros(tone_index.size, signal_length)

    for offset in range(tone_index.size):
        signal[offset, tone_index[offset]:tone_index[offset] + len(tone_burst)] = tone_burst.T

    # plot the signal if required
    if plot_signal:
        raise NotImplementedError

    return signal


def reorder_binary_sensor_data(sensor_data: np.ndarray, reorder_index: np.ndarray):
    """

    Args:
        sensor_data: N x K
        reorder_index: N

    Returns:

    """
    reorder_index = np.squeeze(reorder_index)
    assert sensor_data.ndim == 2
    assert reorder_index.ndim == 1

    return sensor_data[reorder_index.argsort()]


def calc_max_freq(max_spat_freq, c):
    filter_cutoff_freq = max_spat_freq * c / (2 * np.pi)
    return filter_cutoff_freq


def freq2wavenumber(N, k_max, filter_cutoff, c, k_dim):
    """
    Args:
        N:
        k_max:
        filter_cutoff:
        c:
        k_dim:

    Returns:

    """
    k_cutoff = 2 * np.pi * filter_cutoff / c

    # set the alpha_filter size
    filter_size = round(N * k_cutoff / k_dim[-1])

    # check the alpha_filter size
    if filter_size > N:
        # set the alpha_filter size to be the same as the grid size
        filter_size = N
        filter_cutoff = k_max * c / (2 * np.pi)
    return filter_size, filter_cutoff


def get_alpha_filter(kgrid, medium, filter_cutoff, taper_ratio=0.5):
    """
     getAlphaFilter uses get_win to create a Tukey window via rotation to
     pass to the medium.alpha_filter input field of the first order
     simulation functions (kspaceFirstOrder1D, kspaceFirstOrder2D, and
     kspaceFirstOrder3D). This parameter is used to regularise time
     reversal image reconstruction when absorption compensation is
     included.

    Args:
        kgrid (kWaveGrid):
        medium (Medium):
        filter_cutoff (list): Any of the filter_cutoff inputs may be set to 'max' to set the cutoff frequency to the maximum frequency supported by the grid
        taper_ratio:

    Returns:
        alpha_filter:
    """

    dim = num_dim(kgrid.k)
    print(f'    taber ratio: {taper_ratio}')
    # extract the maximum sound speed
    c = max(medium.sound_speed)

    assert len(filter_cutoff) == dim, f"Input filter_cutoff must have {dim} elements for a {dim}D grid"

    # parse cutoff freqs
    filter_size = []
    for idx, freq in enumerate(filter_cutoff):
        if freq == 'max':
            filter_cutoff[idx] = calc_max_freq(kgrid.k_max[idx], c)
            filter_size_local = kgrid.N[idx]
        else:
            filter_size_local, filter_cutoff[idx] = freq2wavenumber(kgrid.N[idx], kgrid.k_max[idx], filter_cutoff[idx],
                                                                    c, kgrid.k[idx])
        filter_size.append(filter_size_local)

    # create the alpha_filter
    filter_sec, _ = get_win(filter_size, 'Tukey', param=taper_ratio, rotation=True)

    # enlarge the alpha_filter to the size of the grid
    alpha_filter = np.zeros(kgrid.N)
    indexes = [round((kgrid.N[idx] - filter_size[idx]) / 2) for idx in range(len(filter_size))]

    if dim == 1:
        alpha_filter[indexes[0]: indexes[0] + filter_size[0]]
    elif dim == 2:
        alpha_filter[indexes[0]: indexes[0] + filter_size[0], indexes[1]: indexes[1] + filter_size[1]] = filter_sec
    elif dim == 3:
        alpha_filter[indexes[0]: indexes[0] + filter_size[0], indexes[1]: indexes[1] + filter_size[1],
        indexes[2]:indexes[2] + filter_size[2]] = filter_sec

    dim_string = lambda cutoff_vals: "".join([str(scale_SI(co)[0]) + " Hz by " for co in cutoff_vals])
    # update the command line status
    print(f'  filter cutoff: ' + dim_string(filter_cutoff)[:-4] + '.')

    return alpha_filter
