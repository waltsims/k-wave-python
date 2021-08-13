from copy import deepcopy
from math import floor

import numpy as np
import scipy

from .misc import sinc, ndgrid, gaussian
from .conversionutils import db2neper


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
            dt_stability_limit = 2/(c_ref * kmax) * np.asin(c_ref/medium.sound_speed.max())

    else:

        # =====================================================================
        # ABSORBING CASE
        # =====================================================================

        # convert the absorption coefficient to nepers.(rad/s)^-y.m^-1
        medium.alpha_coeff = db2neper(medium.alpha_coeff, medium.alpha_power)

        # calculate the absorption constant
        if medium.alpha_mode == 'no_absorption':
            absorb_tau = -2 * medium.alpha_coeff * medium.sound_speed**(medium.alpha_power - 1)
        else:
            absorb_tau = np.array([0])

        # calculate the dispersion constant
        if medium.alpha_mode == 'no_dispersion':
            absorb_eta = 2 * medium.alpha_coeff * medium.sound_speed**medium.alpha_power * np.tan(np.pi * medium.alpha_power / 2)
        else:
            absorb_eta = np.array([0])

        # estimate the timestep required for stability in the absorbing case by
        # assuming the k-space correction factor, kappa = 1 (note that
        # absorb_tau and absorb_eta are negative quantities)
        medium.sound_speed = np.atleast_1d(medium.sound_speed)

        temp1 = medium.sound_speed.max() * absorb_tau.min() * kmax**(medium.alpha_power - 1)
        temp2 = 1 - absorb_eta.min() * kmax**(medium.alpha_power - 1)
        dt_estimate = (temp1 + np.sqrt(temp1**2 + 4 * temp2)) / (temp2 * kmax * medium.sound_speed.max())

        # use a fixed point iteration to find the correct timestep, assuming
        # now that kappa = kappa(dt), using the previous estimate as a starting
        # point

        # first define the function to iterate
        def kappa(dt):
            return sinc(c_ref * kmax * dt / 2)

        def temp3(dt):
            return medium.sound_speed.max() * absorb_tau.min() * kappa(dt) * kmax**(medium.alpha_power - 1)

        def func_to_solve(dt):
            return (temp3(dt) + np.sqrt((temp3(dt))**2 + 4 * temp2)) / (temp2 * kmax * kappa(dt) * medium.sound_speed.max())

        # run the fixed point iteration
        dt_stability_limit = dt_estimate
        dt_old = 0
        while abs(dt_stability_limit - dt_old) > FIXED_POINT_ACCURACY:
            dt_old = dt_stability_limit
            dt_stability_limit = func_to_solve(dt_stability_limit)

    return dt_stability_limit


def get_win(N, type_, *args):
    """
        Return a frequency domain windowing function
        %     getWin returns a 1D, 2D, or 3D frequency domain window of the
        %     specified type of the given dimensions. By default, higher
        %     dimensional windows are created using the outer product. The windows
        %     can alternatively be created using rotation by setting the optional
        %     input 'Rotation' to true. The coherent gain of the window can also be
        %     returned.
    Args:
        N: - number of samples, use. N = Nx for 1D | N = [Nx Ny] for 2D | N = [Nx Ny Nz] for 3D
        type: - window type. Supported values are
        %                   'Bartlett'
        %                   'Bartlett-Hanning'
        %                   'Blackman'
        %                   'Blackman-Harris'
        %                   'Blackman-Nuttall'
        %                   'Cosine'
        %                   'Flattop'
        %                   'Gaussian'
        %                   'HalfBand'
        %                   'Hamming'
        %                   'Hanning'
        %                   'Kaiser'
        %                   'Lanczos'
        %                   'Nuttall'
        %                   'Rectangular'
        %                   'Triangular'
        %                   'Tukey'
        *args: OPTIONAL INPUTS:
        %
        %     'Plot'      - Boolean controlling whether the window is displayed
        %                   (default = false).
        %
        %     'Param' -     Control parameter for the Tukey, Blackman, Gaussian,
        %                   and Kaiser windows:
        %
        %                   Tukey: taper ratio (default = 0.5)
        %                   Blackman: alpha (default = 0.16)
        %                   Gaussian: standard deviation (default = 0.5)
        %                   Kaiser: alpha (default = 3)
        %
        %     'Rotation'  - Boolean controlling whether 2D and 3D windows are
        %                   created via rotation or the outer product (default =
        %                   false). Windows created via rotation will have edge
        %                   values outside the window radius set to the first
        %                   window value.
        %
        %     'Symmetric' - Boolean controlling whether the window is symmetrical
        %                   (default = true). If set to false, a window of length N
        %                   + 1 is created and the first N points are returned. For
        %                   2D and 3D windows, 'Symmetric' can be defined as a
        %                   vector defining the symmetry in each matrix dimension.
        %
        %     'Square'    - Boolean controlling whether the window is forced to
        %                   be square (default = false). If set to true and Nx
        %                   and Nz are not equal, the window is created using the
        %                   smaller variable, and then padded with zeros.

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

    if isinstance(N, int):
        N = np.array(N)

    # set usage defaults
    num_req_input_variables = 2
    plot_win = False
    rotation = False
    symmetric = True
    square = False
    ylim = [0, 1]
    if type_ == 'Tukey':
        param = 0.5
        param_ub = 1
        param_lb = 0
    elif type_ == 'Blackman':
        param = 0.16
        param_ub = 1
        param_lb = 0
    elif type_ == 'Gaussian':
        param = 0.5
        param_ub = 0.5
        param_lb = 0
    elif type_ == 'Kaiser':
        param = 3
        param_ub = 100
        param_lb = 0
    else:
        param = 0

    # replace with user defined values if provided
    if len(args) != 0:
        for input_index in range(0, len(args), 2):
            if args[input_index] == 'Plot':
                plot_win = args[input_index + 1]
            elif args[input_index] == 'Param':
                param = args[input_index + 1]
                if param > param_ub:
                    param = param_ub
                elif param <  param_lb:
                    param = param_lb
            elif args[input_index] == 'Rotation':
                rotation = args[input_index + 1]
                assert isinstance(rotation, bool), 'Optional input Rotation must be Boolean.'
            elif args[input_index] == 'Symmetric':
                # assign input
                symmetric = args[input_index + 1]

                # check type
                assert isinstance(symmetric, bool) or symmetric.dtype == bool, 'Optional input Symmetric must be Boolean.'

                # check size
                assert len(symmetric) in [1, len(N)], 'Optional input Symmetric must have 1 or numel(N) elements.'
            elif args[input_index] == 'Square':
                square = args[input_index + 1]
                assert isinstance(square, bool), 'Optional input Square must be Boolean.'
            else:
                raise AttributeError('Unknown optional input.')

    # set any required input options for recursive function calls
    input_options = {}
    if type_ in {'Tukey', 'Blackman', 'Kaiser', 'Gaussian'}:
        input_options = ['Param', param]

    # if a non-symmetrical window is required, enlarge the window size (note,
    # this expands each dimension individually if symmetric is a vector)
    N = N + 1 * (1 - np.array(symmetric).astype(int))

    # if a square window is required, replace grid sizes with smallest size and
    # store a copy of the original size
    if square and (N.size != 1):
        N_orig = N
        L = min(N)
        N[:] = L

    # create the window
    if N.size == 1:
        n = np.arange(0, N)

        if type_ == 'Bartlett':
            win = (2 / (N - 1) * ( (N - 1) / 2 - abs(n - (N - 1) / 2))).T
        elif type_ == 'Bartlett-Hanning':
            win = (0.62 - 0.48 * abs(n / (N - 1) - 1/2) - 0.38 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == 'Blackman':
            win = cosineSeries(n, N, [(1 - param)/2, 0.5, param/2])
        elif type_ == 'Blackman-Harris':
            win = cosineSeries(n, N, [0.35875, 0.48829, 0.14128, 0.01168])
        elif type_ == 'Blackman-Nuttall':
            win = cosineSeries(n, N, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
        elif type_ == 'Cosine':
            win = (np.cos(np.pi * n / (N - 1) - np.pi/2)).T
        elif type_ == 'Flattop':
            win = cosineSeries(n, N, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
            ylim = [-0.2, 1]
        elif type_ == 'Gaussian':
            win = (np.exp(-0.5 * ( (n - (N - 1) / 2) / (param * (N - 1) / 2))**2)).T
        elif type_ == 'HalfBand':
            win = np.ones((N, 1))
            ramp_length = round(N / 4)
            ramp = 1 / 2 + 9 / 16 * np.cos(np.pi * np.arange(0, ramp_length) / (2 * ramp_length)) - 1 / 16 * np.cos(
                3 * np.pi * np.arange(0, ramp_length) / (2 * ramp_length))
            win[0:ramp_length] = np.fliplr(ramp)
            win[- ramp_length+1:] = ramp
        elif type_ == 'Hamming':
            win = (0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == 'Hanning':
            win = (0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))).T
        elif type_ == 'Kaiser':
            win = (besseli(0, np.pi * param * np.sqrt(1 - (2 * n / (N - 1) - 1)**2)) / besseli(0, np.pi * param)).T
        elif type_ == 'Lanczos':
            win = sinc(2 * np.pi * n / (N - 1) - np.pi).T
        elif type_ == 'Nuttall':
            win = cosineSeries(n, N, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
        elif type_ == 'Rectangular':
            win = np.ones((N, 1))
        elif type_ == 'Triangular':
            win = (2 / N * (N / 2 - abs(n - (N - 1) / 2))).T
        elif type_ == 'Tukey':
            win = np.ones((N, 1))
            index = np.arange(0, N * param / 2)
            param = param * N
            win[0: len(index)] = 0.5 * (1 + np.cos(2 * np.pi / param * (index - param / 2)))[:, None]
            win[np.arange(-1, -len(index)-1, -1)] = win[0:len(index)]
            win = win.squeeze(axis=-1)
        else:
            raise ValueError(f'Unknown window type: {type_}')

        # trim the window if required
        if not symmetric:
            N -= 1
        win = win[0:N]
        win = np.expand_dims(win, axis=-1)

        # calculate the coherent gain
        cg = sum(win)/N
    elif N.size == 2:
        # create the 2D window
        if rotation:

            # create the window in one dimension using getWin recursively
            L = max(N)
            win_lin, _ = get_win(L, type_, *input_options)
            win_lin = np.squeeze(win_lin)

            # create the reference axis
            radius = (L - 1)/2
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
            win_x = get_win(N[0], type_, *input_options)
            win_y = get_win(N[1], type_, *input_options)

            # create the 2D window using the outer product
            win = (win_y * win_x.T).T

        # trim the window if required
        N = N - 1 * (1 - np.array(symmetric).astype(int))
        win = win[0:N[0], 0:N[1]]

        # calculate the coherent gain
        cg = win.sum()/np.prod(N)
    elif N.size == 3:
        # create the 3D window
        if rotation:

            # create the window in one dimension using getWin recursively
            L = N.max()
            win_lin, _ = get_win(L, type_, *input_options)

            # create the reference axis
            radius = (L - 1)/2
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
            win_x = get_win(N[0], type_, *input_options)
            win_y = get_win(N[1], type_, *input_options)
            win_z = get_win(N[2], type_, *input_options)

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
            index1 = round((N(1) - L)/2) + 1
            index2 = round((N(2) - L)/2) + 1
            win[index1:index1 + L, index2:index2 + L] = win_sq
        elif N.size == 3:
            index1 = floor((N_orig(1) - L)/2) + 1
            index2 = floor((N_orig(2) - L)/2) + 1
            index3 = floor((N_orig(3) - L)/2) + 1
            win[index1:index1 + L, index2:index2 + L, index3:index3 + L] = win_sq

    if plot_win:
        raise NotImplementedError

    return win, cg


def toneBurst(sample_freq, signal_freq, num_cycles, *args):
    """
        Create an enveloped single frequency tone burst.
        %     toneBurst creates an enveloped single frequency tone burst for use in
        %     ultrasound simulations. If an array is given for the optional input
        %     'SignalOffset', a matrix of tone bursts is created where each row
        %     corresponds to a tone burst for each value of the 'SignalOffset'. If
        %     a value for the optional input 'SignalLength' is  given, the tone
        %     burst/s are zero padded to this length (in samples).
    Args:
        sample_freq: sampling frequency [Hz]
        signal_freq: frequency of the tone burst signal [Hz]
        num_cycles: number of sinusoidal oscillations
        *args:
        % OPTIONAL INPUTS:
        %     Optional 'string', value pairs that may be used to modify the default
        %     computational settings.
        %
        %     'Envelope'      - Envelope used to taper the tone burst. Valid inputs
        %                       are:
        %
        %                           'Gaussian' (the default)
        %                           'Rectangular'
        %                           [num_ring_up_cycles, num_ring_down_cycles]
        %
        %                       The last option generates a continuous wave signal
        %                       with a cosine taper of the specified length at the
        %                       beginning and end.
        %
        %     'Plot'          - Boolean controlling whether the created tone
        %                       burst is plotted.
        %     'SignalLength'  - Signal length in number of samples, if longer
        %                       than the tone burst length, the signal is
        %                       appended with zeros.
        %     'SignalOffset'  - Signal offset before the tone burst starts in
        %                       number of samples.

    Returns: created tone burst

    """
    # set usage defaults
    num_req_input_variables = 3
    envelope = 'Gaussian'
    signal_length = []
    signal_offset = 0
    plot_signal = False

    # replace with user defined values if provided
    if len(args) != 0:
        for input_index in range(0, len(args), 2):
            if args[input_index] == 'Envolope':
                envelope = args[input_index + 1]
            elif args[input_index] == 'Plot':
                plot_signal = args[input_index + 1]
            elif args[input_index] == 'SignalOffset':
                signal_offset = args[input_index + 1]
                signal_offset = np.round(signal_offset).astype(np.int)  # force integer
            elif args[input_index] == 'SignalLength':
                signal_length = args[input_index + 1]
                signal_length = np.round(signal_length).astype(np.int)  # force integer
            else:
                raise ValueError('Unknown optional input.')

    # calculate the temporal spacing
    dt = 1 / sample_freq

    # create the tone burst
    tone_length = num_cycles / signal_freq
    # Matlab compatibility if-else statement
    # Basically, we want to include endpoint but only if it's divisible by the stepsize
    if tone_length % dt < 1e-18:
        tone_t = np.linspace(0, tone_length, int(tone_length / dt) + 1)
    else:
        tone_t = np.arange(0, tone_length, dt)
    tone_burst = np.sin(2 * np.pi * signal_freq * tone_t)[:, None]
    tone_index = round(signal_offset)

    # check for ring up and ring down input
    if (isinstance(envelope, list) or isinstance(envelope, np.ndarray)) and envelope.size == 2:

        # assign the inputs
        num_ring_up_cycles, num_ring_down_cycles = envelope

        # check signal is long enough for ring up and down
        assert num_cycles >= (num_ring_up_cycles + num_ring_down_cycles), \
            'Input num_cycles must be longer than num_ring_up_cycles + num_ring_down_cycles.'

        # get period
        period = 1 / signal_freq

        # create x-axis for ramp between 0 and pi
        up_ramp_length_points   = round(num_ring_up_cycles * period / dt)
        down_ramp_length_points = round(num_ring_down_cycles * period / dt)
        up_ramp_axis            = np.arange(0, np.pi + 1e-8, np.pi / (up_ramp_length_points - 1))
        down_ramp_axis          = np.arange(0, np.pi + 1e-8, np.pi / (down_ramp_length_points - 1))

        # create ramp using a shifted cosine
        up_ramp                 = (-np.cos(up_ramp_axis)   + 1) * 0.5
        down_ramp               = (np.cos(down_ramp_axis) + 1) * 0.5

        # apply the ramps
        tone_burst[0:up_ramp_length_points] = tone_burst[1:up_ramp_length_points] * up_ramp
        tone_burst[-down_ramp_length_points + 1:] = tone_burst[-down_ramp_length_points + 1:] * down_ramp

    else:

        # create the envelope
        if envelope == 'Gaussian':
            x_lim = 3
            window_x = np.arange(-x_lim, x_lim+1e-8, 2 * x_lim / (len(tone_burst) - 1))
            window = gaussian(window_x, 1, 0, 1)
        if envelope == 'Rectangular':
            window = np.ones_like(tone_burst)
        if envelope == 'RingUpDown':
            pass
        else:
            ValueError(f'Unknown envelope {envelope}.')

        # apply the envelope
        tone_burst = tone_burst * window[:, None]

        # force the ends to be zero by applying a second window
        if envelope == 'Gaussian':
            tone_burst = tone_burst * get_win(len(tone_burst), 'Tukey', 'Param', 0.05)[0]

    # calculate the expected FWHM in the frequency domain
    # t_var = tone_length/(2*x_lim);
    # w_var = 1/(4*pi^2*t_var);
    # fw = 2 * sqrt(2 * log(2) * w_var)

    # create the signal with the offset tone burst
    tone_index = np.array([tone_index])
    signal_offset = np.array(signal_offset)
    if len(signal_length) == 0:
        signal = np.zeros((tone_index.size, signal_offset.max() + len(tone_burst)))
    else:
        signal = np.zeros(tone_index.size, signal_length)

    for offset in range(tone_index.size):
        signal[offset, tone_index[offset]:tone_index[offset] + len(tone_burst)] = tone_burst[:, 0]

    # plot the signal if required
    if plot_signal:
        raise NotImplementedError

    return signal