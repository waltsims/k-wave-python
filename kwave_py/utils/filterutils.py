import numpy as np
import scipy
from scipy.signal import lfilter

from .misc import sinc
from .conversionutils import scale_SI
from .checkutils import num_dim2


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
        raise ValueError('The input fields medium.sound_speed or medium.sound_speed_compression and medium.sound_speed_shear must be defined.')

    # extract the maximum supported frequency (two points per wavelength)
    f_max = kgrid.k_max_all * c0 / (2 * np.pi)

    # calculate the filter cut-off frequency
    filter_cutoff_f = 2 * f_max / points_per_wavelength

    # calculate the wavelength of the filter cut-off frequency as a number of time steps
    filter_wavelength = ((2 * np.pi / filter_cutoff_f) / kgrid.dt)

    # filter the signal if required
    if points_per_wavelength != 0:
        filtered_signal = applyFilter(signal, Fs, float(filter_cutoff_f), 'LowPass',
                                          'ZeroPhase', zero_phase, 'StopBandAtten', float(stop_band_atten),
                                          'TransitionWidth', transition_width, 'Plot', plot_spectrums)

    # add a start-up ramp if required
    if ramp_points_per_wavelength != 0:

        # calculate the length of the ramp in time steps
        ramp_length = round(ramp_points_per_wavelength * filter_wavelength / (2 * points_per_wavelength))

        # create the ramp
        ramp = (-np.cos( np.arange(0, ramp_length - 1 + 1) * np.pi / ramp_length ) + 1) / 2

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
        print(f'  ramp frequency: {scale_SI(2 * np.pi / (2 * ramp_length * kgrid.dt))}Hz (ramp_points_per_wavelength PPW)')
    print('  computation complete.')

    # plot signals if required
    if plot_signals:
        raise NotImplementedError

    return filtered_signal


def applyFilter(signal, Fs, cutoff_f, filter_type, *args):
    """
    %     applyFilter filters an input signal using filter. The FIR filter
    %     coefficients are based on a Kaiser window with the specified cut-off
    %     frequency and filter type ('HighPass', 'LowPass' or 'BandPass'). Both
    %     causal and zero phase filters can be applied.
    Args:
        signal:
        Fs:
        cutoff_f:
        filter_type:
        *args:

    Returns:

    """
    # set optional input defaults
    num_req_input_variables     = 4
    zero_phase                  = False
    transition_width            = 0.1  # as proportion of sampling frequency
    stop_band_atten             = 60   # [dB]
    plot_filter                 = False

    # replace with user defined values if provided
    for input_index in range(0, len(args), 2):
        if args[input_index] == 'Plot':
            plot_filter = args[input_index + 1]
        elif args[input_index] == 'StopBandAtten':
            stop_band_atten = args[input_index + 1]
        elif args[input_index] == 'TransitionWidth':
            transition_width = args[input_index + 1]
        elif args[input_index] == 'Window':
            w = args[input_index + 1]
        elif args[input_index] == 'ZeroPhase':
            zero_phase = args[input_index + 1]
        else:
            raise ValueError('Unknown optional input.')

    # store the specified cutoff frequency before modificiation for plotting
    if plot_filter:
        cutoff_f_plot = cutoff_f

    # for a bandpass filter, use applyFilter recursively
    if filter_type == 'BandPass':

        # apply the low pass filter
        func_filt_lp = applyFilter(signal, Fs, cutoff_f(2), 'LowPass', 'StopBandAtten', stop_band_atten, 'TransitionWidth', transition_width, 'ZeroPhase', zero_phase);

        # apply the high pass filter
        filtered_signal = applyFilter(func_filt_lp, Fs, cutoff_f(1), 'HighPass', 'StopBandAtten', stop_band_atten, 'TransitionWidth', transition_width, 'ZeroPhase', zero_phase);

    else:

        # check filter type
        if filter_type == 'LowPass':
            high_pass = False
        elif filter_type == 'HighPass':
            high_pass = True
            cutoff_f = (Fs/2 - cutoff_f)
        else:
            ValueError(f'Unknown filter type {filter_type}')

        # make sure input is the correct way around
        m, n = signal.shape
        if m > n:
            signal = signal.T

        # correct the stopband attenuation if a zero phase filter is being used
        if zero_phase:
            stop_band_atten = stop_band_atten/2

        # decide the filter order
        N = np.ceil((stop_band_atten - 7.95) / (2.285*(transition_width*np.pi)))
        N = int(N)

        # construct impulse response of ideal bandpass filter h(n), a sinc function
        fc = cutoff_f/Fs  # normalised cut-off
        n = np.arange(-N/2, N/2)
        h = 2*fc*sinc(2*np.pi*fc*n)

        # if no window is given, use a Kaiser window
        if 'w' not in locals():

            # compute Kaiser window parameter beta
            if stop_band_atten > 50:
                beta = 0.1102*(stop_band_atten - 8.7)
            elif stop_band_atten >= 21:
                beta = 0.5842*(stop_band_atten - 21)^0.4 + 0.07886*(stop_band_atten - 21)
            else:
                beta = 0

            # construct the Kaiser smoothing window w(n)
            m = np.arange(0, N)
            w = np.real(scipy.special.iv(0,np.pi*beta*np.sqrt(1-(2*m/N-1)**2)))/np.real(scipy.special.iv(0,np.pi*beta))

        # window the ideal impulse response with Kaiser window to obtain the FIR filter coefficients hw(n)
        hw = w * h

        # modify to make a high_pass filter
        if high_pass:
            hw = (-1*np.ones((1, len(hw))))**(np.arange(1, len(hw)))*hw

        # add some zeros to allow the reverse (zero phase) filtering room to work
        L = signal.size    # length of original input signal
        filtered_signal = np.hstack([np.zeros((1, N)), signal])

        # apply the filter
        filtered_signal = lfilter(hw, 1, filtered_signal);
        if zero_phase:
            filtered_signal = np.fliplr(lfilter(hw, 1, filtered_signal(np.arange(L+N, 1, -1))))

        # remove the part of the signal corresponding to the added zeros
        filtered_signal = filtered_signal[:, N:]

    # plot the amplitude spectrum and the filter if required
    if plot_filter:
        raise NotImplementedError

    return filtered_signal