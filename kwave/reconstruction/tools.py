import numpy as np
from uff import Position


def log_compression(signal, cf, normalize=False):
    """
    Log compress an input signal.

    Args:
        signal: signal to be log compressed
        cf: compression factor
        normalize (bool): when true, signal is normalized before compression

   Returns: signal: log-compressed signal
    """
    if normalize:
        ms = max(signal)
        signal = ms * (np.log10(1 + cf * signal / ms) / np.log10(1 + cf))
    else:
        signal = np.log10(1 + cf * signal) / np.log10(1 + cf)
    return signal


def apodize(distance, aperture, window):
    """
    Function that assigns different apodization to a set of pixels and elements

    """


    if window == 'none':
        apod = np.ones(np.size(distance))
    elif window == 'boxcar':
        apod = np.double(distance <= aperture / 2.0)
    elif window == 'hanning':
        apod = np.double(distance <= aperture / 2) * (0.5 + 0.5 * np.cos(2 * np.pi * distance / aperture))
    elif window == 'hamming':
        apod = np.double(distance <= aperture) / 2 * (0.53836 + 0.46164 * np.cos(2 * np.pi * distance / aperture))
    elif window == 'tukey25':
        roll = 0.25
        apod = (distance < (aperture / 2 * (1 - roll))) + (distance > (aperture / 2 * (1 - roll))) * (
                    distance < (aperture / 2)) * 0.5 * (1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2)))
    elif window == 'tukey50':
        roll = 0.5
        apod = (distance < (aperture / 2 * (1 - roll))) + (distance > (aperture / 2 * (1 - roll))) * (
                    distance < (aperture / 2)) * 0.5 * (1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2)))
    elif window == 'tukey75':
        roll = 0.75
        apod = (distance < (aperture / 2 * (1 - roll))) + (distance > (aperture / 2 * (1 - roll))) * (
                    distance < (aperture / 2)) * 0.5 * (1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2)))
    else:
        raise ValueError('Unknown window type. Known types are: boxcar, hamming, hanning, tukey25, tukey50, tukey75.')

    return apod


def get_t0(transmit_wave):
    serialized_tx_wave = transmit_wave.time_zero_reference_point.serialize()
    return np.array(Position.deserialize(serialized_tx_wave))


def get_origin_array(channel_data, transmit_wave):
    serialized_origin = channel_data.unique_waves[transmit_wave.wave - 1].origin.position.serialize()
    return np.array(
        Position.deserialize(serialized_origin))


def make_time_vector(num_samples, sampling_freq, time_offset):
    return np.linspace(start=0, num=num_samples,
                       stop=num_samples / sampling_freq) + time_offset
