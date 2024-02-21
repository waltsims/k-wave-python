import numpy as np


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
        ms = np.max(signal, axis=-1)
        if np.ndim(signal) == 2:
            ms = ms[:, np.newaxis]
        signal = ms * (np.log10(1 + cf * signal / ms) / np.log10(1 + cf))
    else:
        signal = np.log10(1 + cf * signal) / np.log10(1 + cf)
    return signal


def db(x):
    return 20 * np.log10(np.abs(x))


def apodize(distance, aperture, window):
    """
    Function that assigns different apodization to a set of pixels and elements

    """

    if window == "none":
        apod = np.ones(np.size(distance))
    elif window == "boxcar":
        apod = np.double(distance <= aperture / 2.0)
    elif window == "hanning":
        apod = np.double(distance <= aperture / 2) * (0.5 + 0.5 * np.cos(2 * np.pi * distance / aperture))
    elif window == "hamming":
        apod = np.double(distance <= aperture) / 2 * (0.53836 + 0.46164 * np.cos(2 * np.pi * distance / aperture))
    elif window == "tukey25":
        roll = 0.25
        apod = (distance < (aperture / 2 * (1 - roll))) + (distance > (aperture / 2 * (1 - roll))) * (distance < (aperture / 2)) * 0.5 * (
            1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2))
        )
    elif window == "tukey50":
        roll = 0.5
        apod = (distance < (aperture / 2 * (1 - roll))) + (distance > (aperture / 2 * (1 - roll))) * (distance < (aperture / 2)) * 0.5 * (
            1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2))
        )
    elif window == "tukey75":
        roll = 0.75
        apod = (distance < (aperture / 2 * (1 - roll))) + (distance > (aperture / 2 * (1 - roll))) * (distance < (aperture / 2)) * 0.5 * (
            1 + np.cos(2 * np.pi / roll * (distance / aperture - roll / 2 - 1 / 2))
        )
    else:
        raise ValueError("Unknown window type. Known types are: boxcar, hamming, hanning, tukey25, tukey50, tukey75.")

    return apod


def make_time_vector(num_samples, sampling_freq, time_offset):
    return np.linspace(start=0, num=num_samples, stop=num_samples / sampling_freq) + time_offset
