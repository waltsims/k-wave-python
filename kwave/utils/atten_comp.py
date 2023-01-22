import time

import numpy as np
from matplotlib import pyplot as plt

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_SI, scale_time
from kwave.utils.filters import next_pow2
from kwave.utils.matrix import expand_matrix
from kwave.utils.tictoc import TicToc


def atten_comp(
        signal, dt, c, alpha_0, y,
        display_updates=False,
        distribution='Rihaczek',
        energy_cutoff=0.98,
        freq_multiplier=2,
        filter_cutoff='auto',
        fit_type='spline',
        noise_cutoff=0.03,
        num_splines=40,
        plot_tfd=False,
        plot_range='auto',
        t0=1,
        taper_ratio=0.5,
):
    """

    Args:
        signal: time series to compensate, indexed as (sensor_index, time_index)
        dt: time step [s]
        c: sound speed [m/s]
        alpha_0: power law absorption prefactor [dB/(MHz^y cm)]
        y: power law absorption exponent [0 < y < 3, y ~= 1]
        display_updates: Boolean controlling whether command line updates
%       and compute time are printed to the command line
        distribution: default TF distribution
        energy_cutoff: cutoff frequency as a [%] of total energy
        freq_multiplier: used to increase the cutoff_f for a smoother filter
        filter_cutoff: automatically compute cutoff based on TFD
        fit_type: default fit type used for smooth distribution
        noise_cutoff: [%] of signal max, used to threshold the signals
        num_splines: used for fit_type = 'spline'
        plot_tfd: plot TFD and cutoff
        plot_range: plot range
        t0: index of laser pulse
        taper_ratio: taper ratio used for Tukey Window

    Returns:

    """
    # dynamic range used in TFD plot [dB]
    PLOT_DNR = 40
    # multiplier used to scale the auto-plot range
    PLOT_RANGE_MULT = 1.5

    # rotate input signal from (sensor_index, time_index) to (time_index,
    # sensor_index)
    signal = signal.T

    # extract signal characteristics
    N, num_signals = signal.shape

    # convert absorption coefficient to nepers
    alpha_0 = db2neper(alpha_0, y)

    # update command line status
    if display_updates:
        print('Applying time variant filter...')

    # check FitType input
    if fit_type == 'mav':

        # define settings for moving average based on the
        # length of the input signals
        mav_terms = int(round(N * 1e-2))
        mav_terms = mav_terms + (mav_terms % 2)

    elif fit_type != 'linear' or fit_type == 'spline':

        # throw error for unknown input
        raise ValueError('Optional input ''FitType'' must be set to ''spline'', ''mav'', or ''linear''.')

