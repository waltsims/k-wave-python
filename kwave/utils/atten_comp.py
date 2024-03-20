import logging
import numpy as np
from matplotlib import pyplot as plt
from beartype import beartype as typechecker
from jaxtyping import Float

from kwave.utils.conversion import db2neper
from kwave.utils.math import find_closest


# =========================================================================
# FITTING FUNCTION
# =========================================================================


@typechecker
def constlinfit(x: float, y: float, a: float, b: float, neg_penalty: float = 10):
    error = a * x + b - y
    error[error < 0] = error[error < 0] * neg_penalty
    return sum(abs(error))


@typechecker
def atten_comp(
    signal: Float[np.ndarray, "SensorIndex TimeIndex"],
    dt: float,
    c: int,
    alpha_0: float,
    y: float,
    display_updates: bool = False,
    distribution: str = "Rihaczek",
    energy_cutoff: float = 0.98,
    freq_multiplier: float = 2,
    filter_cutoff: str = "auto",
    fit_type: str = "spline",
    noise_cutoff: float = 0.03,
    num_splines: int = 40,
    plot_tfd: bool = False,
    plot_range: str = "auto",
    t0: int = 1,
    taper_ratio: float = 0.5,
):
    """

        Args:
            signal: time series to compensate, indexed as (sensor_index, time_index)
            dt: time step [s]
            c: sound speed [m/s]
            alpha_0: power law absorption prefactor [dB/(MHz^y cm)]
            y: power law absorption exponent [0 < y < 3, y != 1]
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
    # multiplier used to scale the auto-plot range
    PLOT_RANGE_MULT = 1.5

    # rotate input signal from (sensor_index, time_index) to (time_index,
    # sensor_index)
    signal = signal.T

    # extract signal characteristics
    N, num_signals = signal.shape

    if y == 1:
        raise ValueError("A power exponent [y] of 1 is not valid.")

    # convert absorption coefficient to nepers
    alpha_0 = db2neper(alpha_0, y)

    # update command line status
    if display_updates:
        logging.log(logging.INFO, "Applying time variant filter...")

    # check FitType input
    if fit_type == "mav":
        # define settings for moving average based on the
        # length of the input signals
        mav_terms = int(round(N * 1e-2))
        mav_terms = mav_terms + (mav_terms % 2)

    elif fit_type not in ["linear", "spline"]:
        # throw error for unknown input
        raise ValueError("Optional input " "FitType" " must be set to " "spline" ", " "mav" ", or " "linear" ".")

    # =========================================================================
    # COMPUTE AVERAGE TIME FREQUENCY DISTRIBUTION (TFD) OF INPUT
    # =========================================================================

    # sample frequency
    Fs = 1 / dt

    # create time and frequency axes
    t_array = dt * np.arange(N)

    # compute the double-sided frequency axis
    if N % 2 == 0:
        # N is even
        f_array = np.arange(-N / 2, N / 2) * Fs / N

    else:
        # N is odd
        f_array = np.arange(-(N - 1) / 2, (N - 1) / 2) * Fs / N

    # compute the TFD if required
    if filter_cutoff == "auto" or plot_tfd:
        # update display
        if display_updates:
            if num_signals > 1:
                logging.log(logging.INFO, "  calculating average time-frequency spectrum...")
            else:
                logging.log(logging.INFO, "  calculating time-frequency spectrum...")

        # compute the TFD of the input signal

        if distribution == "Rihaczek":
            tfd = np.outer(np.conj(np.fft.fft(signal[:, 0])), signal[:, 0])
            if num_signals > 1:
                for index in range(1, num_signals):
                    tfd += np.outer(np.conj(np.fft.fft(signal[:, index])), signal[:, index])
            inc = 2 * np.pi / N
            tfd *= np.exp(np.outer(-1j * np.arange(0, 2 * np.pi * (1 - 1 / N) + inc, inc), np.arange(N)))
            tfd = np.fft.fftshift(tfd, 0) / (N * num_signals)
        elif distribution == "Wigner":

            def qwigner2(x: Float[np.ndarray, "Dim1"], Fs: float):
                raise NotImplementedError

            tfd = qwigner2(signal[:, 0], Fs)
            if num_signals > 1:
                for index in range(1, num_signals):
                    tfd += qwigner2(signal[:, index], Fs)
            tfd /= num_signals

        # take the absolute value
        # tfd = abs(real(tfd));
        tfd = np.abs(tfd)

    # plot the time-frequency spectra of the time signal
    if plot_tfd:
        raise NotImplementedError

    # =========================================================================
    # FIND CUTOFF FREQUENCIES
    # =========================================================================

    @typechecker
    def findClosest(arr: Float[np.ndarray, "Dim1"], value: float):
        return (np.abs(arr - value)).argmin()

    if filter_cutoff == "auto":  # noqa: F821
        # update display
        if display_updates:
            logging.log(logging.INFO, "finding filter thresholds... ")

        f_array_hs = f_array[f_array >= 0]
        tfd_hs = 2 * tfd[f_array >= 0, :]

        threshold = noise_cutoff * np.max(tfd_hs)
        tfd_hs[tfd_hs < threshold] = 0

        tfd_int = np.cumsum(tfd_hs, axis=0)
        tfd_tot = np.sum(tfd_hs, axis=0)

        cutoff_index = np.zeros(t_array.shape, dtype=int)

        for tw_index in range(len(tfd_tot)):
            _, closes_idx = find_closest(tfd_int[:, tw_index] / tfd_tot[tw_index], energy_cutoff)
            assert len(closes_idx) == 1
            cutoff_index[tw_index] = closes_idx[0]

        cutoff_freq_array = f_array_hs[cutoff_index] * freq_multiplier

        if plot_tfd:
            plt.hold(True)
            plt.plot((np.arange(N) * dt * 1e6, cutoff_freq_array * 1e-6, "y-"))
            plt.plot((np.arange(N) * dt * 1e6, -cutoff_freq_array * 1e-6, "y-"))
            plt.draw()

        if fit_type == "linear":
            neg_penalty = 10
            x0 = np.array([-f_array_hs[-1] / N, f_array_hs[int(len(f_array_hs) / 2)]])
            opt_vals = opt.fmin(constlinfit, x0, args=(1, N, cutoff_freq_array, neg_penalty))  # noqa: F821
            x = np.array(list(range(1, N + 1)))
            cutoff_freq_array = opt_vals[0] * x + opt_vals[1]

        elif fit_type == "spline":
            pp = splinefit(list(range(1, N + 1)), cutoff_freq_array, num_splines, "r")  # noqa: F821
            cutoff_freq_array = ppval(pp, list(range(1, N + 1)))  # noqa: F821

        elif fit_type == "mav":
            cutoff_freq_array = np.convolve(cutoff_freq_array, np.ones(mav_terms) / mav_terms, mode="same")
            cutoff_freq_array = np.roll(cutoff_freq_array, int(-(mav_terms // 2)))

        elif fit_type == "off":
            pass
        else:
            raise ValueError("unknown fit_type setting")

    else:
        assert not isinstance(filter_cutoff, str)
        # set manual values
        cutoff_freq_array = (filter_cutoff[1] - filter_cutoff[0]) / (N - 1) * np.arange(1, N + 1) + filter_cutoff[0]

    # Constrain values outside frequency range
    cutoff_freq_array = np.where(cutoff_freq_array > f_array[-1], f_array[-1], cutoff_freq_array)

    # Constrain negative values
    cutoff_freq_array = np.maximum(cutoff_freq_array, 0)

    # Plot final threshold
    if plot_tfd:
        # Plot
        plt.plot(t_array * 1e6, cutoff_freq_array * 1e-6, "w--", label="cutoff_frequency")
        plt.plot(t_array * 1e6, -cutoff_freq_array * 1e-6, "w--", label="cutoff_frequency")

        # Set the plot range
        if isinstance(plot_range, (int, float)):
            plt.ylim(plot_range)
        elif plot_range == "auto" and (PLOT_RANGE_MULT * np.amax(cutoff_freq_array) < f_array[-1]):
            plt.ylim(PLOT_RANGE_MULT * np.amax(cutoff_freq_array) * [-1, 1] * 1e-6)
        plt.show()

    # =========================================================================
    # CREATE FILTER
    # =========================================================================

    if display_updates:
        logging.log(logging.INFO, "  creating time variant filter... ")

    # create distance vector accounting for the index of the laser pulse
    # relative to the start of the signals
    dist_vec = c * dt * (np.arange(N) - t0)
    dist_vec[dist_vec < 0] = 0

    # Check if f_array and dist_vec are valid
    assert f_array is not None and len(f_array) > 0, "f_array must have non-zero length."
    assert dist_vec is not None and len(dist_vec) > 0, "dist_vec must have non-zero length."

    # Create the time variant filter
    f_mat, dist_mat = np.meshgrid(f_array, dist_vec)

    assert y != 1, "A power exponent [y] of 1 is not valid."  # this is a duplicate assertion to attempt to remove a warning

    # Add conditionals or use np.where to manage zero and NaN
    part_1 = (2 * np.pi * np.abs(f_mat)) ** y
    part_2 = 1j * np.tan(np.pi * y / 2)
    part_3 = 2 * np.pi * f_mat
    part_4 = (2 * np.pi * np.abs(f_mat)) ** (y - 1)

    tv_filter = alpha_0 * dist_mat * (part_1 - part_2 * part_3 * part_4)

    # convert cutoff frequency to a window size
    N_win_array = np.floor((cutoff_freq_array / f_array[-1]) * N) - 1
    N_win_array[np.fmod(N_win_array, 2) == 1] = N_win_array[np.fmod(N_win_array, 2) == 1] + 1

    # loop through each time/distance and create a row of the filter
    for t_index in range(N):
        # get the filter cutoff freq
        cutoff_freq = cutoff_freq_array[t_index]
        N_win = int(N_win_array[t_index])

        if cutoff_freq != 0:
            # create window
            win = np.zeros(N)
            win_pos = int(np.ceil((N - N_win) / 2)) + 1
            win[win_pos : win_pos + N_win] = np.tukey(N_win, taper_ratio)

            # window row of tv_filter
            tv_filter[t_index, :] *= win
        else:
            tv_filter[t_index, :] = 0

    # take exponential
    tv_filter = np.exp(tv_filter)

    # shift frequency axis (filter_mat is created with 0 in center) and then
    # compute the inverse FFT
    tv_filter = np.real(np.fft.ifft(np.fft.ifftshift(tv_filter, axes=[1]), axis=1))

    # apply circular shift
    for t_index in range(0, N):
        tv_filter[t_index, :] = np.roll(tv_filter[t_index, :], t_index)

    # zero out lower and upper triangles
    tv_filter[np.tril_indices(N, -np.ceil(N / 2) + 1)] = 0
    tv_filter[np.triu_indices(N, np.ceil(N / 2) + 1)] = 0

    # apply the filter
    for index in range(num_signals):
        signal[:, index] = np.dot(tv_filter, signal[:, index])

    # rotate output signal from (time_index, sensor_index) back to
    # (sensor_index, time_index)
    signal = np.transpose(signal)

    return signal, tfd, cutoff_freq
