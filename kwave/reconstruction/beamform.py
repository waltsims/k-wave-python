import logging
import os
from typing import Tuple, Optional

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from uff import ChannelData
from uff.position import Position

from kwave.utils.conversion import cart2pol
from kwave.utils.data import scale_time
from kwave.utils.tictoc import TicToc
from .shifted_transform import ShiftedTransform
from .tools import make_time_vector, get_t0, get_origin_array, apodize


def beamform(channel_data: ChannelData) -> None:
    """

    Args:
        channel_data: shape => (1, 96, 32, 1585)

    Returns:

    """
    f_number = 1.2
    num_px_z = 256
    imaging_depth = 40e-3
    # apodization_window = 'boxcar'
    apodization_window = "none"
    number_samples = np.size(channel_data.data, axis=-1)

    # create depth vector
    z = np.linspace(0, imaging_depth, num_px_z)

    # allocate memory for beamformed image
    beamformed_data = np.zeros((len(z), len(channel_data.sequence)), dtype=complex)

    # hilbert transform rf data to get envelope
    channel_data.data = hilbert(channel_data.data, axis=3)

    # allocate memory for
    wave_origin_x = np.empty(len(channel_data.sequence))

    for e_id, event in enumerate(channel_data.sequence):
        # todo event.event should be event.event_id or event.key
        # todo make itteratable getter for events
        event = channel_data.unique_events[e_id]
        probe = event.receive_setup.probe
        sampling_freq = event.receive_setup.sampling_frequency
        # We assume one transmit wave per transmit event... hence 0 index
        transmit_wave = event.transmit_setup.transmit_waves[0]

        # make time vector
        time_vector = make_time_vector(num_samples=number_samples, sampling_freq=sampling_freq,
                                       time_offset=event.receive_setup.time_offset)

        # todo: make indexing 0 min and not 1 min
        wave_origin_x[e_id] = channel_data.unique_waves[transmit_wave.wave - 1].origin.position.x

        # todo: make position objects
        pixel_positions = np.stack([wave_origin_x[e_id] * np.ones(len(z)), np.zeros(len(z)), z]).T
        expanding_aperture = pixel_positions[:, 2] / f_number

        # time zero delays for spherical waves
        origin = get_origin_array(channel_data, transmit_wave)
        t0_point = get_t0(transmit_wave)

        # logging.log(logging.INFO,  origin, t0_point)

        transmit_distance = np.sign(pixel_positions[:, 2] - origin[2]) * \
                            np.sqrt(np.sum((pixel_positions - origin) ** 2, axis=1)) + \
                            np.abs(1.2 * t0_point[0])
        # np.sqrt(np.sum((origin - t0_point) ** 2))

        probe = channel_data.probes[probe - 1]
        # todo: why are element positions saved as transforms and not positions?
        transform = ShiftedTransform.deserialize(probe.transform.serialize())
        # todo: remove list from channel mapping. currently [[<element_number>,]...]

        # dataset.channel_data.unique_waves[transmit_wave.wave - 1].origin.position.x

        # event.transmit_setup.channel_mapping = np.arange(1, 33)  # Added by Farid
        plt.plot(transmit_distance)

        for element_number in event.transmit_setup.channel_mapping:
            element_number = element_number[0]  # Changed by Farid

            # todo: why are element positions saved as transformations?
            element_position = Position.deserialize(
                probe.element[element_number - 1].transform.translation.serialize())
            element_location = Position.deserialize(transform(element_position).serialize())

            pixel_element_lateral_distance = abs(pixel_positions[:, 0] - element_location[0])
            # logging.log(logging.INFO,  pixel_element_lateral_distance)
            receive_apodization = apodize(pixel_element_lateral_distance, expanding_aperture, apodization_window)

            # receive distance
            receive_distance = np.sqrt(np.sum((pixel_positions - np.array(element_location)) ** 2, axis=1))

            t0 = transmit_wave.time_offset

            # round trip delay
            delay = (transmit_distance + receive_distance) / channel_data.sound_speed + t0

            # beamformed data
            chan_id = element_number - 1 - event.transmit_setup.channel_mapping[0][0]  # tricky part
            signal = np.squeeze(channel_data.data[:, e_id, chan_id, :])
            interp = interp1d(x=time_vector, y=signal, kind='cubic', bounds_error=False, fill_value=0)
            beamformed_data[:, e_id] += np.squeeze(receive_apodization * interp(delay).T)

    # Envelope and plot
    envelope_beamformed_data = np.absolute(beamformed_data)
    compressed_beamformed_data = 20 * np.log10(envelope_beamformed_data / np.amax(envelope_beamformed_data) + 1e-12)

    plt.figure
    x_dis = 1e3 * wave_origin_x
    z_dis = 1e3 * z
    plt.imshow(compressed_beamformed_data, vmin=-60, vmax=0, cmap='Greys_r',
               extent=[min(x_dis), max(x_dis), max(z_dis), min(z_dis)])
    plt.xlabel('x[mm]', fontsize=12)
    plt.ylabel('z[mm]', fontsize=12)
    plt.title(channel_data.description)
    plt.colorbar()

    filename = "example_bmode.png"
    plt.savefig(os.path.join(os.getcwd(), filename))
    logging.log(logging.INFO,  f"Plot saved to {os.path.join(os.getcwd(), filename)}")

    pass


def focus(kgrid, input_signal, source_mask, focus_position, sound_speed):
    """
    focus Create input signal based on source mask and focus position.
    focus takes a single input signal and a source mask and creates an
    input signal matrix (with one input signal for each source point).
    The appropriate time delays required to focus the signals at a given
    position in Cartesian space are automatically added based on the user
    inputs for focus_position and sound_speed.

    Args:
         kgrid:             k-Wave grid object returned by kWaveGrid
         input_signal:      single time series input
         source_mask:       matrix specifying the positions of the time
                            varying source distribution (i.e., source.p_mask
                            or source.u_mask)
         focus_position:    position of the focus in Cartesian coordinates
         sound_speed:       scalar sound speed

    Returns:
         input_signal_mat:  matrix of time series following the source points
    """

    assert kgrid.t_array != 'auto', "kgrid.t_array must be defined."
    if isinstance(sound_speed, int):
        sound_speed = float(sound_speed)

    assert isinstance(sound_speed, float), "sound_speed must be a scalar."

    positions = [kgrid.x.flatten(), kgrid.y.flatten(), kgrid.z.flatten()]

    # filter_positions
    positions = [position for position in positions if (position != np.nan).any()]
    assert len(positions) == kgrid.dim
    positions = np.array(positions)

    if isinstance(focus_position, list):
        focus_position = np.array(focus_position)
    assert isinstance(focus_position, np.ndarray)

    dist = np.linalg.norm(positions[:, source_mask.flatten() == 1] - focus_position[:, np.newaxis])

    # distance to delays
    delay = int(np.round(dist / (kgrid.dt * sound_speed)))
    max_delay = np.max(delay)
    rel_delay = -(delay - max_delay)

    signal_mat = np.zeros((rel_delay.size, input_signal.size + max_delay))

    # for src_idx, delay in enumerate(rel_delay):
    #     signal_mat[src_idx, delay:max_delay - delay] = input_signal
    # signal_mat[rel_delay, delay:max_delay - delay] = input_signal

    logging.log(logging.WARN, "This method is not fully migrated, might be depricated and is untested.", PendingDeprecationWarning)
    return signal_mat


def scan_conversion(
        scan_lines: np.ndarray,
        steering_angles,
        image_size: Tuple[float, float],
        c0,
        dt,
        resolution: Optional[Tuple[int, int]]
) -> np.ndarray:
    if resolution is None:
        resolution = (256, 256)  # in pixels

    x_resolution, y_resolution = resolution

    # assign the inputs
    x, y = image_size

    # start the timer
    TicToc.tic()

    # update command line status
    logging.log(logging.INFO,  'Computing ultrasound scan conversion...')

    # extract a_line parameters
    Nt = scan_lines.shape[1]

    # calculate radius variable based on the sound speed in the medium and the
    # round trip distance
    r = c0 * np.arange(1, Nt + 1) * dt / 2  # [m]

    # create regular Cartesian grid to remap to
    pos_vec_y_new = np.linspace(0, 1, y_resolution) * y - y / 2
    pos_vec_x_new = np.linspace(0, 1, x_resolution) * x
    [pos_mat_x_new, pos_mat_y_new] = np.array(np.meshgrid(pos_vec_x_new, pos_vec_y_new, indexing='ij'))

    # convert new points to polar coordinates
    [th_cart, r_cart] = cart2pol(pos_mat_x_new, pos_mat_y_new)

    # TODO: move this import statement at the top of the file
    # Not possible now due to cyclic dependencies
    from kwave.utils.interp import interpolate2d_with_queries

    # below part has some modifications
    # we flatten the _cart matrices and build queries
    # then we get values at the query locations
    # and reshape the values to the desired size
    # These three steps can be accomplished in one step in Matlab
    # However, we don't want to add custom logic to the `interpolate2D_with_queries` method.

    # Modifications -start
    queries = np.array([r_cart.flatten(), th_cart.flatten()]).T

    b_mode = interpolate2d_with_queries(
        [r, 2 * np.pi * steering_angles / 360],
        scan_lines.T,
        queries,
        method='linear',
        copy_nans=False
    )
    image_size_points = (len(pos_vec_x_new), len(pos_vec_y_new))
    b_mode = b_mode.reshape(image_size_points)
    # Modifications -end

    b_mode[np.isnan(b_mode)] = 0

    # update command line status
    logging.log(logging.INFO,  f'  completed in {scale_time(TicToc.toc())}')

    return b_mode


def envelope_detection(signal):
    """
    envelopeDetection applies the Hilbert transform to extract the
    envelope from an input vector x. If x is a matrix, the envelope along
    the last axis.

    Args:
        signal:

    Returns:
        signal_envelope:

    """

    return np.abs(scipy.signal.hilbert(signal))
