import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from uff import UFF, ChannelData
from matplotlib import pyplot as plt
import kwave.reconstruction.tools as tools
from uff.position import Position
from kwave.reconstruction.shifted_transform import ShiftedTransform


def beamform(channel_data: ChannelData):
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
        time_vector = tools.make_time_vector(num_samples=number_samples, sampling_freq=sampling_freq,
                                       time_offset=event.receive_setup.time_offset)

        # todo: make indexing 0 min and not 1 min
        wave_origin_x[e_id] = channel_data.unique_waves[transmit_wave.wave - 1].origin.position.x

        # todo: make position objects
        pixel_positions = np.stack([wave_origin_x[e_id] * np.ones(len(z)), np.zeros(len(z)), z]).T
        expanding_aperture = pixel_positions[:, 2] / f_number

        # time zero delays for spherical waves
        origin = tools.get_origin_array(channel_data, transmit_wave)
        t0_point = tools.get_t0(transmit_wave)

        # print(origin, t0_point)

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
            # print(pixel_element_lateral_distance)
            receive_apodization = tools.apodize(pixel_element_lateral_distance, expanding_aperture, apodization_window)

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
    plt.show()
