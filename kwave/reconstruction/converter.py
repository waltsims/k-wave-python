import scipy
import uff
import numpy as np
from kwave import NotATransducer
from uff.linear_array import LinearArray
import time


def build_channel_data(sensor_data: np.ndarray,
                       kgrid,
                       not_transducer: NotATransducer,
                       sampling_frequency,
                       prf,
                       focal_depth):
    number_scan_lines = sensor_data.shape[1]
    input_signal = not_transducer.input_signal
    transducer = not_transducer.transducer

    x0 = np.arange(0, 96) * transducer.element_width * kgrid.dy
    x0 = x0 - (x0.max() / 2)

    unique_excitations = uff.excitation.Excitation(
        waveform=input_signal,
        sampling_frequency=sampling_frequency,
        pulse_shape='Gaussian'
    )

    probe = LinearArray(
        number_elements=transducer.number_elements * 4,
        pitch=transducer.element_width * kgrid.dy + transducer.element_spacing * kgrid.dx,
        element_width=transducer.element_width * kgrid.dy,
        element_height=transducer.element_length * kgrid.dx,
        transform=uff.transform.Transform(uff.transform.Translation(0, 0, 0), uff.transform.Rotation(0, 0, 0))
    )

    ####### Move to LinearArray.update

    if probe.pitch and probe.number_elements:

        # for all lines in this block, probe.element_geometry[0] is hard coded as 0. Should be dynamic!

        # update perimeter
        if probe.element_width and probe.element_height:

            if not probe.element_geometry:
                probe.element_geometry = [uff.element_geometry.ElementGeometry(
                    uff.perimeter.Perimeter([
                        uff.Position(0, 0, 0),
                        uff.Position(0, 0, 0),
                        uff.Position(0, 0, 0),
                        uff.Position(0, 0, 0),
                    ])
                )]

            if not probe.element_geometry[0].perimeter:
                probe.element_geometry.perimeter = uff.perimeter.Perimeter([
                    uff.Position(0, 0, 0),
                    uff.Position(0, 0, 0),
                    uff.Position(0, 0, 0),
                    uff.Position(0, 0, 0),
                ])

            probe.element_geometry[0].perimeter.position[0] = uff.Position(x=-probe.element_width / 2,
                                                                           y=-probe.element_height / 2, z=0)
            probe.element_geometry[0].perimeter.position[1] = uff.Position(x=probe.element_width / 2,
                                                                           y=-probe.element_height / 2, z=0)
            probe.element_geometry[0].perimeter.position[2] = uff.Position(x=probe.element_width / 2,
                                                                           y=probe.element_height / 2, z=0)
            probe.element_geometry[0].perimeter.position[3] = uff.Position(x=-probe.element_width / 2,
                                                                           y=probe.element_height / 2, z=0)

        # compute element position in the x-axis
        x0_local = np.arange(1, probe.number_elements + 1) * probe.pitch
        x0_local = x0_local - x0_local.mean()
        x0_local = x0_local.astype(np.float32)

        # create array of elements
        probe.element = []
        for n in range(probe.number_elements):
            element = uff.Element(
                transform=uff.Transform(
                    translation=uff.Translation(x=x0_local[n], y=0, z=0),
                    rotation=uff.Rotation(x=0, y=0, z=0)
                ),
                # linear array can only have a single perimenter impulse_response
                # impulse_response=[],  # probe.element_impulse_response,
                element_geometry=probe.element_geometry[0]
            )

            probe.element.append(element)

    ####### Move end

    unique_waves = []
    for n in range(number_scan_lines):
        wave = uff.Wave(
            wave_type=uff.WaveType.CONVERGING,
            origin=uff.SphericalWaveOrigin(
                position=uff.Position(x=x0[n], z=focal_depth)
            ),
            aperture=uff.Aperture(
                origin=uff.Position(x=x0[n]),
                fixed_size=transducer.element_length*kgrid.dx,  # wrong, correct one is in below
                # fixed_size=[active_aperture*el_pitch, el_height],
                window='rectangular'
            )
        )
        unique_waves.append(wave)

    unique_events = []
    lag = 1.370851370851371e-06  # (length(input_signal) / 2 + 1) * dt
    downsample_factor = 1
    first_sample_time = 650e-9 * np.ones((number_scan_lines, 1))  # CHANGE

    for n in range(number_scan_lines):
        # we select a time zero reference point: in this case the location of the first element fired
        arg_min = np.argmax(np.sqrt((x0[n] - x0) ** 2 + focal_depth ** 2))
        time_zero_reference_point = x0[arg_min]

        transmit_wave = uff.TransmitWave(
            wave=n,
            time_zero_reference_point=uff.TimeZeroReferencePoint(x=time_zero_reference_point, y=0, z=0),
            time_offset=lag,
            weight=1
        )

        # transmit and receive channel
        # channel_mapping = np.arange(n, number_elements + n).tolist()
        channel_mapping = np.arange(n, transducer.number_elements + n)[None, :].T  # shape: [number_elements x 1]
        transmit_setup = uff.TransmitSetup(
            probe=1,  # different from Matlab
            channel_mapping=channel_mapping,
            transmit_waves=[transmit_wave]  # should be list!
        )

        # active_elment_positions = channel_data.probes{1}.get_elements_centre();

        receive_setup = uff.ReceiveSetup(
            probe=1,  # different from Matlab
            time_offset=first_sample_time[n],
            channel_mapping=channel_mapping,
            sampling_frequency=sampling_frequency / downsample_factor
        )

        unique_event = uff.Event(
            transmit_setup=transmit_setup,
            receive_setup=receive_setup
        )

        unique_events.append(unique_event)

    sequence = []
    for n in range(len(unique_events)):
        sequence.append(uff.TimedEvent(
            event=n,
            time_offset=(n - 1) / prf
        ))

    channel_data = uff.channel_data.ChannelData(
        data=sensor_data,
        probes=[probe],
        unique_excitations=[unique_excitations],
        unique_waves=unique_waves,
        unique_events=unique_events,
        description='US B-Mode Linear Transducer Scan, kWave.',
        authors='kWave.py',
        sound_speed=not_transducer.sound_speed,
        system='kWave.py',
        country_code='DE',
        local_time=time.strftime('%Y-%m-%dTHH:MM:SS+00'),  # fix
        repetition_rate=prf / len(unique_events),
        sequence=sequence
    )
    return channel_data
