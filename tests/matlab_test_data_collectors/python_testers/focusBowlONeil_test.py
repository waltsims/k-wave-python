import os
from pathlib import Path

import numpy as np

from kwave.utils.mapgen import focused_bowl_oneil
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_focused_bowl_oneil():
    test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/focusBowlONeil.mat')
    reader = TestRecordReader(test_record_path)

    radius = reader.expected_value_of('radius')
    diameter = reader.expected_value_of('diameter')
    velocity = reader.expected_value_of('velocity')
    frequency = reader.expected_value_of('frequency')
    sound_speed = reader.expected_value_of('sound_speed')
    density = reader.expected_value_of('density')
    axial_positions = reader.expected_value_of('axial_position')
    lateral_positions = reader.expected_value_of('lateral_position')

    p_axial, p_lateral, p_axial_complex = focused_bowl_oneil(radius, diameter, velocity, frequency, sound_speed,
                                                             density, axial_positions=axial_positions,
                                                             lateral_positions=lateral_positions)

    assert np.allclose(p_axial, reader.expected_value_of('p_axial'))
    assert np.allclose(p_lateral, reader.expected_value_of('p_lateral'))
    assert np.allclose(p_axial_complex, reader.expected_value_of('p_axial_complex'))

    [_, p_lateral, _] = focused_bowl_oneil(radius, diameter,
                                           velocity, frequency, sound_speed, density,
                                           axial_positions=axial_positions)

    assert p_lateral is None

    [p_axial, _, p_axial_complex] = focused_bowl_oneil(radius, diameter, velocity, frequency, sound_speed, density,
                                                       lateral_positions=lateral_positions)

    assert p_axial is None
    assert p_axial_complex is None

    print('focused_bowl_oneil(..) works as expected!')
