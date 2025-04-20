import logging
import os
from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from kwave.utils.mapgen import focused_annulus_oneil
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_focused_annulus_oneil():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/focusAnnulusONeil.mat")
    reader = TestRecordReader(test_record_path)

    radius = reader.expected_value_of("radius")
    diameters = reader.expected_value_of("diameters")
    amplitude = reader.expected_value_of("amplitude")
    source_phase = reader.expected_value_of("source_phase")
    frequency = reader.expected_value_of("frequency")
    sound_speed = reader.expected_value_of("sound_speed")
    density = reader.expected_value_of("density")
    axial_positions = reader.expected_value_of("axial_position")

    p_axial = focused_annulus_oneil(
        radius,
        diameters,
        amplitude / (sound_speed * density),
        source_phase,
        frequency,
        sound_speed,
        density,
        axial_positions=axial_positions,
    )

    assert np.allclose(p_axial, reader.expected_value_of("p_axial"))

    p_axial = focused_annulus_oneil(
        radius,
        diameters.T,
        amplitude / (sound_speed * density),
        source_phase,
        frequency,
        sound_speed,
        density,
        axial_positions=axial_positions,
    )

    assert np.allclose(p_axial, reader.expected_value_of("p_axial"))

    logging.log(logging.INFO, "focused_annulus_oneil(..) works as expected!")

    with pytest.raises(BeartypeCallHintParamViolation):
        focused_annulus_oneil(
            radius,
            diameters[0, :],
            amplitude / (sound_speed * density),
            source_phase,
            frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )

    # Test phase out of range
    with pytest.raises(ValueError):
        focused_annulus_oneil(
            radius,
            diameters,
            amplitude / (sound_speed * density),
            np.ones_like(source_phase) * -8,
            frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )

    # Test negative radius
    with pytest.raises(ValueError):
        focused_annulus_oneil(
            -radius,
            diameters,
            amplitude / (sound_speed * density),
            source_phase,
            frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )

    # Test negative diameter
    with pytest.raises(ValueError):
        focused_annulus_oneil(
            radius,
            -diameters,
            amplitude / (sound_speed * density),
            source_phase,
            frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )
    # Test inf diameter
    with pytest.raises(ValueError):
        focused_annulus_oneil(
            radius,
            diameters * np.inf,
            amplitude / (sound_speed * density),
            source_phase,
            frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )
    # Test inf amplitude
    with pytest.raises(ValueError):
        focused_annulus_oneil(
            radius,
            diameters,
            np.inf * amplitude / (sound_speed * density),
            source_phase,
            frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )

    # Test inf frequency
    with pytest.raises(ValueError):
        focused_annulus_oneil(
            radius,
            diameters,
            amplitude / (sound_speed * density),
            source_phase,
            np.inf * frequency,
            sound_speed,
            density,
            axial_positions=axial_positions,
        )
