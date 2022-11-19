from kwave.utils.misc import round_even, round_odd, find_closest, focused_bowl_oneil

import numpy as np


def test_round_odd_down():
    assert round_odd(3.9) == 3


def test_round_odd_up():
    assert round_odd(2.1) == 3


def test_round_even_up():
    assert round_even(21.1) == 22


def test_round_even_down():
    assert round_even(22.9) == 22


def test_find_closest():
    a = np.array([1, 2, 3, 4, 5, 6])
    a_close, idx_close = find_closest(a, 2.1)
    assert a_close == 2
    assert idx_close == (1,)


def test_focused_bowl_oneil():
    # define transducer parameters
    radius = 140e-3  # [m]
    diameter = 120e-3  # [m]
    velocity = 100e-3  # [m / s]
    frequency = 1e6  # [Hz]
    sound_speed = 1500  # [m / s]
    density = 1000  # [kg / m ^ 3]

    # define position vectors
    axial_position = np.arange(0, 250e-3 + 1e-4, 1e-4)  # [m]
    lateral_position = np.arange(-15e-3, 15e-3 + 1e-4, 1e-4)  # [m]

    # evaluate pressure
    [p_axial, p_lateral] = focused_bowl_oneil(radius, diameter,
                                              velocity, frequency, sound_speed, density,
                                              axial_position, lateral_position)
    # compare first and last vals
    assert np.isclose(p_axial[0], 1227.01736354) and np.isclose(p_axial[-1], 13036.24465513)
    assert np.isclose(p_lateral[0], 82805.42312207383) and np.isclose(p_lateral[-1], 82805.42312207383)
    assert np.isclose(p_lateral[5], 12880.47179330867)  # one final test value

    # Test passing one arg as none
    [_, p_lateral] = focused_bowl_oneil(radius, diameter,
                                        velocity, frequency, sound_speed, density,
                                        axial_positions=axial_position)
    assert p_lateral is None

    # Test passing the other arg as none
    [p_axial, _] = focused_bowl_oneil(radius, diameter,
                                      velocity, frequency, sound_speed, density,
                                      lateral_positions=lateral_position)

    assert p_axial is None
