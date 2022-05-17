from kwave.utils.misc import round_even, round_odd


def test_round_odd_down():
    assert round_odd(3.9) == 3


def test_round_odd_up():
    assert round_odd(2.1) == 3


def test_round_even_up():
    assert round_even(21.1) == 22


def test_round_even_down():
    assert round_even(22.9) == 22
