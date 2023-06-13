
import numpy as np

from kwave.utils.math import round_odd, round_even, find_closest


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

