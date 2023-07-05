import os
from pathlib import Path

import numpy as np

from kwave.utils.mapgen import make_cart_disc
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_make_cart_disc():
    test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/makeCartDisc.mat')
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        params = reader.expected_value_of('params')
        disc_pos, radius, focus_pos, num_points, plot_disc, use_spiral = params
        focus_pos = focus_pos.astype(np.float64)
        disc_pos = disc_pos.astype(np.float64)
        radius = float(radius)
        coordinates = make_cart_disc(disc_pos, radius, focus_pos, num_points, plot_disc, use_spiral)
        assert np.allclose(coordinates, reader.expected_value_of('coordinates'), equal_nan=True)
        reader.increment()

    print('make_cart_disc(..) works as expected!')


test_make_cart_disc()
