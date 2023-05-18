import os

import numpy as np

from kwave.utils.mapgen import make_cart_rect
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_make_cart_rect():
    # test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/makeCartRect.mat')
    test_record_path = os.path.join('/Users/farid/workspace/black_box_testing', 'collectedValues/makeCartRect.mat')
    reader = TestRecordReader(test_record_path)

    for i in range(len(reader)):
        print(i)
        params = reader.expected_value_of('params')
        rect_pos, Lx, Ly, theta, num_points, plot_rect = params
        coordinates = make_cart_rect(rect_pos, Lx, Ly, theta, num_points, plot_rect)
        assert np.allclose(coordinates, reader.expected_value_of('coordinates'))
        reader.increment()

    print('make_cart_rect(..) works as expected!')
