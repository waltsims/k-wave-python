import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.data import Vector
from kwave.utils.mapgen import make_cart_arc


def test_makeCartArc():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeCartArc')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        arc_pos, radius, diameter, focus_pos, num_points = recorded_data['params'][0]
        arc_pos = Vector(arc_pos[0])
        radius = radius[0][0]
        diameter = diameter[0][0]
        focus_pos = Vector(np.asfarray(focus_pos[0]))
        num_points = num_points[0][0]
        expected_value = recorded_data['cart_arc']

        cart_arc = make_cart_arc(arc_pos, radius, diameter, focus_pos, num_points)

        assert np.allclose(expected_value, cart_arc), "File {} failed!".format(filepath)

    print('makeCartArc(..) works as expected!')
