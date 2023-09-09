import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.data import Vector
from kwave.utils.mapgen import make_cart_circle


def test_makeCartCircle():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeCartCircle')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        if len(recorded_data['params'][0]) == 4:
            radius, num_points, center, arc_angle = recorded_data['params'][0]
            center = Vector(center[0])
            num_points = num_points[0][0]
            radius = radius[0][0]
            arc_angle = arc_angle[0][0]
            if np.isclose(arc_angle, 2 * np.pi):
                arc_angle = 2 * np.pi

            circle = make_cart_circle(radius, num_points, center, arc_angle)
        else:
            radius, num_points, center = recorded_data['params'][0]
            center = Vector(center[0])
            num_points = num_points[0][0]
            radius = radius[0][0]
            circle = make_cart_circle(radius, num_points, center)
        expected_value = recorded_data['circle']


        assert np.allclose(expected_value, circle)

    logging.log(logging.INFO, 'makeCartCircle(..) works as expected!')
