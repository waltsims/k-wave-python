from kwave.utils.maputils import make_cart_circle

from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeCartCircle():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeCartCircle')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)


        radius, num_points, center, arc_angle = recorded_data['params'][0]
        expected_value = recorded_data['circle']

        circle = make_cart_circle(radius, num_points, center, arc_angle)

        assert np.allclose(expected_value, circle)

    print('makeDisc(..) works as expected!')
