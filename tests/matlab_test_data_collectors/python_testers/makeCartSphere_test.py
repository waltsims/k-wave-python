from kwave.utils.mapgen import make_cart_sphere

import logging
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeCartSphere():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeCartSphere')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        radius, num_points, center = recorded_data['params'][0]
        center = center[0]
        expected_value = recorded_data['sphere']

        sphere = make_cart_sphere(radius, num_points, center)

        assert np.allclose(expected_value, sphere)

    logging.log(logging.INFO, 'makeCartSphere(..) works as expected!')
