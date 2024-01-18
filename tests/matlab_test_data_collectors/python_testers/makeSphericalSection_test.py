from kwave.utils.mapgen import make_spherical_section

import logging
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeSphericalSection():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeSphericalSection')

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath, simplify_cells=True)

        params = recorded_data['params']
        if len(params) == 2:
            radius, height = params
            radius, height = int(radius), int(height)
            width, plot_section, binary = None, False, False
        else:
            radius, height, width, plot_section, binary = recorded_data['params']
            radius, height, width, plot_section, binary = int(radius), int(height), int(width), bool(plot_section), bool(binary)
        expected_spherical_section = recorded_data['spherical_section']
        expected_distance_map = recorded_data['distance_map']

        spherical_section, distance_map = make_spherical_section(radius, height, width, plot_section, binary)

        assert np.allclose(expected_spherical_section, spherical_section)
        assert np.allclose(expected_distance_map, distance_map)

    logging.log(logging.INFO, 'make_spherical_section(..) works as expected!')
