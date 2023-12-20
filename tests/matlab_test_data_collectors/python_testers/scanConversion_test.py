import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.reconstruction.beamform import scan_conversion


def test_scanConversion():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/scanConversion')

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        scan_lines = recorded_data['scan_lines'].astype(float)
        steering_angles = np.squeeze(recorded_data['steering_angles'])
        image_size = np.squeeze(recorded_data['image_size'])
        c0 = float(recorded_data['c0'])
        dt = float(recorded_data['dt'])
        resolution = np.squeeze(recorded_data['resolution'])
        expected_b_mode = recorded_data['b_mode']

        calculated_b_mode = scan_conversion(
            scan_lines, steering_angles, image_size,
            c0, dt, resolution
        )

        assert np.allclose(expected_b_mode, calculated_b_mode, equal_nan=True)

    logging.log(logging.INFO, 'scan_conversion(..) works as expected!')
