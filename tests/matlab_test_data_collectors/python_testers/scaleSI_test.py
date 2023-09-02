import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.data import scale_SI


def test_scale_si():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/scaleSI')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        x = np.squeeze(recorded_data['x'])
        expected_x_sc = recorded_data['x_sc'][0]
        expected_scale = recorded_data['scale'][0][0]
        expected_prefix = recorded_data['prefix']
        if len(expected_prefix) == 0:
            expected_prefix = ''
        else:
            expected_prefix = str(expected_prefix[0])

        expected_prefix_fullname = np.squeeze(recorded_data['prefix_fullname'])
        if expected_prefix_fullname.size == 0:
            expected_prefix_fullname = ''
        else:
            expected_prefix_fullname = str(expected_prefix_fullname)

        [x_sc, scale, prefix, prefix_fullname] = scale_SI(x)

        assert x_sc == expected_x_sc
        assert scale == expected_scale
        assert prefix == expected_prefix
        assert prefix_fullname == expected_prefix_fullname

        logging.log(logging.INFO,  'aaa')

        # calculated_reordered_data = reorder_binary_sensor_data(sensor_data, reorder_index)
        # assert np.allclose(expected_reordered_data, calculated_reordered_data, equal_nan=True)

    logging.log(logging.INFO,  'scale_si(..) works as expected!')
