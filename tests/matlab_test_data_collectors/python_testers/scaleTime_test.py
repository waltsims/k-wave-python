import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.data import scale_time


def test_scale_time():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/scaleTime')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        seconds = np.squeeze(recorded_data['seconds'])
        expected_time = str(recorded_data['time'][0])

        time = scale_time(seconds)
        assert time == expected_time

    logging.log(logging.INFO, 'scale_time(..) works as expected!')
