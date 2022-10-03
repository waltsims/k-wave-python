from kwave.utils.conversionutils import scale_time

from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


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

    print('scale_time(..) works as expected!')
