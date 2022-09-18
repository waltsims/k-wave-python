from kwave.utils import reorder_binary_sensor_data
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_binary_sensor_data():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/reorderBinarySensorData')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        sensor_data = recorded_data['sensor_data']
        reorder_index = recorded_data['reorder_index']
        expected_reordered_data = recorded_data['reordered_data']

        calculated_reordered_data = reorder_binary_sensor_data(sensor_data, reorder_index)
        assert np.allclose(expected_reordered_data, calculated_reordered_data, equal_nan=True)

    print('reorder_binary_sensor_data(..) works as expected!')
