from kwave.kgrid import kWaveGrid
from kwave.utils import reorder_sensor_data, dotdict
from scipy.io import loadmat
import numpy as np
from pathlib import Path
import os


def test_reorderSensorData():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/reorderSensorData')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        mask_size = np.squeeze(recorded_data['mask_size'])
        kgrid_size = np.squeeze(recorded_data['kgrid_size'])
        kgrid_dx_dy = np.squeeze(recorded_data['kgrid_dx_dy'])

        sensor_data_size = np.squeeze(recorded_data['sensor_data_size'])
        mask = recorded_data['mask']
        sensor_data = recorded_data['sensor_data']
        expected_reordered_sensor_data = recorded_data['reordered_sensor_data']

        sensor = dotdict()
        sensor.mask = mask

        kgrid = kWaveGrid(kgrid_size, [kgrid_dx_dy, kgrid_dx_dy])

        calculated_reordered_sensor_data = reorder_sensor_data(kgrid, sensor, sensor_data)

        assert np.allclose(expected_reordered_sensor_data, calculated_reordered_sensor_data, equal_nan=True)

    print('reorder_sensor_data(..) works as expected!')
