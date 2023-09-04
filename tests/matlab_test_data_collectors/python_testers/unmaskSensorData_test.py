import logging
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from scipy.io import loadmat

from kwave.utils.signals import unmask_sensor_data


def test_unmask_sensor_data():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/unmaskSensorData')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        # Read recorded data
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        data_dim = recorded_data['data_dim'][0]
        sensor_mask = recorded_data['sensor_mask']
        sensor_data = recorded_data['sensor_data']
        expected_unmasked_sensor_data = recorded_data['unmasked_sensor_data']

        kgrid = Mock()
        kgrid.Nx = data_dim[0]
        kgrid.k = data_dim.size

        if data_dim.size in [2, 3]:
            kgrid.Ny = data_dim[1]

        if data_dim.size == 3:
            kgrid.Nz = data_dim[2]

        sensor = Mock()
        sensor.mask = sensor_mask

        unmasked_sensor_data = unmask_sensor_data(kgrid, sensor, sensor_data)

        assert np.allclose(unmasked_sensor_data, expected_unmasked_sensor_data)

    logging.log(logging.INFO, 'unmask_sensor_data(..) works as expected!')
