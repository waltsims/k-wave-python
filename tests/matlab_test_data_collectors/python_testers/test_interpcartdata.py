from scipy.io import loadmat
from pathlib import Path
import numpy as np
import os

from kwave.utils.interputils import interpCartData
from kwave.kgrid import kWaveGrid


def test_interpcartdata():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/interpCartData')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        # 'params', 'kgrid', 'sensor_data', 'sensor_mask', 'binary_sensor_mask', 'trbd'
        trbd = recorded_data['trbd']

        dim = int(recorded_data['kgrid']['dim'])
        Nx = int(recorded_data['kgrid']['Nx'])
        dx = float(recorded_data['kgrid']['dx'])

        if dim in [2, 3]:
            Ny = int(recorded_data['kgrid']['Ny'])
            dy = float(recorded_data['kgrid']['dy'])

        if dim == 3:
            Nz = int(recorded_data['kgrid']['Nz'])
            dz = float(recorded_data['kgrid']['dz'])
            kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
        else:
            kgrid = kWaveGrid([Nx, Ny], [dx, dy])

        sensor_data = recorded_data['sensor_data']
        sensor_mask = recorded_data['sensor_mask']
        binary_sensor_mask = recorded_data['binary_sensor_mask']

        trbd_py = interpCartData(kgrid, cart_sensor_data=sensor_data, cart_sensor_mask=sensor_mask, binary_sensor_mask=binary_sensor_mask )

        assert np.allclose(trbd, trbd_py)

    print('cart2grid(..) works as expected!')