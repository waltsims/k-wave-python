from unittest.mock import Mock

from kwave.utils.interp import cart2grid

from kwave.utils.conversion import scale_time

from scipy.io import loadmat
import numpy as np
import os
import pytest


@pytest.mark.skip(reason="Missing cart2grid collector")
def test_cart2grid():
    collected_values_folder = os.path.join(os.path.curdir, 'collectedValues/cart2grid')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        # 'kgrid', 'cart_data', 'grid_data', ...
        # 'order_index', 'reorder_index'
        kgrid = Mock()
        kgrid.dim = int(recorded_data['kgrid']['dim'])
        kgrid.Nx = int(recorded_data['kgrid']['Nx'])
        kgrid.dx = float(recorded_data['kgrid']['dx'])

        if kgrid.dim in [2, 3]:
            kgrid.Ny = int(recorded_data['kgrid']['Ny'])
            kgrid.dy = float(recorded_data['kgrid']['dy'])

        if kgrid.dim == 3:
            kgrid.Nz = int(recorded_data['kgrid']['Nz'])
            kgrid.dz = float(recorded_data['kgrid']['dz'])

        cart_data = recorded_data['cart_data']
        expected_grid_data = recorded_data['grid_data']
        expected_order_index = recorded_data['order_index']
        expected_reorder_index = recorded_data['reorder_index']
        is_axisymmetric = bool(recorded_data['is_axisymmetric'])

        print(is_axisymmetric)

        if kgrid.dim == 3:
            expected_reorder_index = np.reshape(expected_reorder_index, (-1, 1, 1))

        grid_data, order_index, reorder_index = cart2grid(kgrid, cart_data, axisymmetric=is_axisymmetric)

        assert np.allclose(expected_grid_data, grid_data)
        assert np.allclose(expected_order_index, order_index)
        assert np.allclose(expected_reorder_index, reorder_index)

    print('cart2grid(..) works as expected!')
