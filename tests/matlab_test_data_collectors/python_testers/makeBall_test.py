from kwave.utils.maputils import makeBall

from scipy.io import loadmat
import numpy as np
import os
import pytest


@pytest.mark.skip(reason="no way of currently testing this")
def test_makeBall():
    collected_values_folder = '/data/code/Work/black_box_testing/collectedValues_makeBall'
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        Nx, Ny, Nz, cx, cy, cz, radius, plot_ball, binary = recorded_data['params'][0]
        Nx, Ny, Nz = int(Nx), int(Ny), int(Nz)
        cx, cy, cz = int(cx), int(cy), int(cz)
        radius, plot_ball, binary = int(radius), bool(plot_ball), bool(binary)

        expected_ball = recorded_data['ball']

        ball = makeBall(Nx, Ny, Nz, cx, cy, cz, radius, plot_ball, binary)

        assert np.allclose(expected_ball, ball)

    print('makeBall(..) works as expected!')
