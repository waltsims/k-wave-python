from kwave.utils.mapgen import make_ball

from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeBall():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeBall')
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

        ball = make_ball(Nx, Ny, Nz, cx, cy, cz, radius, plot_ball, binary)

        assert np.allclose(expected_ball, ball)

    print('make_ball(..) works as expected!')
