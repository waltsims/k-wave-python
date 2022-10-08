
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path

from kwave.utils import revolve2D


def test_revolve2D():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/revolve2D')

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        params = recorded_data['params'][0]
        mat2D = params[0]

        expected_mat3D = recorded_data['mat3D']

        mat3D = revolve2D(mat2D)

        assert np.allclose(expected_mat3D, mat3D)

    print('revolve2D(..) works as expected!')
