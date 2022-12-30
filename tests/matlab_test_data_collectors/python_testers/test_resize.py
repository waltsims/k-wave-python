import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.matrix import resize


def test_resize():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/resize')

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)
        # 'params', 'volume', 'resized_volume', 'new_size'
        mat = recorded_data['volume']
        expected_mat = recorded_data['resized_volume'].squeeze()
        new_size = recorded_data['new_size'][0]
        method = recorded_data['method'][0]  # TODO: does not work for spline cases

        resized_mat = resize(mat, new_size, interp_mode=method)

        assert np.allclose(expected_mat, resized_mat), f"Results do not match for {i + 1} dimensional case."

    print('revolve2d(..) works as expected!')
