import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.mapgen import make_line


def test_makeLine():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeLine')

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)

        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        params = recorded_data['params'][0]
        if len(params) == 4:
            Nx, Ny, startpoint, endpoint = params
            Nx, Ny, startpoint, endpoint = int(Nx), int(Ny), startpoint[0], endpoint[0]
            line = make_line(Nx, Ny, startpoint, endpoint)
        else:
            Nx, Ny, startpoint, angle, length = params
            Nx, Ny, startpoint, angle, length = int(Nx), int(Ny), startpoint[0], float(angle), int(length)
            line = make_line(Nx, Ny, startpoint, endpoint=None, angle=angle, length=length)

        expected_line = recorded_data['line']

        if i == 3:
            print('here')

        assert np.allclose(expected_line, line)

    print('make_line(..) works as expected!')
