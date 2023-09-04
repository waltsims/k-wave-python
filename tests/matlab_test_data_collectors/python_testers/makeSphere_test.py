from kwave.data import Vector
from kwave.utils.mapgen import make_sphere

from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeSphere():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeSphere')

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        Nx, Ny, Nz, radius, plot_sphere, binary = recorded_data['params'][0]
        Nx, Ny, Nz, radius, plot_sphere, binary = int(Nx), int(Ny), int(Nz), int(radius), bool(plot_sphere), bool(binary)
        expected_sphere = recorded_data['sphere']

        grid_size = Vector([Nx, Ny, Nz])
        sphere = make_sphere(grid_size, radius, plot_sphere, binary)

        assert np.allclose(expected_sphere, sphere)

    print('make_sphere(..) works as expected!')
