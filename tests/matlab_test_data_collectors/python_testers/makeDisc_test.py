from kwave.utils.maputils import makeDisc

from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeDisc():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeDisc')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        Nx, Ny, cx, cy, radius, plot_disc = recorded_data['params'][0]
        Nx, Ny, cx, cy, radius, plot_disc = int(Nx), int(Ny), int(cx), int(cy), int(radius), bool(plot_disc)
        expected_disc = recorded_data['disc']

        disc = makeDisc(Nx, Ny, cx, cy, radius, plot_disc)

        assert np.allclose(expected_disc, disc)

    print('makeDisc(..) works as expected!')
