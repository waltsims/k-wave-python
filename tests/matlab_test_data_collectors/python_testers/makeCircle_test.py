import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.mapgen import make_circle


def test_makeCircle():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeCircle')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        Nx = float(recorded_data['Nx'])
        Ny = float(recorded_data['Ny'])
        cx = int(recorded_data['cx'])
        cy = int(recorded_data['cy'])
        radius = int(recorded_data['radius'])
        arc_angle = float(recorded_data['arc_angle'])
        expected_circle = recorded_data['circle']

        circle = make_circle(Nx, Ny, cx, cy, radius, arc_angle)

        assert np.allclose(expected_circle, circle)

    print('make_circle(..) works as expected!')
