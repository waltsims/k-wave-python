from kwave.utils.mapgen import make_arc

from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeArc():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/makeArc')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        grid_size, arc_pos, radius, diameter, focus_pos = recorded_data['params'][0]
        grid_size, arc_pos, diameter, focus_pos = grid_size[0], arc_pos[0], int(diameter), focus_pos[0]
        try:
            radius = int(radius)
        except OverflowError:
            radius = float(radius)
        expected_arc = recorded_data['arc']

        arc = make_arc(grid_size, arc_pos, radius, diameter, focus_pos)

        assert np.allclose(expected_arc, arc)

    print('make_arc(..) works as expected!')
