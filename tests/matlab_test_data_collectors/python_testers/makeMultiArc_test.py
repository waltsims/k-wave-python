from kwave.utils.maputils import makeMultiArc

from scipy.io import loadmat
import numpy as np
import os
import pytest


# @pytest.mark.skip(reason="no way of currently testing this")
def test_makeMultiArc():
    collected_values_folder = '/data/code/Work/black_box_testing/collectedValues/makeMultiArc'
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        print(i)
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        grid_size, arc_pos, radius, diameter, focus_pos = recorded_data['params'][0]
        grid_size, radius, diameter, focus_pos = grid_size[0], radius[0], diameter[0], focus_pos
        expected_multi_arc = recorded_data['multi_arc']

        multi_arc, _ = makeMultiArc(grid_size, arc_pos, radius, diameter, focus_pos)

        assert np.allclose(expected_multi_arc, multi_arc)

    print('makeMultiArc(..) works as expected!')
