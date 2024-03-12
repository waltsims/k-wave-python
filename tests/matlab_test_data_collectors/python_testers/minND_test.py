import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.matrix import min_nd


def test_minND():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/minND")
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath)

        matrix = recorded_data["matrix"]
        expected_min_val = float(recorded_data["min_val"])
        expected_ind = np.squeeze(recorded_data["ind"])

        min_val, ind = min_nd(matrix)

        assert np.allclose(expected_min_val, min_val, equal_nan=True)
        assert np.allclose(expected_ind, ind)

    logging.log(logging.INFO, "min_nd(..) works as expected!")
