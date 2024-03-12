import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.matrix import revolve2d


def test_revolve2D():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/revolve2D")

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath)

        params = recorded_data["params"][0]
        mat2D = params[0]

        expected_mat3D = recorded_data["mat3D"]

        mat3D = revolve2d(mat2D)

        assert np.allclose(expected_mat3D, mat3D)

    logging.log(logging.INFO, "revolve2d(..) works as expected!")
