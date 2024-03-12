from kwave.data import Vector
from kwave.utils.mapgen import make_ball

import logging
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeBall():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/makeBall")
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath)

        Nx, Ny, Nz, cx, cy, cz, radius, plot_ball, binary = recorded_data["params"][0]
        grid_size = Vector([int(Nx), int(Ny), int(Nz)])
        ball_center = Vector([int(cx), int(cy), int(cz)])
        radius, plot_ball, binary = int(radius), bool(plot_ball), bool(binary)

        expected_ball = recorded_data["ball"]

        ball = make_ball(grid_size, ball_center, radius, plot_ball, binary)

        assert np.allclose(expected_ball, ball)

    logging.log(logging.INFO, "make_ball(..) works as expected!")
