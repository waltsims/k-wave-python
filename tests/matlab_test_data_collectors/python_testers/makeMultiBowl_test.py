import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.data import Vector
from kwave.utils.mapgen import make_multi_bowl


def test_makeMultiBowl():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/makeMultiBowl")
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath, simplify_cells=True)

        params = recorded_data["params"]
        grid_size, bowl_pos, radius, diameter, focus_pos = params[:5]
        grid_size = Vector(grid_size)

        binary = bool(params[6])
        remove_overlap = bool(params[8])
        expected_multi_bowl = recorded_data["multiBowl"]

        multi_bowl, _ = make_multi_bowl(grid_size, bowl_pos, radius, diameter, focus_pos, binary=binary, remove_overlap=remove_overlap)

        assert np.allclose(expected_multi_bowl, multi_bowl)

    logging.log(logging.INFO, "make_multi_bowl(..) works as expected!")
