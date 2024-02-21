from kwave.data import Vector
from kwave.utils.mapgen import make_multi_arc

import logging
from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path


def test_makeMultiArc():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/makeMultiArc")

    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        logging.log(logging.INFO, i)
        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath, simplify_cells=True)

        grid_size, arc_pos, radius, diameter, focus_pos = recorded_data["params"]
        expected_multi_arc = recorded_data["multi_arc"]

        grid_size = Vector(grid_size)
        multi_arc, _ = make_multi_arc(grid_size, arc_pos, radius, diameter, focus_pos)

        assert np.allclose(expected_multi_arc, multi_arc)

    logging.log(logging.INFO, "make_multi_arc(..) works as expected!")
