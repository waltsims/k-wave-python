import logging
import os
from pathlib import Path

import numpy as np

from kwave.data import Vector
from kwave.utils.mapgen import make_disc
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeDisc():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/makeDisc.mat")
    reader = TestRecordReader(collected_values_file)

    for i in range(len(reader)):
        logging.log(logging.INFO, i)

        Nx, Ny, cx, cy, radius, plot_disc = reader.expected_value_of("params")
        Nx, Ny, cx, cy, radius, plot_disc = int(Nx), int(Ny), int(cx), int(cy), int(radius), bool(plot_disc)
        expected_disc = reader.expected_value_of("disc")

        grid_size = Vector([Nx, Ny])
        center = Vector([cx, cy])
        disc = make_disc(grid_size, center, radius, plot_disc)

        assert np.allclose(expected_disc, disc)

    logging.log(logging.INFO, "make_disc(..) works as expected!")
