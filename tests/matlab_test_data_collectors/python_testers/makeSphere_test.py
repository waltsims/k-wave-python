import logging
import os
from pathlib import Path

import numpy as np

from kwave.data import Vector
from kwave.utils.mapgen import make_sphere
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeSphere():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/makeSphere.mat")
    reader = TestRecordReader(collected_values_file)

    for i in range(len(reader)):
        logging.log(logging.INFO, i)

        Nx, Ny, Nz, radius, plot_sphere, binary = reader.expected_value_of("params")
        Nx, Ny, Nz, radius, plot_sphere, binary = int(Nx), int(Ny), int(Nz), int(radius), bool(plot_sphere), bool(binary)
        expected_sphere = reader.expected_value_of("sphere")

        grid_size = Vector([Nx, Ny, Nz])
        sphere = make_sphere(grid_size, radius, plot_sphere, binary)

        assert np.allclose(expected_sphere, sphere)
        reader.increment()

    logging.log(logging.INFO, "make_sphere(..) works as expected!")
