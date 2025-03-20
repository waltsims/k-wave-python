import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from kwave.utils.colormap import get_color_map
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_get_color_map():
    matplotlib.use("Agg")

    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/getColorMap.mat")
    reader = TestRecordReader(collected_values_file)

    for i in range(len(reader)):
        logging.log(logging.INFO, i)
        # Read recorded data

        num_colors = reader.expected_value_of("num_colors")
        expected_color_map = reader.expected_value_of("color_map")

        # Execute implementation
        color_map = get_color_map(num_colors)

        # Check correctness
        assert np.allclose(color_map.colors, expected_color_map)

        plt.imshow(np.random.rand(5, 5), cmap=color_map)
        plt.show()

    logging.log(logging.INFO, "get_color_map(..) works as expected!")
