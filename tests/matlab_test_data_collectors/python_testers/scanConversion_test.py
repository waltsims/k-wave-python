import logging
import os
from pathlib import Path

import numpy as np

from kwave.reconstruction.beamform import scan_conversion
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_scanConversion():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/scanConversion.mat")
    reader = TestRecordReader(collected_values_file)

    for _ in range(len(reader)):
        scan_lines = reader.expected_value_of("scan_lines")
        steering_angles = reader.expected_value_of("steering_angles")
        image_size = reader.expected_value_of("image_size")
        c0 = reader.expected_value_of("c0")
        dt = reader.expected_value_of("dt")
        resolution = reader.expected_value_of("resolution")
        expected_b_mode = reader.expected_value_of("b_mode")

        calculated_b_mode = scan_conversion(scan_lines, steering_angles, image_size, c0, dt, resolution)

        assert np.allclose(expected_b_mode, calculated_b_mode, equal_nan=True)

        reader.increment()

    logging.log(logging.INFO, "scan_conversion(..) works as expected!")
