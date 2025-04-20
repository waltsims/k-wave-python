import os
from pathlib import Path

import numpy as np

from kwave.utils.angular_spectrum_cw import angular_spectrum_cw
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_angular_spectrum_cw():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/angularSpectrumCW.mat")
    reader = TestRecordReader(test_record_path)

    input_plane = reader.expected_value_of("input_plane")
    dx = reader.expected_value_of("dx")
    z_pos = reader.expected_value_of("z_pos")
    f0 = reader.expected_value_of("f0")
    c0 = reader.expected_value_of("c0")
    grid_expansion = reader.expected_value_of("grid_expansion")

    expected_pressure = reader.expected_value_of("pressure")

    pressure = angular_spectrum_cw(input_plane, dx, z_pos, f0, c0, grid_expansion=grid_expansion)
    assert np.allclose(pressure.squeeze(), expected_pressure)
