import os
from pathlib import Path

import numpy as np

from kwave.utils.angular_spectrum import angular_spectrum
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_angular_spectrum():
    test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/angularSpectrum.mat')
    reader = TestRecordReader(test_record_path)

    input_plane = reader.expected_value_of('input_plane')
    dx = reader.expected_value_of('dx')
    dt = reader.expected_value_of('dt')
    z_pos = reader.expected_value_of('z_pos')
    c0 = reader.expected_value_of('c0')
    grid_expansion = reader.expected_value_of('grid_expansion')

    expected_pressure_max = reader.expected_value_of('pressure_max')
    expected_pressure_time = reader.expected_value_of('pressure_time')

    pressure_max, pressure_time = angular_spectrum(input_plane, dx, dt, z_pos, c0,
                                                   grid_expansion=grid_expansion, record_time_series=True)

    assert np.allclose(pressure_time.squeeze(), expected_pressure_time)
    assert np.allclose(pressure_max.squeeze(), expected_pressure_max)
