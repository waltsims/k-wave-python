from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader
from tests.matlab_test_data_collectors.python_testers.utils.check_equality import check_kwave_array_equality
from tests.matlab_test_data_collectors.python_testers.utils.check_equality import check_kgrid_equality
import numpy as np
import os
from pathlib import Path

import kwave.data
from kwave.kgrid import kWaveGrid
from kwave.utils.kwave_array import kWaveArray


def test_linear_array_transducer():
    test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/linear_array_transducer.mat')
    reader = TestRecordReader(test_record_path)

    c0 = 1500
    source_f0 = 1e6
    source_focus = 20e-3
    element_num = 15
    element_width = 1e-3
    element_length = 10e-3
    element_pitch = 2e-3
    translation = kwave.data.Vector([5e-3, 0, 8e-3])
    rotation = kwave.data.Vector([0, 20, 0])
    grid_size_x = 40e-3
    grid_size_y = 20e-3
    grid_size_z = 40e-3
    ppw = 3
    t_end = 35e-6
    cfl = 0.5
    bli_tolerance = 0.05
    upsampling_rate = 10

    # GRID
    dx = c0 / (ppw * source_f0)
    Nx = round(grid_size_x / dx)
    Ny = round(grid_size_y / dx)
    Nz = round(grid_size_z / dx)
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    kgrid.makeTime(c0, cfl, t_end)

    check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
    # SOURCE
    if element_num % 2 != 0:
        centering_offset = np.ceil(element_num / 2)
    else:
        centering_offset = (element_num + 1) / 2

    positional_basis = np.arange(1, element_num + 1) - centering_offset
    
    time_delays = -(np.sqrt((positional_basis * element_pitch) ** 2 + source_focus ** 2) - source_focus) / c0
    time_delays = time_delays - min(time_delays)

    karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate)

    for ind in range(1, element_num + 1):
        x_pos = 0 - (element_num * element_pitch / 2 - element_pitch / 2) + (ind - 1) * element_pitch
        karray.add_rect_element([x_pos, 0, kgrid.z_vec.flat[0]], element_width, element_length, rotation)

    karray.set_array_position(translation, rotation)

    check_kwave_array_equality(karray, reader.expected_value_of('karray'))