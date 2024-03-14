import os
from pathlib import Path

import numpy as np
import pytest
from kwave.data import Vector
from kwave.kgrid import kWaveGrid

from kwave.utils.kwave_array import kWaveArray
from tests.matlab_test_data_collectors.python_testers.utils.check_equality import check_kwave_array_equality
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_kwave_array():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/kWaveArray.mat")
    reader = TestRecordReader(test_record_path)

    kwave_array = kWaveArray()

    # Useful for checking if the defaults are set correctly
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array = kWaveArray(axisymmetric=True, bli_tolerance=0.5, bli_type="sinc", single_precision=True, upsampling_rate=20)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array = kWaveArray(axisymmetric=False, bli_tolerance=0.5, bli_type="exact", single_precision=False, upsampling_rate=1)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_annular_array([3, 5, 10], 5, [[1.2, 0.5]], [12, 21, 3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_annular_array([3, 5, 10], 5, [[1.2, 0.5], [5.3, 1.0]], [12, 21, 3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_annular_element([0, 0, 0], 5, [0.001, 0.03], [1, 5, -3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_bowl_element([0, 0, 0], 5, 4.3, [1, 5, -3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_custom_element(
        integration_points=np.array(
            [[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32
        ),
        measure=9,
        element_dim=2,
        label="custom_3d",
    )
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))

    with pytest.raises(ValueError):
        kwave_array.add_custom_element(
            integration_points=np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.float32),
            measure=9,
            element_dim=2,
            label="custom_3d",
        )

    reader.increment()

    kwave_array.add_rect_element([12, -8, 0.3], 3, 4, [2, 4, 5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_disc_element([0, 0.3, 12], 5, [1, 5, 8])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    # test list input
    kwave_array = kWaveArray()
    kwave_array.add_arc_element([0, 0.3], 5, 4.3, [1, 5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    # test tuple input
    kwave_array = kWaveArray()
    kwave_array.add_arc_element((0, 0.3), 5, 4.3, (1, 5))
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    # test Vector input
    kwave_array = kWaveArray()
    kwave_array.add_arc_element(Vector([0, 0.3]), 5, 4.3, Vector([1, 5]))
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_disc_element([0, 0.3], 5)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.add_custom_element(
        np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.float32), 9, 1, label="custom_2d"
    )
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    # Useful for testing addRectElement in 2D
    kwave_array.add_rect_element([12, -8], 3, 4, 2)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array.remove_element(2)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()
    kwave_array.remove_element(0)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()
    kwave_array.remove_element(0)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0], [5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0, 3], [5, 2])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0, 3, -3], [5, 2, -9])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kgrid = kWaveGrid([100, 200, 150], [0.1, 0.3, 0.4])
    grid_weights = kwave_array.get_element_grid_weights(kgrid, 0)
    assert np.allclose(grid_weights, reader.expected_value_of("grid_weights"))

    mask = kwave_array.get_element_binary_mask(kgrid, 0)
    assert np.allclose(mask, reader.expected_value_of("mask"))
    reader.increment()

    grid_weights = kwave_array.get_array_grid_weights(kgrid)
    assert np.allclose(grid_weights, reader.expected_value_of("grid_weights"))

    mask = kwave_array.get_array_binary_mask(kgrid)
    assert np.allclose(mask, reader.expected_value_of("mask"))
    reader.increment()

    source_signal = reader.expected_value_of("source_signal")
    distributed_source_signal = kwave_array.get_distributed_source_signal(kgrid, source_signal)
    assert np.allclose(distributed_source_signal, reader.expected_value_of("distributed_source_signal"))
    reader.increment()

    kgrid = kWaveGrid([10, 100, 100], [0.1, 0.1, 0.1])
    sensor_data = reader.expected_value_of("sensor_data")
    combined_sensor_data = kwave_array.combine_sensor_data(kgrid, sensor_data)
    assert np.allclose(combined_sensor_data, reader.expected_value_of("combined_sensor_data"))
    reader.increment()

    translation = reader.expected_value_of("translation")
    rotation = reader.expected_value_of("rotation")
    kwave_array.set_array_position(translation, rotation)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    affine_transform = reader.expected_value_of("affine_transform")
    kwave_array.set_affine_transform(affine_transform)
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_annular_array([3.1, 5.2, 10.9], 5, [[1.2, 0.5]], [12, 21, 3])
    element_pos = kwave_array.get_element_positions()
    assert np.allclose(element_pos.squeeze(axis=-1), reader.expected_value_of("element_pos"))
    reader.increment()
