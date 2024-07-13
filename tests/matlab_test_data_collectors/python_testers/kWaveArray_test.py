import os
from pathlib import Path

import numpy as np
import pytest
from kwave.data import Vector
from kwave.kgrid import kWaveGrid

from kwave.utils.kwave_array import (
    ArcElement,
    BowlElement,
    CustomElement,
    DiscElement,
    LineElement,
    RectElement,
    kWaveArray,
    AnnulusElement,
)
from tests.matlab_test_data_collectors.python_testers.utils.check_equality import check_kwave_array_equality
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader

type_to_class = {
    "annulus": "AnnulusElement",
    "bowl": "BowlElement",
    "custom": "CustomElement",
    "disc": "DiscElement",
    "line": "LineElement",
    "rect": "RectElement",
    "arc": "ArcElement",
}


def compare_elements(expected_element, python_element):
    # itterate through the properties of the element and compare them
    # be sure to include properties of annulus element
    for key, expected_value in expected_element.items():
        if key == "type":
            assert type_to_class[expected_value] == python_element.__class__.__name__
            continue
        actual_value = getattr(python_element, key)
        if key == "integration_points":
            actual_value = actual_value.tolist()
            expected_value = expected_value.tolist()
        if isinstance(expected_value, np.ndarray) and isinstance(actual_value, np.ndarray):
            if expected_value.dtype == object or actual_value.dtype == object:
                pass
        elif isinstance(expected_value, str) and isinstance(actual_value, str):
            actual_value == expected_value
        else:
            assert np.all(np.isclose(actual_value, expected_value))


def test_kwave_array():
    test_record_path = os.path.join(Path(__file__).parent, "collectedValues/kWaveArray.mat")
    reader = TestRecordReader(test_record_path)

    kwave_array = kWaveArray()

    # TODO: test elements individually
    # create an annular element in kWaveArray and compare only the element created and it's properties to elements create in python

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
    # compare with annular element created in python
    expected_annulus_element = reader.expected_value_of("kwave_array")["elements"][-1]
    annulus_element = AnnulusElement([0, 0, 0], 5, 0.001, 0.03, [1, 5, -3])
    compare_elements(expected_annulus_element, annulus_element)

    reader.increment()

    kwave_array.add_bowl_element([0, 0, 0], 5, 4.3, [1, 5, -3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    expected_bowl_element = reader.expected_value_of("kwave_array")["elements"][-1]
    bowl_element = BowlElement([0, 0, 0], 5, 4.3, [1, 5, -3])
    compare_elements(expected_bowl_element, bowl_element)

    reader.increment()
    integration_points = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    kwave_array.add_custom_element(
        integration_points=integration_points,
        measure=9,
        element_dim=2,
        label="custom_3d",
    )
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))

    expected_custom_element = reader.expected_value_of("kwave_array")["elements"][-1]
    custom_element = CustomElement(integration_points, 9, 2, "custom_3d")
    compare_elements(expected_custom_element, custom_element)

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
    expected_rect_element = reader.expected_value_of("kwave_array")["elements"][-1]
    # explicit fix due to dimension logic in kWaveArray
    expected_rect_element["dim"] = 3
    rect_element = RectElement([12, -8, 0.3], 3, 4, [2, 4, 5])
    compare_elements(expected_rect_element, rect_element)
    reader.increment()

    kwave_array.add_disc_element([0, 0.3, 12], 5, [1, 5, 8])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))

    expected_disc_element = reader.expected_value_of("kwave_array")["elements"][-1]
    expected_disc_element["dim"] = 3
    disc_element = DiscElement([0, 0.3, 12], 5, [1, 5, 8])
    compare_elements(expected_disc_element, disc_element)
    reader.increment()

    # test list input
    kwave_array = kWaveArray()
    kwave_array.add_arc_element([0, 0.3], 5, 4.3, [1, 5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of("kwave_array"))
    arc_element = ArcElement([0, 0.3], 5, 4.3, [1, 5])
    expected_arc_element = reader.expected_value_of("kwave_array")["elements"]
    compare_elements(expected_arc_element, arc_element)
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
    disc_element = DiscElement([0, 0.3], 5)
    expected_disc_element = reader.expected_value_of("kwave_array")["elements"][-1]
    compare_elements(expected_disc_element, disc_element)
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
    line_element = LineElement([0, 3], [5, 2])
    expected_line_element = reader.expected_value_of("kwave_array")["elements"]
    compare_elements(expected_line_element, line_element)
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

    # 2D karray tests

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0, 3], [5, 2])
    kgrid = kWaveGrid([100, 200], [0.1, 0.3])
    assert kwave_array.dim == 2
    assert np.allclose(kwave_array.get_array_grid_weights(kgrid).shape, reader.expected_value_of("grid_weights").shape)
    assert np.allclose(kwave_array.get_array_grid_weights(kgrid), np.squeeze(reader.expected_value_of("grid_weights")))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
