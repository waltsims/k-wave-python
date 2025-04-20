import logging
import os
import typing
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import cart2grid
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


class kGridMock(Mock):
    @property
    def __class__(self) -> type:
        return kWaveGrid

    def set_props(self, props):
        self.kprops = props

    def __getattr__(self, name: str) -> typing.Any:
        if name in self.kprops.keys():
            return self.kprops[name]
        return super().__getattr__(name)


def test_cart2grid():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/cart2grid.mat")
    record_reader = TestRecordReader(collected_values_file)

    for i in range(len(record_reader)):
        # 'kgrid', 'cart_data', 'grid_data', ...
        # 'order_index', 'reorder_index'
        kgrid = kGridMock()
        kgrid_props = record_reader.expected_value_of("kgrid")
        kgrid.set_props(kgrid_props)

        cart_data = record_reader.expected_value_of("cart_data")
        if cart_data.ndim == 1:
            cart_data = np.expand_dims(cart_data, axis=0)
        expected_grid_data = record_reader.expected_value_of("grid_data")
        expected_order_index = record_reader.expected_value_of("order_index")
        expected_reorder_index = record_reader.expected_value_of("reorder_index")
        is_axisymmetric = bool(record_reader.expected_value_of("is_axisymmetric"))

        logging.log(logging.INFO, is_axisymmetric)

        if kgrid.dim == 3:
            expected_reorder_index = np.reshape(expected_reorder_index, (-1, 1, 1))

        grid_data, order_index, reorder_index = cart2grid(kgrid, cart_data, axisymmetric=is_axisymmetric)

        assert len(expected_order_index) == len(order_index), f"Failed on example {i}"
        assert np.allclose(expected_order_index, order_index.squeeze()), f"Failed on example {i}"
        assert np.allclose(expected_reorder_index, reorder_index.squeeze()), f"Failed on example {i}"
        assert np.allclose(expected_grid_data, grid_data.squeeze()), f"Failed on example {i}"

    logging.log(logging.INFO, "cart2grid(..) works as expected!")


if __name__ == "__main__":
    test_cart2grid()
