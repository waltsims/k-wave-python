import logging
import os
import typing
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import grid2cart
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


def test_grid2cart():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/grid2cart.mat")
    record_reader = TestRecordReader(collected_values_file)

    for i in range(len(record_reader)):
        # 'kgrid', 'cart_data', 'grid_data', 'order_index'
        kgrid = kGridMock()
        kgrid_props = record_reader.expected_value_of("kgrid")
        kgrid.set_props(kgrid_props)

        grid_data = record_reader.expected_value_of("grid_data")

        expected_cart_data = record_reader.expected_value_of("cart_data")
        expected_order_index = record_reader.expected_value_of("order_index")

        cart_data, order_index = grid2cart(kgrid, grid_data)

        cart_data = cart_data.squeeze() 
        order_index = order_index.squeeze() + 1

        assert np.allclose(expected_cart_data, cart_data), f"Failed on example {i}"
        assert len(expected_order_index) == len(order_index), f"Failed on example {i}"
        assert np.allclose(expected_order_index, order_index), f"Failed on example {i}"
        
    logging.log(logging.INFO, "grid2cart(..) works as expected!")


def test_grid2cart_grid_dimensions():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/grid2cart.mat")
    record_reader = TestRecordReader(collected_values_file)
    grid_data = record_reader.expected_value_of("grid_data")
    bad_dims = [0, 4]
    for bad_dim in bad_dims:
        kgrid_dummy_props = {'dim': bad_dim}
        kgrid_dummy = kGridMock()
        kgrid_dummy.set_props(kgrid_dummy_props)
        with pytest.raises(ValueError, match=("kGrid with unsupported size passed.")):
            grid2cart(kgrid_dummy, grid_data)



if __name__ == "__main__":
    test_grid2cart()
    test_grid2cart_grid_dimensions()