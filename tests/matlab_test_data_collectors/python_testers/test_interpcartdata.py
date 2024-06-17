import logging
import typing
from unittest.mock import Mock
from pathlib import Path
import numpy as np
import os

from kwave.utils.interp import interp_cart_data
from kwave.kgrid import kWaveGrid
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


def test_interpcartdata():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/interpCartData.mat"))

    for _ in range(len(reader)):
        # 'params', 'kgrid', 'sensor_data', 'sensor_mask', 'binary_sensor_mask', 'trbd'
        trbd = reader.expected_value_of("trbd")
        kgrid_props = reader.expected_value_of("kgrid")
        sensor_data = reader.expected_value_of("sensor_data")
        sensor_mask = reader.expected_value_of("sensor_mask")
        binary_sensor_mask = reader.expected_value_of("binary_sensor_mask")

        kgrid = kGridMock()
        kgrid.set_props(kgrid_props)

        trbd_py = interp_cart_data(kgrid, cart_sensor_data=sensor_data, cart_sensor_mask=sensor_mask, binary_sensor_mask=binary_sensor_mask)

        assert np.allclose(trbd, trbd_py)
        reader.increment()

    logging.log(logging.INFO, "cart2grid(..) works as expected!")


if __name__ == "__main__":
    test_interpcartdata()
