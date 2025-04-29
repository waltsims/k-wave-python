import logging
import os
import typing
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.utils.interp import interp_cart_data
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

    for i in range(len(reader)):
        # 'params', 'kgrid', 'sensor_data', 'sensor_mask', 'binary_sensor_mask', 'trbd'
        trbd = reader.expected_value_of("trbd")
        kgrid_props = reader.expected_value_of("kgrid")
        sensor_data = reader.expected_value_of("sensor_data")
        sensor_mask = reader.expected_value_of("sensor_mask")
        binary_sensor_mask = reader.expected_value_of("binary_sensor_mask")
        interp_method = reader.expected_value_of("interp_method")

        kgrid = kGridMock()
        kgrid.set_props(kgrid_props)

        trbd_py = interp_cart_data(
            kgrid,
            cart_sensor_data=sensor_data,
            cart_sensor_mask=sensor_mask,
            binary_sensor_mask=binary_sensor_mask.astype(bool),
            interp=interp_method,
        )

        sorted_trbd = np.sort(trbd, axis=1)
        sorted_trbd_py = np.sort(trbd_py, axis=1)
        print(i, sorted_trbd[0, :])
        print(i, sorted_trbd_py[0, :])

        assert np.allclose(sorted_trbd, sorted_trbd_py), f"{i}. interpolated values not correct with method: {interp_method}"
        reader.increment()

    logging.log(logging.INFO, "interp_cart_data(..) works as expected!")


def test_unknown_interp_method():
    with pytest.raises(ValueError):
        reader = TestRecordReader(os.path.join(Path(__file__).parent, "collectedValues/interpCartData.mat"))
        kprops = reader.expected_value_of("kgrid")
        kgrid = kGridMock()
        kgrid.set_props(kprops)
        interp_cart_data(
            kgrid,
            cart_sensor_data=reader.expected_value_of("sensor_data"),
            cart_sensor_mask=reader.expected_value_of("sensor_mask"),
            binary_sensor_mask=reader.expected_value_of("binary_sensor_mask").astype(bool),
            interp="unknown",
        )


if __name__ == "__main__":
    pytest.main([__file__])
