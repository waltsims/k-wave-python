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

    for _ in range(len(reader)):
        # 'params', 'kgrid', 'sensor_data', 'sensor_mask', 'binary_sensor_mask', 'trbd'
        trbd = reader.expected_value_of("trbd")
        kgrid_props = reader.expected_value_of("kgrid")
        sensor_data = reader.expected_value_of("sensor_data")
        sensor_mask = reader.expected_value_of("sensor_mask")
        binary_sensor_mask = reader.expected_value_of("binary_sensor_mask")
        interp_method = reader.expected_value_of("interp_method")

        kgrid = kGridMock()
        kgrid.set_props(kgrid_props)

        print(kgrid.Nx, kgrid.Ny, kgrid.Nz, np.shape(sensor_data), np.shape(sensor_mask), np.shape(binary_sensor_mask), interp_method)

        trbd_py = interp_cart_data(
            kgrid,
            cart_sensor_data=sensor_data,
            cart_sensor_mask=sensor_mask,
            binary_sensor_mask=binary_sensor_mask.astype(bool),
            interp=interp_method,
        )

        assert np.allclose(trbd, trbd_py), "interpolated values not correct"
        reader.increment()

    logging.log(logging.INFO, "cart2grid(..) works as expected!")


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
