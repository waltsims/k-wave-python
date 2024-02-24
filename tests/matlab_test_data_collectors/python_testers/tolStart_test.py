import os
from pathlib import Path
import typing
from unittest.mock import Mock

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import tol_star

from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


class kGridMock(Mock):

    @property
    def __class__(self) -> type:
        return kWaveGrid
    
    def set_props(self, props):
        self.kprops = props
    
    def __getattr__(self, name: str) -> typing.Any:
        if name in vars(self.kprops)['_fieldnames']:
            return self.kprops.__getattribute__(name)
        return super().__getattr__(name)


def test_tol_star():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, 'collectedValues/tolStar.mat'))

    for i in range(len(reader)):
        tolerance, kgrid_props, point = reader.expected_value_of("params")

        kgrid = kGridMock()
        kgrid.set_props(kgrid_props)

        if not isinstance(point, np.ndarray):
            point = np.array([float(point)])
        else:
            point = point.astype(float)

        lin_ind, is_, js, ks = tol_star(tolerance, kgrid, point, debug=False)

        expected_lin_ind = reader.expected_value_of("lin_ind")
        expected_is = reader.expected_value_of("is")
        expected_js = reader.expected_value_of("js")
        expected_ks = reader.expected_value_of("ks")

        assert np.allclose(lin_ind, expected_lin_ind), "tone_burst did not match expected tone_burst"
        assert np.allclose(is_, expected_is - 1), "tone_burst did not match expected tone_burst"
        assert np.allclose(js, expected_js - 1), "tone_burst did not match expected tone_burst"
        assert np.allclose(ks, expected_ks - 1), "tone_burst did not match expected tone_burst"
        reader.increment()
