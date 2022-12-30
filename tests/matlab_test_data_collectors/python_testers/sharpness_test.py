import os
from pathlib import Path

import numpy as np

from kwave.utils.filters import sharpness
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_tennenbaum2d():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/tenenbaum.mat'))
    reader = TestRecordReader(test_record_path)
    img = reader.expected_value_of('img2')

    out = sharpness(img, 'Tenenbaum')

    out_prime = reader.expected_value_of('out2')
    assert np.allclose(out, out_prime), "Tenenbaum sharpness did not match expected sharpness in 2D"


def test_tennenbaum3d():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/tenenbaum.mat'))
    reader = TestRecordReader(test_record_path)

    img = reader.expected_value_of('img3')
    out = sharpness(img, 'Tenenbaum')
    out_prime = reader.expected_value_of('out3')
    assert np.allclose(out, out_prime), "Tenenbaum sharpness did not match expected sharpness in 3D"


def test_brenner2d():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/brenner.mat'))
    reader = TestRecordReader(test_record_path)
    img = reader.expected_value_of('img2')
    out = sharpness(img, 'Brenner')

    out_prime = reader.expected_value_of('out2')
    assert np.allclose(out, out_prime), "Brenner sharpness did not match expected sharpness in 2D"


def test_brenner3d():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/brenner.mat'))
    reader = TestRecordReader(test_record_path)

    img = reader.expected_value_of('img3')
    out = sharpness(img, 'Brenner')
    out_prime = reader.expected_value_of('out3')
    assert np.allclose(out, out_prime), "Brenner sharpness did not match expected sharpness in 3D"


def test_normvar():
    test_record_path = os.path.join(Path(__file__).parent, Path(
        'collectedValues/normvar.mat'))
    reader = TestRecordReader(test_record_path)
    img = reader.expected_value_of('img2')

    out = sharpness(img, 'NormVariance')

    out_prime = reader.expected_value_of('out2')
    assert np.allclose(out, out_prime), "NormVariance sharpness did not match expected sharpness in 2D"

    img = reader.expected_value_of('img3')
    out = sharpness(img, 'NormVariance')
    out_prime = reader.expected_value_of('out3')
    assert np.allclose(out, out_prime), "NormVariance sharpness did not match expected sharpness in 3D"
