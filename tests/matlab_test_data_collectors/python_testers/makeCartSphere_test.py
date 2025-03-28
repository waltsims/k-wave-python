import logging
import os
from pathlib import Path

import numpy as np

from kwave.data import Vector
from kwave.utils.mapgen import make_cart_sphere
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_makeCartSphere():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/make_cart_sphere.mat")
    reader = TestRecordReader(collected_values_file)

    for i in range(len(reader)):
        logging.log(logging.INFO, i)
        params = reader.expected_value_of("params")
        radius = params[0]
        num_points = params[1]
        center = params[2]
        center = Vector(center.astype(int))
        expected_value = reader.expected_value_of("cart_sphere")

        sphere = make_cart_sphere(radius, num_points, center)

        assert np.allclose(expected_value, sphere)
        reader.increment()

    logging.log(logging.INFO, "makeCartSphere(..) works as expected!")

    if __name__ == "__main__":
        test_makeCartSphere()
