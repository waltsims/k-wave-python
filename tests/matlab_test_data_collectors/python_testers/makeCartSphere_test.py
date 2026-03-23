import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.data import Vector
from kwave.utils.mapgen import make_cart_sphere


def test_makeCartSphere():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/makeCartSphere")
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f"{i:06d}.mat")
        recorded_data = loadmat(filepath)
        params = recorded_data["params"][0]

        radius, num_points, center = params
        radius = float(radius[0][0])
        num_points = int(num_points[0][0])
        center = Vector(center[0].astype(int))

        expected_value = recorded_data["sphere"]
        sphere = make_cart_sphere(radius, num_points, center)

        assert np.allclose(expected_value, sphere)

    logging.log(logging.INFO, "makeCartSphere(..) works as expected!")


if __name__ == "__main__":
    test_makeCartSphere()
