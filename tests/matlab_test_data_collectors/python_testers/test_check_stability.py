import os
from pathlib import Path

import numpy as np
import pytest
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.checks import check_stability
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_stability_check():
    collected_values_file = os.path.join(Path(__file__).parent, "collectedValues/checkstability.mat")
    reader = TestRecordReader(collected_values_file)

    kgrid_props = reader.expected_value_of("kgrid")
    kspacing = Vector([kgrid_props["dx"], kgrid_props["dy"], kgrid_props["dz"]])
    ksize = Vector([kgrid_props["Nx"], kgrid_props["Ny"], kgrid_props["Nz"]])
    kgrid = kWaveGrid(ksize, kspacing)

    medium_vals = reader.expected_value_of("medium")
    medium = kWaveMedium(
        sound_speed=medium_vals["sound_speed"],
        density=medium_vals["density"],
        alpha_coeff=medium_vals["alpha_coeff"],
        alpha_power=medium_vals["alpha_power"],
    )

    dt = check_stability(kgrid, medium)
    expected_dt = reader.expected_value_of("dt")
    assert np.allclose(dt, expected_dt), f"Stability check failed, expected {expected_dt}, got {dt}."
    pass


if __name__ == "__main__":
    pytest.main([__file__])
