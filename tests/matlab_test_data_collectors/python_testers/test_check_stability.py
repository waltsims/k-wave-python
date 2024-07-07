import os
from pathlib import Path
from unittest.mock import Mock

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
    reader.increment()
    medium_vals = reader.expected_value_of("medium")
    medium = kWaveMedium(
        sound_speed=medium_vals["sound_speed"],
        density=medium_vals["density"],
        alpha_coeff=medium_vals["alpha_coeff"],
        alpha_power=medium_vals["alpha_power"],
        sound_speed_ref=medium_vals["sound_speed_ref"],
        alpha_mode=medium_vals["alpha_mode"],
    )
    dt = check_stability(kgrid, medium)
    expected_dt = reader.expected_value_of("dt")
    assert np.allclose(dt, expected_dt), f"Stability check failed, expected {expected_dt}, got {dt}."
    reader.increment()
    medium_vals = reader.expected_value_of("medium")
    medium = kWaveMedium(
        sound_speed=medium_vals["sound_speed"],
        density=medium_vals["density"],
        alpha_coeff=medium_vals["alpha_coeff"],
        alpha_power=medium_vals["alpha_power"],
        sound_speed_ref=medium_vals["sound_speed_ref"],
        alpha_mode=medium_vals["alpha_mode"],
    )
    dt = check_stability(kgrid, medium)
    expected_dt = reader.expected_value_of("dt")
    assert np.allclose(dt, expected_dt), f"Stability check failed, expected {expected_dt}, got {dt}."

    pass


def test_stability_check_fail():
    kgrid = Mock()
    kgrid.k.max().return_value = 0.5
    medium = kWaveMedium(
        sound_speed=1500,
        density=1600,
        alpha_coeff=0.5,
        alpha_power=1.5,
        sound_speed_ref="blah",
    )

    with pytest.raises(NotImplementedError):
        check_stability(kgrid, medium)

    pass


if __name__ == "__main__":
    pytest.main([__file__])
