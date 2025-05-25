from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball


def make_simulation_parameters(directory: Path):
    # create the computational grid
    PML_size = 10  # size of the PML in grid points
    N = Vector([32, 64, 64]) - 2 * PML_size  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3])  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using makeBall
    ball_magnitude = 10  # [Pa]
    ball_radius = 3  # [grid points]
    p0_array = ball_magnitude * make_ball(N, N / 2, ball_radius)
    p0_array = smooth(p0_array, restore_max=True)

    source = kSource()
    source.p0 = p0_array

    # define a binary planar sensor
    sensor = kSensor()
    sensor_mask_array = np.zeros(N)
    sensor_mask_array[0, :, :] = 1 # Corrected to be a plane for 3D
    sensor.mask = sensor_mask_array

    input_filename = directory / "kwave_input.h5"
    output_filename = directory / "kwave_output.h5"
    checkpoint_filename = directory / "kwave_checkpoint.h5"

    simulation_options = SimulationOptions(
        save_to_disk=True, # Must be true for kspaceFirstOrder3D
        pml_size=PML_size,
        pml_inside=False,
        smooth_p0=False, # p0 is already smoothed
        data_cast="single",
        input_filename=input_filename,
        output_filename=output_filename,
    )

    checkpoint_timesteps = 300

    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False, # Assuming CPU for basic test
        checkpoint_file=checkpoint_filename,
        checkpoint_timesteps=checkpoint_timesteps,
        verbose_level=0 # Keep test output clean
    )
    return kgrid, medium, source, sensor, simulation_options, execution_options


def test_kspaceFirstOrder3D_input_state_preservation():
    with TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        kgrid, medium, source, sensor, simulation_options, execution_options = make_simulation_parameters(tmpdir)

        # Store original states of critical attributes for comparison
        original_source_p0 = deepcopy(source.p0)
        original_sensor_mask = deepcopy(sensor.mask)
        
        # If source.p or source.u were time-varying, store their initial states too.
        # For this test, p0 is the main source attribute.

        # First run
        try:
            _ = kspaceFirstOrder3D(kgrid, medium, source, sensor, simulation_options, execution_options)
        except Exception as e:
            pytest.fail(f"First call to kspaceFirstOrder3D failed: {e}")

        # Check if original source and sensor attributes are unchanged
        assert np.array_equal(source.p0, original_source_p0), "source.p0 was modified after first run"
        assert np.array_equal(sensor.mask, original_sensor_mask), "sensor.mask was modified after first run"

        # Second run (should not fail if state is preserved)
        # For the second run, we need new input/output filenames or to ensure the C++ code can overwrite.
        # Easiest is to use new filenames for the test.
        simulation_options_run2 = deepcopy(simulation_options)
        simulation_options_run2.input_filename = tmpdir / "kwave_input_run2.h5"
        simulation_options_run2.output_filename = tmpdir / "kwave_output_run2.h5"
        
        execution_options_run2 = deepcopy(execution_options)
        if execution_options_run2.checkpoint_file: # Only change if it exists
            execution_options_run2.checkpoint_file = tmpdir / "kwave_checkpoint_run2.h5"

        try:
            _ = kspaceFirstOrder3D(kgrid, medium, source, sensor, simulation_options_run2, execution_options_run2)
        except Exception as e:
            pytest.fail(f"Second call to kspaceFirstOrder3D with original objects failed: {e}")

        # Final check that attributes are still the same as the initial state
        assert np.array_equal(source.p0, original_source_p0), "source.p0 was modified after second run"
        assert np.array_equal(sensor.mask, original_sensor_mask), "sensor.mask was modified after second run"

```
