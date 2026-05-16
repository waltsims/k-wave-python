import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark import BenchmarkOptions, build_case, grid_sizes, run


def small_options(**overrides):
    values = {
        "start_size": 8,
        "x_scale_array": (1,),
        "y_scale_array": (1,),
        "z_scale_array": (1,),
        "number_sensor_points": 8,
        "number_time_points": 5,
        "num_averages": 1,
        "sensor_radius": 1e-3,
    }
    values.update(overrides)
    return BenchmarkOptions(**values)


def test_grid_sizes_match_matlab_scale_arrays():
    sizes = grid_sizes(BenchmarkOptions())

    assert sizes[0] == (32, 32, 32, 1)
    assert sizes[-1] == (512, 512, 256, 8)
    assert len(sizes) == 12


def test_build_case_matches_benchmark_defaults_for_small_grid():
    options = small_options()

    kgrid, medium, source, sensor = build_case(options, 8, 8, 8, 1)

    assert tuple(kgrid.N) == (8, 8, 8)
    assert np.isclose(kgrid.dx, options.domain_size / 8)
    assert kgrid.Nt == options.number_time_points
    assert medium.sound_speed.shape == (8, 8, 8)
    assert np.all(medium.sound_speed[:2, :, :] == 1800)
    assert np.all(medium.sound_speed[2:, :, :] == 1500)
    assert np.all(medium.density[:, :1, :] == 1000)
    assert np.all(medium.density[:, 1:, :] == 1200)
    assert medium.alpha_coeff == pytest.approx(0.75)
    assert medium.alpha_power == pytest.approx(1.5)
    assert source.p0.shape == (8, 8, 8)
    assert np.max(source.p0) == pytest.approx(10)
    assert sensor.mask.shape == (8, 8, 8)
    assert sensor.mask.dtype == bool
    assert 0 < np.count_nonzero(sensor.mask) <= options.number_sensor_points
    assert sensor.record == ["p"]


def test_single_data_cast_uses_float32_arrays():
    options = small_options(data_cast="single")

    _, medium, source, _ = build_case(options, 8, 8, 8, 1)

    assert medium.sound_speed.dtype == np.float32
    assert medium.density.dtype == np.float32
    assert source.p0.dtype == np.float32


def test_run_aggregates_timings_and_saves_json_file(tmp_path: Path):
    options = small_options(num_averages=2)
    output_path = tmp_path / "benchmark.json"
    times = iter([0.0, 1.0, 1.0, 3.0])
    calls = []

    def fake_solver(kgrid, medium, source, sensor, **kwargs):
        calls.append((kgrid, medium, source, sensor, kwargs))
        return {"p": np.zeros((1, 1))}

    result = run(options, max_cases=1, output_path=output_path, solver=fake_solver, timer=lambda: next(times))

    assert len(calls) == 2
    assert calls[0][4]["pml_size"] == options.pml_size
    assert calls[0][4]["pml_inside"] is True
    assert calls[0][4]["smooth_p0"] is False
    assert result["comp_size"] == [8 * 8 * 8]
    assert result["comp_time"] == [pytest.approx(1.5)]
    assert result["output_path"] == str(output_path)
    assert result["error_reached"] is False
    assert output_path.exists()

    saved = json.loads(output_path.read_text())
    assert saved["comp_size"] == [8 * 8 * 8]
    assert saved["comp_time"] == [pytest.approx(1.5)]
    assert saved["output_path"] == str(output_path)
    assert saved["options"]["start_size"] == 8


def test_run_saves_partial_results_after_solver_error(tmp_path: Path):
    options = small_options()
    output_path = tmp_path / "benchmark.json"

    def failing_solver(*args, **kwargs):
        raise RuntimeError("solver failed")

    result = run(options, max_cases=1, output_path=output_path, solver=failing_solver)

    assert result["error_reached"] is True
    assert result["error_message"] == "solver failed"
    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["error_reached"] is True
    assert saved["error_message"] == "solver failed"
