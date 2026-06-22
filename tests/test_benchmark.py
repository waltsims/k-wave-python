import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.benchmark import BenchmarkOptions, build_case, grid_sizes, run
from benchmarks.helpers import ChildPeakMemorySampler


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


def test_run_keeps_distinct_cases_with_the_same_total_grid_points(tmp_path: Path):
    options = small_options(
        x_scale_array=(1, 2),
        y_scale_array=(2, 1),
        z_scale_array=(1, 1),
    )
    output_path = tmp_path / "benchmark.json"
    times = iter([0.0, 1.0, 1.0, 3.0])
    shapes = []

    def fake_solver(kgrid, *args, **kwargs):
        shapes.append(tuple(kgrid.N))
        return {"p": np.zeros((1, 1))}

    result = run(options, output_path=output_path, solver=fake_solver, timer=lambda: next(times))

    assert shapes == [(8, 16, 8), (16, 8, 8)]
    assert result["comp_size"] == [8 * 16 * 8, 16 * 8 * 8]
    assert result["comp_time"] == [pytest.approx(1.0), pytest.approx(2.0)]

    saved = json.loads(output_path.read_text())
    assert saved["comp_size"] == [8 * 16 * 8, 16 * 8 * 8]
    assert saved["comp_time"] == [pytest.approx(1.0), pytest.approx(2.0)]


def test_start_size_must_be_positive():
    with pytest.raises(ValueError, match="start_size must be positive"):
        small_options(start_size=0)


def test_run_reports_memory_usage_and_saves_valid_json(tmp_path: Path):
    options = small_options(report_mem_usage=True, num_averages=2)
    output_path = tmp_path / "benchmark.json"
    times = iter([0.0, 1.0, 1.0, 3.0])
    memory_samples = iter([512.0, 1024.0, 2048.0, 4096.0, 8192.0])

    def fake_solver(*args, **kwargs):
        return {"p": np.zeros((1, 1))}

    result = run(
        options,
        max_cases=1,
        output_path=output_path,
        solver=fake_solver,
        timer=lambda: next(times),
        memory_reader=lambda: next(memory_samples),
        memory_sampling_interval=0,
    )

    assert result["mem_usage"] == [pytest.approx(5120.0)]

    saved = json.loads(output_path.read_text())
    assert saved["mem_usage"] == [pytest.approx(5120.0)]


def test_report_memory_usage_fails_before_writing_when_unavailable(tmp_path: Path):
    options = small_options(report_mem_usage=True)
    output_path = tmp_path / "benchmark.json"

    def unsupported_memory_reader():
        raise RuntimeError("memory unavailable")

    with pytest.raises(ValueError, match="report_mem_usage is not supported"):
        run(options, max_cases=1, output_path=output_path, memory_reader=unsupported_memory_reader)

    assert not output_path.exists()


@pytest.mark.parametrize(
    "kwargs, expected_match",
    [
        ({"domain_size": 0}, "domain_size must be positive"),
        ({"domain_size": -1e-3}, "domain_size must be positive"),
        ({"sensor_radius": 0}, "sensor_radius must be positive"),
        ({"sensor_radius": -1}, "sensor_radius must be positive"),
        ({"pml_size": -1}, "pml_size must be non-negative"),
    ],
)
def test_options_post_init_rejects_invalid_geometry(kwargs, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        small_options(**kwargs)


class _FakeChildSampler:
    """Test-only stand-in for ChildPeakMemorySampler. Records construction
    and enter/exit counts so we can assert the "one instance per grid,
    N enter/exits per grid" lifetime model is honored."""

    instances: list = []
    enter_count = 0

    def __init__(self):
        type(self).instances.append(self)
        self._peak = 0

    def __enter__(self):
        type(self).enter_count += 1
        return self

    def __exit__(self, *exc):
        # Simulate cumulative growth: peak grows by 1 KB each iteration.
        self._peak += 1024

    @property
    def peak_bytes(self) -> float:
        return float(self._peak)

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.enter_count = 0


def test_cpp_backend_shares_child_sampler_across_iterations(tmp_path: Path):
    """For backend='cpp', one ChildPeakMemorySampler is constructed per grid
    (NOT per iteration). ru_maxrss is cumulative; re-baselining per iter
    would silently zero the delta (Greptile P1 on the original impl)."""
    _FakeChildSampler.reset()
    options = small_options(report_mem_usage=True, num_averages=3)
    output_path = tmp_path / "benchmark.json"
    times = iter([0.0, 1.0, 1.0, 3.0, 3.0, 7.0])

    def fake_solver(*args, **kwargs):
        return {"p": np.zeros((1, 1))}

    with patch("benchmarks.benchmark.ChildPeakMemorySampler", _FakeChildSampler):
        result = run(
            options,
            backend="cpp",
            max_cases=1,
            output_path=output_path,
            solver=fake_solver,
            timer=lambda: next(times),
        )

    # max_cases=1 grid → constructed twice: once at the startup probe, once
    # for the actual grid loop. NOT once per iteration.
    assert len(_FakeChildSampler.instances) == 2
    # num_averages=3 iterations on that one grid → 3 enter/exit pairs on the SAME instance.
    assert _FakeChildSampler.enter_count == 3
    assert _FakeChildSampler.instances[1].peak_bytes == pytest.approx(3072.0)
    assert "mem_usage" in result and len(result["mem_usage"]) == 1


def test_cpp_backend_reports_clear_error_when_child_sampler_unsupported(tmp_path: Path):
    """Windows + cpp + report_mem_usage should raise a clear error at startup."""
    options = small_options(report_mem_usage=True)
    output_path = tmp_path / "benchmark.json"

    def raises_not_implemented(*args, **kwargs):
        raise NotImplementedError("ChildPeakMemorySampler is not supported on Windows (resource.RUSAGE_CHILDREN is POSIX-only).")

    with patch("benchmarks.benchmark.ChildPeakMemorySampler", raises_not_implemented):
        with pytest.raises(ValueError, match="not supported on Windows"):
            run(options, backend="cpp", max_cases=1, output_path=output_path)
    # Failed-fast, before writing the output file.
    assert not output_path.exists()


def test_child_peak_memory_sampler_refuses_on_windows():
    # Patch the module-level _PLATFORM constant (computed at import time).
    with patch("benchmarks.helpers._PLATFORM", "windows"):
        with pytest.raises(NotImplementedError, match="not supported on Windows"):
            ChildPeakMemorySampler()


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
