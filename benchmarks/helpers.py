from __future__ import annotations

import json
import math
import platform
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil

import kwave
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.utils.conversion import cart2grid
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball, make_cart_sphere


@dataclass(frozen=True)
class BenchmarkOptions:
    data_cast: str = "off"
    heterogeneous_media: bool = True
    absorbing_media: bool = True
    nonlinear_media: bool = False
    binary_sensor_mask: bool = True
    number_sensor_points: int = 100
    number_time_points: int = 1000
    num_averages: int = 3
    start_size: int = 32
    x_scale_array: tuple[int, ...] = (1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16)
    y_scale_array: tuple[int, ...] = (1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16)
    z_scale_array: tuple[int, ...] = (1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8)
    domain_size: float = 22e-3
    sensor_radius: float = 10e-3
    pml_size: int = 10
    pml_inside: bool = True
    report_mem_usage: bool = False

    def __post_init__(self) -> None:
        if self.data_cast not in {"off", "single"}:
            raise ValueError("data_cast must be 'off' or 'single'. Use device='gpu' to run on a GPU.")
        if not (len(self.x_scale_array) == len(self.y_scale_array) == len(self.z_scale_array)):
            raise ValueError("scale arrays must have the same length")
        if self.number_time_points <= 0 or self.num_averages <= 0:
            raise ValueError("number_time_points and num_averages must be positive")
        if self.start_size <= 0:
            raise ValueError("start_size must be positive")
        if self.number_sensor_points <= 1:
            raise ValueError("number_sensor_points must be greater than 1")
        if self.domain_size <= 0:
            raise ValueError("domain_size must be positive")
        if self.sensor_radius <= 0:
            raise ValueError("sensor_radius must be positive")
        if self.pml_size < 0:
            raise ValueError("pml_size must be non-negative")

    @property
    def dtype(self) -> type[np.floating[Any]]:
        return np.float32 if self.data_cast == "single" else np.float64


def grid_sizes(options: BenchmarkOptions = BenchmarkOptions()) -> list[tuple[int, int, int, int]]:
    return [
        (
            options.start_size * xscale,
            options.start_size * yscale,
            options.start_size * zscale,
            min(xscale, yscale, zscale),
        )
        for xscale, yscale, zscale in zip(options.x_scale_array, options.y_scale_array, options.z_scale_array)
    ]


def build_case(options: BenchmarkOptions, nx: int, ny: int, nz: int, scale: int) -> tuple[kWaveGrid, kWaveMedium, kSource, kSensor]:
    dtype = options.dtype
    dx = options.domain_size / nx
    dy = options.domain_size / ny
    dz = options.domain_size / nz
    kgrid = kWaveGrid(Vector([nx, ny, nz]), Vector([dx, dy, dz]))

    c0 = dtype(1500)
    rho0 = dtype(1000)
    alpha_coeff = dtype(0.75)
    alpha_power = dtype(1.5)

    if options.heterogeneous_media:
        sound_speed = c0 * np.ones((nx, ny, nz), dtype=dtype)
        # MATLAB: sound_speed(1:Nx/4, :, :)  →  Python: [:nx//4, :, :] (head slice, no -1).
        sound_speed[: nx // 4, :, :] = c0 * dtype(1.2)
        density = rho0 * np.ones((nx, ny, nz), dtype=dtype)
        # MATLAB: density(:, Ny/4:end, :)  →  Python: [:, ny//4-1:, :] (tail slice — the
        # -1 converts MATLAB's 1-indexed inclusive start to Python's 0-indexed start).
        # max(..., 0) guards tiny grids where ny//4 - 1 would be negative.
        density[:, max(ny // 4 - 1, 0) :, :] = rho0 * dtype(1.2)
    else:
        sound_speed = np.array(c0, dtype=dtype)
        density = np.array(rho0, dtype=dtype)

    medium = kWaveMedium(sound_speed=sound_speed, density=density)
    if options.absorbing_media:
        medium.alpha_coeff = alpha_coeff
        medium.alpha_power = alpha_power
    if options.nonlinear_media:
        medium.BonA = dtype(6)

    source = kSource()
    # make_ball treats the supplied center as 1-indexed internally (mapgen.py),
    # so passing nx//2 (= MATLAB's Nx/2 1-indexed value) yields the same centroid.
    source.p0 = dtype(10) * make_ball(Vector([nx, ny, nz]), Vector([nx // 2, ny // 2, nz // 2]), 2 * scale)
    # smooth() upcasts to float64 via the FFT path even when given float32;
    # the trailing .astype(dtype) restores user-requested precision.
    source.p0 = smooth(source.p0.astype(dtype, copy=False), restore_max=True).astype(dtype, copy=False)

    sensor_mask = make_cart_sphere(options.sensor_radius, options.number_sensor_points)
    if options.binary_sensor_mask:
        sensor_mask, _, _ = cart2grid(kgrid, sensor_mask, order="C")
        sensor_mask = sensor_mask.astype(bool)
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]

    kgrid.makeTime(np.max(np.asarray(medium.sound_speed)))
    kgrid.setTime(options.number_time_points, kgrid.dt)

    return kgrid, medium, source, sensor


def default_output_path(options: BenchmarkOptions) -> Path:
    computer_name = platform.node() or "unknown-computer"
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"benchmark_data-{computer_name}-{options.data_cast}-{date}.json")


def rolling_average(previous_average: float, new_value: float, count: int) -> float:
    return (previous_average * (count - 1) + new_value) / count


def store_case_result(
    result: dict[str, Any], case_index: int, comp_size: int, comp_time: float, mem_usage: float, report_mem_usage: bool
) -> None:
    if case_index == len(result["comp_size"]):
        result["comp_size"].append(comp_size)
        result["comp_time"].append(comp_time)
        if report_mem_usage:
            result["mem_usage"].append(mem_usage)
        return

    result["comp_size"][case_index] = comp_size
    result["comp_time"][case_index] = comp_time
    if report_mem_usage:
        result["mem_usage"][case_index] = mem_usage


def save_results(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comp_size": [int(size) for size in result["comp_size"]],
        "comp_time": [float(time) for time in result["comp_time"]],
        "options": result["options"],
        "output_path": result["output_path"],
        "error_reached": bool(result["error_reached"]),
        "error_message": result["error_message"],
    }
    if "mem_usage" in result:
        payload["mem_usage"] = [float(usage) for usage in result["mem_usage"]]
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def options_payload(options: BenchmarkOptions, backend: str, device: str) -> dict[str, Any]:
    payload = asdict(options)
    payload.update(
        {
            "backend": backend,
            "device": device,
            "computer_name": platform.node(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "kwave_python_version": kwave.__version__,
        }
    )
    return payload


def validate_memory_bytes(value: float) -> float:
    memory_bytes = float(value)
    if not math.isfinite(memory_bytes) or memory_bytes < 0:
        raise ValueError("memory usage must be a finite non-negative value")
    return memory_bytes





def current_memory_bytes() -> float:
    """Return the current resident-set-size of THIS Python process, in bytes.

    Note: this returns the *current* RSS at the moment of the call, not a
    historical peak. Use ``PeakMemorySampler`` to track peak-over-time, or
    ``ChildPeakMemorySampler`` for subprocess (``backend="cpp"``) measurement.
    """
    return validate_memory_bytes(psutil.Process().memory_info().rss)


# Back-compat alias — the old name is misleading but kept so external callers
# (and any pre-merge benchmark JSON tooling) don't break.
peak_memory_bytes = current_memory_bytes


class ChildPeakMemorySampler:
    """Capture the peak RSS of child processes spawned during the sampler's lifetime.

    Use this for ``backend="cpp"`` benchmark runs where the simulation work
    happens in a subprocess — ``PeakMemorySampler`` only sees the Python
    parent's RSS and reports a number that has nothing to do with the C++
    process's actual footprint.

    **Lifetime model — use ONE instance per grid size, enter/exit per iteration.**
    ``resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`` is a monotonically
    non-decreasing cumulative max across *all* reaped children since the
    parent process started, so capturing a baseline at each ``__enter__``
    would yield ``delta == 0`` for any iteration after the first when the
    workload is uniform. Instead the baseline is captured at construction
    and ``peak_bytes`` returns growth since then; reuse the same instance
    across all iterations of a grid size so the answer reflects the peak
    for that grid's runs, not a per-iteration delta that's always zero.

    On Linux ``ru_maxrss`` is reported in kilobytes; on macOS it's in bytes
    (per BSD historical convention). Windows has no equivalent in
    ``resource`` — the sampler raises ``NotImplementedError`` on Windows so
    the caller can refuse the combination cleanly.
    """

    def __init__(self) -> None:
        if platform.system().lower() == "windows":
            raise NotImplementedError(
                "ChildPeakMemorySampler is not supported on Windows "
                "(resource.RUSAGE_CHILDREN is POSIX-only). "
                "Use backend='python' to measure simulation memory on Windows."
            )
        import resource  # POSIX-only; gated by the Windows check above

        self._baseline = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        self._latest = self._baseline

    def __enter__(self) -> ChildPeakMemorySampler:
        # No-op: ru_maxrss is cumulative across the lifetime of the parent,
        # so re-capturing a baseline at every iteration would silently
        # collapse the reported delta to zero. Baseline is fixed at __init__.
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any) -> None:
        import resource

        self._latest = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    @property
    def peak_bytes(self) -> float:
        delta = max(0, self._latest - self._baseline)
        # ru_maxrss units differ by platform (Linux: KB, macOS: bytes).
        if platform.system().lower() == "linux":
            return float(delta * 1024)
        return float(delta)


class PeakMemorySampler:
    def __init__(self, reader: Callable[[], float] = current_memory_bytes, interval: float = 0.05):
        self._reader = reader
        self._interval = interval
        self._peak_bytes = 0.0
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_bytes(self) -> float:
        with self._lock:
            return self._peak_bytes

    def __enter__(self) -> PeakMemorySampler:
        self._record_sample()
        if self._interval > 0:
            self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self._record_sample()
        except Exception as exc:
            if exc_type is None:
                raise exc
        if exc_type is None and self._error is not None:
            raise self._error

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self._record_sample()
            except Exception as exc:
                self._error = exc
                self._stop_event.set()

    def _record_sample(self) -> None:
        memory_bytes = validate_memory_bytes(self._reader())
        with self._lock:
            self._peak_bytes = max(self._peak_bytes, memory_bytes)
