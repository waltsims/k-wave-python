from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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
        if self.number_sensor_points <= 1:
            raise ValueError("number_sensor_points must be greater than 1")

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
    bon_a = dtype(6)

    if options.heterogeneous_media:
        sound_speed = c0 * np.ones((nx, ny, nz), dtype=dtype)
        sound_speed[: nx // 4, :, :] = c0 * dtype(1.2)
        density = rho0 * np.ones((nx, ny, nz), dtype=dtype)
        density[:, max(ny // 4 - 1, 0) :, :] = rho0 * dtype(1.2)
    else:
        sound_speed = np.array(c0, dtype=dtype)
        density = np.array(rho0, dtype=dtype)

    medium = kWaveMedium(sound_speed=sound_speed, density=density)
    if options.absorbing_media:
        medium.alpha_coeff = alpha_coeff
        medium.alpha_power = alpha_power
    if options.nonlinear_media:
        medium.BonA = bon_a

    source = kSource()
    source.p0 = dtype(10) * make_ball(Vector([nx, ny, nz]), Vector([nx // 2, ny // 2, nz // 2]), 2 * scale)
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


def store_case_result(result: dict[str, Any], comp_size: int, comp_time: float, mem_usage: float, report_mem_usage: bool) -> None:
    if len(result["comp_size"]) == 0 or result["comp_size"][-1] != comp_size:
        result["comp_size"].append(comp_size)
        result["comp_time"].append(comp_time)
        if report_mem_usage:
            result["mem_usage"].append(mem_usage)
        return

    result["comp_time"][-1] = comp_time
    if report_mem_usage:
        result["mem_usage"][-1] = mem_usage


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
    path.write_text(json.dumps(payload, indent=2) + "\n")


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


def peak_memory_bytes() -> float:
    try:
        import resource
    except ImportError:
        return float("nan")

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system().lower() == "darwin":
        return float(usage)
    return float(usage * 1024)
