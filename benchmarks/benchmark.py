"""
k-Wave 3D Performance Benchmark

Ported from: k-Wave/benchmark.m

Runs a sequence of 3D initial-value simulations with increasing grid sizes and
records average execution time for each grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from benchmarks.helpers import (
    BenchmarkOptions,
    ChildPeakMemorySampler,
    PeakMemorySampler,
    build_case,
    current_memory_bytes,
    default_output_path,
    grid_sizes,
    options_payload,
    rolling_average,
    save_results,
    store_case_result,
    validate_memory_bytes,
)
from kwave.kspaceFirstOrder import kspaceFirstOrder


def run(
    options: BenchmarkOptions = BenchmarkOptions(),
    *,
    backend: str = "python",
    device: str = "cpu",
    max_cases: int | None = None,
    output_path: str | Path | None = None,
    quiet: bool = True,
    solver: Callable[..., Any] | None = None,
    timer: Callable[[], float] = perf_counter,
    memory_reader: Callable[[], float] = current_memory_bytes,
    memory_sampling_interval: float = 0.05,
) -> dict[str, Any]:
    solver = kspaceFirstOrder if solver is None else solver
    cases = grid_sizes(options)
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]

    path = default_output_path(options) if output_path is None else Path(output_path)
    result: dict[str, Any] = {
        "comp_size": [],
        "comp_time": [],
        "options": options_payload(options, backend, device),
        "output_path": str(path),
        "error_reached": False,
        "error_message": "",
    }
    if options.report_mem_usage:
        # Probe early so unsupported combinations (e.g. Windows + cpp) fail
        # before any output file is written. cpp probes the sampler class
        # (Windows refusal happens in __init__); python probes the reader
        # so platform-missing /proc or ps surfaces here, not mid-simulation.
        try:
            if backend == "cpp":
                ChildPeakMemorySampler()
            else:
                validate_memory_bytes(memory_reader())
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
        except Exception as exc:
            raise ValueError("report_mem_usage is not supported on this platform") from exc
        result["mem_usage"] = []

    for case_index, (nx, ny, nz, scale) in enumerate(cases):
        loop_time = 0.0
        loop_mem_usage = 0.0
        try:
            kgrid, medium, source, sensor = build_case(options, nx, ny, nz, scale)
            # cpp backend: ChildPeakMemorySampler's baseline must be captured
            # once per grid (ru_maxrss is cumulative — re-baselining per
            # iteration would silently zero the delta). Python backend's
            # PeakMemorySampler is per-iteration by design.
            grid_sampler = ChildPeakMemorySampler() if (options.report_mem_usage and backend == "cpp") else None
            for loop_num in range(1, options.num_averages + 1):
                if options.report_mem_usage:
                    memory_cm = grid_sampler if grid_sampler is not None else PeakMemorySampler(memory_reader, memory_sampling_interval)
                    with memory_cm as memory_sampler:
                        start = timer()
                        solver(
                            kgrid,
                            medium,
                            source,
                            sensor,
                            backend=backend,
                            device=device,
                            quiet=quiet,
                            pml_size=options.pml_size,
                            pml_inside=options.pml_inside,
                            smooth_p0=False,
                        )
                        elapsed_time = timer() - start
                    loop_mem_usage = rolling_average(loop_mem_usage, memory_sampler.peak_bytes, loop_num)
                else:
                    start = timer()
                    solver(
                        kgrid,
                        medium,
                        source,
                        sensor,
                        backend=backend,
                        device=device,
                        quiet=quiet,
                        pml_size=options.pml_size,
                        pml_inside=options.pml_inside,
                        smooth_p0=False,
                    )
                    elapsed_time = timer() - start
                loop_time = rolling_average(loop_time, elapsed_time, loop_num)
                store_case_result(result, case_index, nx * ny * nz, loop_time, loop_mem_usage, options.report_mem_usage)
                save_results(path, result)
        except Exception as exc:
            result["error_reached"] = True
            result["error_message"] = str(exc)
            save_results(path, result)
            break

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the k-Wave 3D performance benchmark.")
    parser.add_argument("--data-cast", choices=("off", "single"), default="off")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-averages", type=int, default=3)
    parser.add_argument("--number-time-points", type=int, default=1000)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--report-mem-usage", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    benchmark_options = BenchmarkOptions(
        data_cast=args.data_cast,
        num_averages=args.num_averages,
        number_time_points=args.number_time_points,
        report_mem_usage=args.report_mem_usage,
    )
    result = run(
        benchmark_options,
        backend=args.backend,
        device=args.device,
        max_cases=args.max_cases,
        output_path=args.output_path,
        quiet=not args.verbose,
    )
    print(f"Benchmark results saved to {result['output_path']}")
    if result["error_reached"]:
        print("Memory limit reached or error encountered, exiting benchmark. Error message:", file=sys.stderr)
        print(f"  {result['error_message']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
