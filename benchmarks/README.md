# Benchmarks

This directory contains performance benchmarks for k-wave-python. These are not standard examples: they are intended to measure runtime and memory behavior and can become expensive as the grid size grows.

## 3D Solver Scaling Benchmark

`benchmark.py` ports MATLAB k-Wave's `benchmark.m`. It runs `kspaceFirstOrder` on a sequence of 3D grids with increasing sizes, averages runtime over repeated runs, and saves partial results after each run.

The default benchmark uses:

- heterogeneous absorbing medium
- smoothed initial pressure ball source
- binary sensor mask built from 100 Cartesian points on a sphere
- 1000 time steps
- 3 averages per grid size
- grid sizes based on MATLAB's original scale arrays, starting at `32 x 32 x 32`

By default, this can run for a long time and may stop once memory limits are reached.

## Usage

Run a small smoke benchmark:

```bash
uv run benchmarks/benchmark.py --max-cases 1 --num-averages 1 --number-time-points 20
```

Run the default CPU benchmark:

```bash
uv run benchmarks/benchmark.py
```

Run with single-precision arrays:

```bash
uv run benchmarks/benchmark.py --data-cast single
```

Run on the Python GPU backend:

```bash
uv run benchmarks/benchmark.py --device gpu
```

Choose an output file:

```bash
uv run benchmarks/benchmark.py --output-path benchmark_data.json
```

## Output

The benchmark writes a JSON file containing:

- `comp_size`: total grid points for each completed grid size
- `comp_time`: rolling average elapsed seconds for each grid size
- `options`: benchmark settings and environment metadata
- `output_path`: path to the JSON output file
- `error_reached`: whether the benchmark stopped after an exception
- `error_message`: exception message, if any
- `mem_usage`: optional process peak memory estimate when `--report-mem-usage` is set

Partial results are saved after each run so completed timings are preserved if a later grid fails.
