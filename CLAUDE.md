# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is k-wave-python?

Python implementation of the [k-Wave](http://www.k-wave.org/) acoustics toolbox for time-domain acoustic and ultrasound simulations. Two backends: a pure Python/NumPy/CuPy solver and C++ binaries (OMP/CUDA).

## Commands

```bash
# Setup
uv sync --extra dev --extra test
uv run pre-commit install

# Run tests
uv run pytest                              # all tests
uv run pytest tests/test_kgrid.py          # single file
uv run pytest tests/test_kgrid.py::test_name  # single test
uv run pytest -m integration               # MATLAB-reference tests only

# Lint/format
uv run ruff check . --fix
uv run ruff format .
uv run pre-commit run --all-files

# Run an example
uv run examples/ivp_homogeneous_medium.py

# Build
uv build
```

Always use `uv run` (not `uv run python`) for pytest, examples, and scripts.

## Code Style

- Line length: 140 (configured in `pyproject.toml` under `[tool.ruff]`)
- Pre-commit hooks: ruff (lint + format), codespell, nb-clean
- Ruff ignores F821 (nested function refs) and F722 (jaxtyping annotations)
- Type annotations use `beartype` + `jaxtyping`

## Architecture

### Entry point

`kwave/kspaceFirstOrder.py` — `kspaceFirstOrder()` is the unified API. It accepts `kgrid`, `medium`, `source`, `sensor` and dispatches to the appropriate backend.

```
kspaceFirstOrder(kgrid, medium, source, sensor, backend="python"|"cpp", device="cpu"|"gpu")
```

### Two backends

- **Python backend** (`kwave/solvers/kspace_solver.py`): `Simulation` class implementing k-space pseudospectral method in NumPy/CuPy. Supports 1D/2D/3D. Uses CuPy for GPU when `device="gpu"`.
- **C++ backend** (`kwave/solvers/cpp_simulation.py` + `kwave/executor.py`): Serializes to HDF5, invokes compiled C++ binary, reads results back. `save_only=True` writes HDF5 without running (for cluster jobs).

### Core data classes

- `kWaveGrid` (`kwave/kgrid.py`) — domain discretization, spacing, time array
- `kWaveMedium` (`kwave/kmedium.py`) — sound speed, density, absorption, nonlinearity
- `kSource` (`kwave/ksource.py`) — pressure/velocity sources with masks and signals
- `kSensor` (`kwave/ksensor.py`) — sensor mask and recording configuration

### Legacy path

`kspaceFirstOrder2D()` / `kspaceFirstOrder3D()` in their respective files route through `kWaveSimulation` (`kwave/kWaveSimulation.py`) — a large legacy dataclass used by the C++ backend path. New code should use `kspaceFirstOrder()` directly.

### PML handling

When `pml_inside=False` (default), `kspaceFirstOrder()` expands the grid by `2*pml_size` before simulation and strips PML from full-grid output fields afterward. `pml_size="auto"` selects optimal sizes via `get_optimal_pml_size()`.

### Key utilities

- `kwave/utils/pml.py` — PML sizing (`get_pml()`, `get_optimal_pml_size()`)
- `kwave/utils/mapgen.py` — geometry generators (`make_disc`, `make_ball`, `make_cart_circle`, etc.)
- `kwave/utils/signals.py` — signal generation (tone bursts, filtering)
- `kwave/utils/filters.py` — spatial smoothing, Gaussian filters
- `kwave/utils/io.py` — HDF5 read/write
- `kwave/utils/conversion.py` — unit conversion, `cart2grid`

## Testing

- Tests in `tests/`, configured via `[tool.pytest.ini_options]` in `pyproject.toml`
- Integration tests (`@pytest.mark.integration`) compare against MATLAB reference data
- Test fixtures in `tests/integration/conftest.py`: `load_matlab_ref` (fixture), `assert_fields_close` (helper function for field comparison)
- MATLAB reference data for `@pytest.mark.integration` tests lives in `tests/matlab_test_data_collectors/python_testers/collectedValues/` (hardcoded in `conftest.py`)
- `test_example_parity.py` (MATLAB-parity tests) reads `KWAVE_MATLAB_REF_DIR` (defaults to `~/git/k-wave-cupy/tests` if unset)

## Naming Conventions

- `backend="python"` or `"cpp"` (not "native")
- `device="cpu"` or `"gpu"` (not `use_gpu` bool)
- `quiet=True` to suppress output; `debug=True` for detailed output
- `pml_size="auto"` for automatic PML sizing (not a separate boolean)
