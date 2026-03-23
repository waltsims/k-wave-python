# CLAUDE.md — k-wave-python

## What this project is

k-Wave-python is a Python port of [k-Wave](http://www.k-wave.org/), a MATLAB toolbox for time-domain acoustic and ultrasound simulations using the k-space pseudospectral method. It supports 1D, 2D, and 3D simulations with optional GPU acceleration via CuPy.

## Key commands

```bash
uv run pytest tests/                    # Run all tests
uv run pytest tests/test_native_solver.py  # Run solver tests
uv run python run_examples.py           # Run all examples (python backend)
uv run python examples/ivp_1D_simulation.py  # Run a single example
```

## Architecture

- **Entry point:** `kwave/kspaceFirstOrder.py` — `kspaceFirstOrder()` is the unified API
- **Python solver:** `kwave/solvers/kspace_solver.py` — `Simulation` class (setup/step/run)
- **C++ backend:** `kwave/solvers/cpp_simulation.py` — HDF5 serialization + binary execution
- **Validation:** `kwave/solvers/validation.py`
- **Legacy bridge:** `kwave/solvers/native.py` — minimal wrapper for deprecated `kspaceFirstOrder2D/3D`
- **MATLAB interop:** `simulate_from_dicts()` in `kspace_solver.py`, shim in k-wave-cupy repo

## Critical conventions

### Memory layout: Fortran vs C order

k-Wave is a MATLAB project. MATLAB uses **column-major (Fortran) order**. NumPy defaults to **row-major (C) order**.

**Current state:** The solver uses `order="F"` extensively for MATLAB compatibility. This is being migrated:
- **Internal solver logic** (FFT, PML, stepping) is order-agnostic
- **Boundary layers** (HDF5 serialization, MATLAB interop) must use F-order
- **Goal for v1.0:** C-order internally, F-order only at boundaries

When modifying array reshaping/flattening code, always check whether `order="F"` is needed for correctness or is legacy MATLAB convention.

### Indexing: 0-based vs 1-based

MATLAB uses 1-based indexing. Two places where this matters:

1. **`sensor.record_start_index`** — uses 1-based (MATLAB convention). Converted to 0-based internally in `_setup_sensor_mask`.
2. **C++ HDF5 format** — sensor/source indices are 1-based. The `+1` in `cpp_simulation.py` is intentional.
3. **`matlab_find()`** — returns 1-based indices. Used by `kWaveArray`.

### k-wave-cupy interop

The k-wave-cupy repo contains the MATLAB wrapper (`kspaceFirstOrderPy.m`) that calls Python via `py.kWavePy.simulate_from_dicts()`. The `kWavePy.py` shim in that repo re-exports from k-wave-python.

- `simulate_from_dicts(device=...)` — NOT `backend=`. The parameter was renamed.
- MATLAB passes dicts with F-order arrays and 1-based indices
- The shim handles `pml_inside=False` expansion externally

### Backend vs Device naming

- `backend` = `"python"` or `"cpp"` (which engine runs the simulation)
- `device` = `"cpu"` or `"gpu"` (hardware target)

These are separate concerns. `kspaceFirstOrder(backend="python", device="gpu")` runs the Python solver on GPU via CuPy.

## Environment variables

- `KWAVE_BACKEND` — Override backend for all `kspaceFirstOrder()` calls (used by CI)
- `KWAVE_DEVICE` — Override device for all `kspaceFirstOrder()` calls (used by CI)

## Examples

Examples are `.py` files with `# %%` cell markers (percent format). They work as:
- Plain scripts: `python examples/foo.py`
- Interactive notebooks: Open in VS Code/JupyterLab
- Jupyter notebooks: Generated via `jupytext --to notebook`

Source of truth is always the `.py` file. Never edit `.ipynb` directly.

## Testing

```bash
uv run pytest tests/test_native_solver.py     # Solver + physics tests
uv run pytest tests/test_validation.py        # Input validation
uv run pytest tests/test_kspaceFirstOrder.py  # Entry point tests
uv run pytest tests/test_compat.py            # Legacy options migration
```

## Style

- Simple > less code > fast
- No backward compatibility aliases — clean breaks
- `quiet=True` to silence, `debug=True` for verbose (not `verbose=True`)
- Use `uv run` for all tools (pytest, python, etc.)
