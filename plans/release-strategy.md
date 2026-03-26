# k-wave-python Release Strategy

## Overview

This release strategy brings the unified solver architecture to fruition.

## Version Roadmap

| Version | Milestone | Focus |
|---------|-----------|-------|
| **0.5.0** | Finalize master/main | Stabilize current codebase |
| **0.6.0** | Python Solver + Unified API + Deprecation | Python solver, `kspaceFirstOrder()` kwargs, deprecation warnings |
| **0.6.1** | C-order Migration (Helpers) | Migrate utils/helpers from F-order to C-order, keep legacy API working |
| **0.6.2** | Example Migration | Port remaining examples to new `kspaceFirstOrder()` API |
| **0.6.3** | Axisymmetric Support | Axisymmetric solver in new API, port AS examples |
| **0.7.0** | CLI (`kwp`) | Command-line interface for running simulations |
| **1.0.0** | Clean Release | Remove deprecated code. Simple, readable, fast. |
| **2.0.0** | Performance & Scale | nanobind CUDA, MPI, Devito, multi-GPU |

---

## Phase 1: v0.5.0 - Finalize master/main

**Goal:** Stabilize the current codebase and release.

**Tasks:**
1. Resolve any outstanding issues on master
2. Ensure CI passes across all platforms
3. Tag and release v0.5.0

---

## Phase 2: v0.6.0 - Native Solver + Unified API + Deprecation

**Goal:** Ship a pure Python/CuPy native solver, create a single entry point with kwargs, and deprecate the old API. Three things in one release.

### 2a: Native Solver

A minimal, dimension-generic k-space pseudospectral solver implemented as a single `Simulation` class (~430 lines) in `kwave/solvers/kspace_solver.py`.

**Design decisions:**

- **N-D generic:** One code path handles 1D, 2D, and 3D. Dimension is auto-detected from the grid. No separate `kspaceFirstOrder1D/2D/3D` implementations — loops over `range(ndim)` handle per-axis operations (PML, velocity, density splits).

- **NumPy/CuPy backend via `self.xp`:** A single `xp` attribute toggles between NumPy and CuPy. All array operations use `xp.fft.*`, `xp.zeros()`, etc. No CUDA-specific code — CuPy's NumPy-compatible API handles GPU dispatch transparently. This gives CPU and GPU support with zero code duplication.

- **setup/step/run separation:** `setup()` builds all operators (PML profiles, k-space gradient/divergence operators, physics lambdas). `step()` advances one time step. `run()` loops to completion. This separation enables interactive debugging — inspect `sim.p`, `sim.u`, `sim.kappa` after setup, or step-by-step through the time loop.

- **Spectral differentiation via `_diff()`:** The core compute primitive. Computes `real(ifft(op * kappa * fft(f)))` — forward FFT, element-wise multiply by operator and kappa correction, inverse FFT, extract real part. Called 2×ndim times per step (gradients + divergences). Uses `xp.fft.fft/fft2/fftn` based on dimensionality. When CuPy is the backend, this uses cuFFT under the hood.

- **Split-field PML:** Each dimension has its own density split (`rho_split[i]`) and PML operator (`pml_list[i]`, `pml_sg_list[i]`). Double PML application (`pml * (pml * field - ...)`) implements second-order absorption. PML profiles are 1D arrays reshaped for broadcasting — no full-grid PML matrices.

- **Physics as lambdas:** Absorption, dispersion, and nonlinearity are set up as lambda functions during `setup()`. Disabled features return `0` (zero overhead). Power-law absorption uses fractional Laplacian operators precomputed once. This avoids if-branches in the hot loop.

- **SimpleNamespace inputs:** Takes `SimpleNamespace` objects (not kWave dataclasses directly). An adapter (`kwave_adapter.py`) bridges kWave objects to this format. This keeps the solver decoupled from the kWave class hierarchy and enables direct use from MATLAB via dict-to-namespace conversion.

- **MATLAB interop:** `simulate_from_dicts()` and `create_simulation()` accept plain dicts (from MATLAB's `py.dict()`), normalize field names (`c0` → `sound_speed`), and return results. This makes k-wave-cupy a thin MATLAB wrapper calling into k-wave-python.

**Tasks:**
1. Finalize `kwave/solvers/kspace_solver.py` — the `Simulation` class
2. Finalize `kwave/solvers/kwave_adapter.py` — bridge to kWave dataclasses
3. Finalize `kwave/solvers/native.py` — `NativeSolver` wrapper
4. Port tests, validate against MATLAB references (<2e-15 relative error)

### 2b: Unified API

**Create:** `kwave/kspaceFirstOrder.py`
```python
def kspaceFirstOrder(
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    source: kSource,
    sensor: kSensor,
    *,
    # Physics (replaces SimulationOptions)
    pml_size: int | tuple = 20,
    pml_alpha: float | tuple = 2.0,
    use_sg: bool = True,
    use_kspace: bool = True,
    smooth_p0: bool = True,
    # Backend (replaces SimulationExecutionOptions)
    backend: str = "cpp",  # "cpp" or "python"
    use_gpu: bool = False,
    # Execution
    save_only: bool = False,
    data_path: str = None,
    verbose: bool = False,
    num_threads: int = None,
    device_num: int = None,
) -> dict:
```

**Simplify:** `kspaceFirstOrder2D.py` and `kspaceFirstOrder3D.py` to ~40-line wrappers
```python
def kspaceFirstOrder2D(kgrid, source, sensor, medium,
                       simulation_options=None, execution_options=None, **kwargs):
    # Unpack old options to kwargs for backward compat
    # Call kspaceFirstOrder()
```

**Create:** `kwave/solvers/serializer.py` - Simple HDF5 writer (~200 lines)

### 2c: Deprecation Warnings

**Add deprecation warnings on old API:**
```python
warnings.warn(
    "SimulationOptions is deprecated. Use kspaceFirstOrder() kwargs instead. "
    "See https://waltersimson.com/k-wave-python/migration",
    DeprecationWarning
)
```

**Create migration guide** covering:
- Options classes → kwargs
- Entry point consolidation (kspaceFirstOrder2DC → kspaceFirstOrder with backend="cpp")
- Backend selection changes

**Tests to add:**
- `tests/test_native_solver.py` - parity with C++ backend
- `tests/test_cupy_backend.py` - GPU acceleration (when CUDA available)
- `tests/test_unified_api.py` - new kwargs API
- `tests/test_deprecation_warnings.py` - verify warnings emitted

---

## Phase 2.1: v0.6.1 - C-order Migration (Helpers)

**Goal:** Migrate helper/utility code from Fortran-order to C-order internally, while keeping the legacy API functional. No user-facing API changes. Fix hacky shaping/indexing throughout.

**Scope:** 55 occurrences of `order="F"` across 10 files. Migrate where possible, keep F-order only at explicit boundaries.

**Migrate (internal helpers — safe to convert):**

| File | `order="F"` count | Notes |
|------|-------------------|-------|
| `kwave/utils/matlab.py` | 11 | reshape, flatten, unflatten utilities |
| `kwave/kgrid.py` | 8 | Grid coordinate generation |
| `kwave/utils/mapgen.py` | 6 | Map generation |
| `kwave/utils/conversion.py` | 2 | Unit conversion |
| `kwave/utils/matrix.py` | 1 | Matrix utilities |
| `kwave/solvers/kspace_solver.py` | 19 | Audit each: keep F-order only at `simulate_from_dicts` boundary. Convert internal field storage, `_expand_to_grid`, `_build_source_op`, sensor mask extraction. |
| `kwave/kspaceFirstOrder.py` | 1 | Likely removable |

**Keep as-is (fixed boundaries):**

| File | Reason |
|------|--------|
| `kwave/solvers/cpp_simulation.py` | C++ binary expects F-order HDF5 |
| `kwave/kWaveSimulation.py` | Legacy, deleted in v1.0.0 |
| `kwave/kWaveSimulation_helper/*` | Legacy, deleted in v1.0.0 |

**Testing:**
- All existing tests must pass (no behavioral change)
- Add `tests/test_memory_layout.py`: verify C/F-order input produces identical results, `simulate_from_dicts` round-trips correctly, `cpp_simulation` writes correct F-order HDF5

---

## Phase 2.2: v0.6.2 - Example Migration

**Goal:** Port remaining examples from legacy `kspaceFirstOrder2D/3D` to `kspaceFirstOrder()`.

| Example | Blocker |
|---------|---------|
| `pr_2D_TR_line_sensor`, `pr_3D_TR_planar_sensor` | `TimeReversal` class uses legacy API internally |
| `us_defining_transducer`, `us_beam_patterns`, `us_bmode_linear_transducer`, `us_bmode_phased_array` | `NotATransducer`-as-source pipeline untested with new API |
| `checkpointing/checkpoint.py` | `checkpoint_file`/`checkpoint_timesteps` not exposed in new API |

---

## Phase 2.3: v0.6.3 - Axisymmetric Support

**Goal:** Add axisymmetric simulation to `kspaceFirstOrder()` and port AS examples.

**Current state:** `kspaceFirstOrderAS.py` / `kspaceFirstOrderASC.py` are standalone entry points. New API hardcodes `axisymmetric_flag=0`.

**Tasks:**
1. Add `axisymmetric: bool = False` kwarg to `kspaceFirstOrder()`
2. Implement radial symmetry terms in `Simulation` class (operates on (r, z) grid)
3. Port `at_circular_piston_AS`, `at_focused_bowl_AS`
4. Validate against MATLAB references
5. Deprecate `kspaceFirstOrderAS` / `kspaceFirstOrderASC`

---

## Phase 3: v1.0.0 - Clean Release

**Goal:** Simple, readable, fast. Remove all deprecated code.

**Delete:**
- `kwave/options/simulation_options.py`
- `kwave/options/simulation_execution_options.py`
- `kwave/kWaveSimulation.py`
- `kwave/kWaveSimulation_helper/` (entire directory)
- `kwave/solvers/base.py`, `cpp.py`, `native.py`, `kwave_adapter.py`

**Keep simplified:**
- `kwave/kspaceFirstOrder.py` (~200 lines)
- `kwave/kspaceFirstOrder2D.py` (~40 lines, compat wrapper)
- `kwave/kspaceFirstOrder3D.py` (~40 lines, compat wrapper)
- `kwave/solvers/simulation.py` (~500 lines)
- `kwave/solvers/serializer.py` (~200 lines)

**Impact:** ~5700 lines → ~1200 lines (-79%)

### Memory Layout and Indexing (v1.0.0)

By v1.0.0, the F→C migration from v0.6.1 should be complete. What remains:

- **C++ HDF5 serialization** (`cpp_simulation.py`) — F-order + 1-based. Stays forever (binary expects this).
- **MATLAB interop** (`simulate_from_dicts`) — converts F→C on input, C→F on output. Single conversion boundary.
- **kWaveArray** — rewrite `combine_sensor_data` and `get_distributed_source_signal` to use `np.where` (0-based) and C-order.
- **sensor.record_start_index** — require 0-based (deprecated 1-based in v0.6.1).

**Principle:** Python-native (C-order, 0-based) everywhere internally. MATLAB compatibility at two explicit boundaries: (1) `simulate_from_dicts` for k-wave-cupy interop, (2) `cpp_simulation._write_hdf5` for the C++ binary.

---

## Phase 3.5: v0.7.0 - CLI (`kwp`)

**Goal:** Command-line tool for running k-wave simulations from the terminal.

```bash
# Run a simulation from a config file
kwp run config.yaml

# Run with GPU
kwp run config.yaml --device gpu

# Generate HDF5 input for C++ binary
kwp prepare config.yaml --output sim_input.h5
```

**Features:**
- `kwp` CLI built on the v0.6 `kspaceFirstOrder()` API
- YAML/JSON config files for simulation parameters
- Supports both `python` and `cpp` backends
- `kwp run` — run simulation, save results
- `kwp prepare` — generate HDF5 input for offline C++ execution
- Installable as `pip install k-wave-python[cli]` (adds `click` dependency)

---

## Post-1.0: 2.x Vision (Performance & Scale)

**Goal:** Further optimization as the user base grows. Decisions driven by real profiling data and user demand.

| Feature | Technology | Benefit |
|---------|-----------|---------|
| Devito solver | Devito + mpi4py | Stencil-based solver for very large domains, MPI-enabled |
| Custom CUDA kernels | nanobind + cuFFT | Kernel fusion (~1.5-3x over CuPy), eliminate CuPy dependency |
| MPI parallelism | mpi4py + CuPy | Multi-node domain decomposition for large 3D grids |
| Multi-GPU (single node) | cupyx.distributed / NCCL | Split grid across GPUs within one machine |

**Decision criteria:**
- Profile real workloads on 1.0 to identify actual bottlenecks
- Measure CuPy overhead vs theoretical peak to decide if kernel fusion is worth it
- User feedback on what performance improvements matter most
- Evaluate whether CuPy's multi-platform support (NVIDIA + AMD ROCm) outweighs custom CUDA gains

**nanobind CUDA solver (if pursued):**

Standalone module in `kwave/cuda/` parallel to the native solver. Strangler fig pattern — mirrors `Simulation` class structure, replaces hot kernels one at a time:

1. Kernel library: Individual CUDA kernels (`spectral_diff`, `momentum_step`, `mass_step`, `eos_step`) callable from Python via nanobind `nb::ndarray`, zero-copy with CuPy via DLPack
2. Fused step: Entire `step()` as single C++ method
3. Full C++ loop: Time-stepping loop in C++, Python calls `run()` once

**Devito solver (if pursued):**

Optional extra: `pip install k-wave-python[devito]`

```python
result = kspaceFirstOrder(kgrid, medium, source, sensor,
                          backend="devito", use_mpi=True)
```

---

## Testing Strategy

**Existing (keep):**
- MATLAB reference tests via CI
- Multi-platform pytest (Windows, Ubuntu, macOS)
- Python 3.10-3.13 coverage

**Add:**
- `tests/test_native_solver.py` - native vs C++ parity
- `tests/test_unified_api.py` - new kwargs API
- `tests/test_deprecation_warnings.py` - verify warnings emitted
- `tests/test_backend_parity.py` - all backends produce same results

---

## Verification

```bash
# v0.5.0 - CI passes
uv run pytest tests/ -v

# v0.6.0 - Unified API works
uv run python -c "
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
import numpy as np

kgrid = kWaveGrid([64, 64], [0.1e-3, 0.1e-3])
kgrid.makeTime(1500, 0.3, 20e-6)
medium = kWaveMedium(sound_speed=1500)
source = kSource()
source.p0 = np.zeros((64, 64)); source.p0[32, 32] = 1
sensor = kSensor()
sensor.mask = np.ones((64, 64), dtype=bool)

result = kspaceFirstOrder(kgrid, medium, source, sensor, backend='native')
print('Success:', result['p'].shape)
"

# v0.6.0 - Deprecation warnings
uv run python -W error::DeprecationWarning -c "
from kwave.options.simulation_options import SimulationOptions
"  # Should warn

# v1.0.0 - Full test suite
uv run pytest tests/ -v
```

---

## Implementation Order

1. **Now:** Finalize master/main for v0.5.0
2. **Next:** Python solver + `kspaceFirstOrder()` API + deprecation for v0.6.0
3. **Then:** C-order migration of helpers/utils for v0.6.1
4. **Then:** Port remaining examples to new API for v0.6.2
5. **Then:** Axisymmetric support in new API for v0.6.3
6. **Then:** `kwp` CLI for v0.7.0
7. **Then:** Clean delete for v1.0.0
8. **Post-1.0:** Devito, nanobind/MPI based on profiling and user demand

---

## Critical Files

| File | Action | Version |
|------|--------|---------|
| `kwave/solvers/kspace_solver.py` | Finalize native solver | 0.6.0 |
| `kwave/solvers/native.py` | Finalize | 0.6.0 |
| `kwave/solvers/kwave_adapter.py` | Finalize | 0.6.0 |
| `kwave/kspaceFirstOrder.py` | Create unified entry point | 0.6.0 |
| `kwave/solvers/serializer.py` | Create HDF5 writer | 0.6.0 |
| `kwave/kspaceFirstOrder2D.py` | Simplify to wrapper | 0.6.0 |
| `kwave/kspaceFirstOrder3D.py` | Simplify to wrapper | 0.6.0 |
| `kwave/options/simulation_options.py` | Delete | 1.0.0 |
| `kwave/options/simulation_execution_options.py` | Delete | 1.0.0 |
| `kwave/kWaveSimulation.py` | Delete | 1.0.0 |
