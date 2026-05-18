# k-wave-python Release Strategy

## Overview

This release strategy brings the unified solver architecture to fruition.

## Version Roadmap

| Version | Milestone | Focus |
|---------|-----------|-------|
| **0.5.0** | Finalize master/main | Stabilize current codebase |
| **0.6.0** | Python Solver + Unified API + Deprecation | Python solver, `kspaceFirstOrder()` kwargs, Future warnings |
| **0.6.1** | C-order + Examples + Docs | C-order migration, 29 examples ported, 47 parity tests, docs cleanup |
| **0.6.2** | Binary refresh (sm_120 / Blackwell) | Bump URL pins to upstream v1.4.0 binaries with NVIDIA Blackwell support; closes #656, #622 |
| **0.6.3** | Tier 2 Features + Examples | Time-reversal, rect sensors, sound_speed_ref, port Tier 2 examples |
| **0.6.4** | Axisymmetric Support | Axisymmetric solver in new API, port AS examples |
| **0.6.5** | Broader Darwin coverage | Universal2 (arm64 + x86_64) OpenMP binary; restores Intel Mac support |
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

## Phase 2: v0.6.0 - Native Solver + Unified API + Future

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

### 2c: Future Warnings

**Add Future warnings on old API:**
```python
warnings.warn(
    "SimulationOptions is deprecated. Use kspaceFirstOrder() kwargs instead. "
    "See https://waltersimson.com/k-wave-python/migration",
    FutureWarning
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
- `tests/test_Future_warnings.py` - verify warnings emitted

---

## v0.6.x Point Releases

### v0.6.1 — C-order + Examples + Docs (released 2026-03-29)

Combined release: C-order migration, example restructure, parity tests, docs cleanup.

**C-order migration:**
- `kspace_solver.py`: F-order → C-order internally
- `cpp_simulation.py`: `_fix_output_order()` for C++ backend compatibility
- `kspaceFirstOrder.py`: `_reshape_sensor_to_grid()` for full-grid sensors
- F-order kept at boundaries: `matlab.py`, `cpp_simulation._write_hdf5()`, legacy API

**Example restructure:**
- 29 Tier 1 examples ported to `setup()/run()/__main__` pattern
- Flattened `examples/ported/` → `examples/`, dropped `example_` prefix
- Old subdirectory examples in `examples/legacy/`
- `run-examples.yml` triggers on PRs touching `examples/`, excludes `legacy/`

**Testing:**
- 47 parity tests passing (machine precision), 6 skipped (missing refs)
- 3D PML fix: MATLAB defaults to pml_size=10 for 3D
- Table-driven parametrized test framework

**Docs & infra cleanup:**
- README: Python-first framing, both backends described
- Dev docs: simplified setup with `uv sync`, removed outdated sections
- macOS C++ hint in executor.py (scoped to linker errors)
- Deleted: Makefile, Dockerfile, run_examples.py, notebook pipeline, dead CI workflows

### v0.6.2 — Binary refresh (sm_120 / Blackwell)

**Goal:** Ship `kwave/__init__.py` URL pins that point at fixed upstream binaries with sm_120 / Blackwell support, the macOS HDF5 ABI refresh, and the macOS fast-math fix.

**What actually shipped (retrospective):** v1.4.0 had to be revoked due to packaging regressions and a numerical bug. v0.6.2 ships against **v1.4.1** instead. Summary:

- **Linux binaries** — `v1.4.1` rebuilt with static CUDA + cufft + FFTW + libstdc++ on ubuntu-22.04 (glibc 2.35 floor), restoring the plug-and-play property the legacy Makefile build had. Permanent regression guard added via `scripts/check-linux-binary-deps.sh` in the unified repo. Fixed in [kspacefirstorder-unified#15](https://github.com/waltsims/kspacefirstorder-unified/issues/15) and [#16](https://github.com/waltsims/kspacefirstorder-unified/pull/16).
- **macOS OMP** — `v1.4.1` picks up the fast-math fix ([k-wave-omp-darwin#4](https://github.com/waltsims/k-wave-omp-darwin/pull/4)) that prevents NaNs in absorbing-media simulations on arm64. ABI refresh (`libhdf5.320.dylib`) preserved from v1.4.0; closes [#661](https://github.com/waltsims/k-wave-python/issues/661).
- **Windows OMP + CUDA** — both pinned to **v1.3.0** for this release. v1.4.x windows builds switched compiler/OpenMP/FFT/CUDA-runtime stacks but neither bundles its runtime DLLs. v1.3.0 binaries are self-contained with the Intel-era DLL bundle (shipped via `WINDOWS_DLLS` with the OMP request and used by both `.exe` files since they share `kwave/bin/windows/`). OMP DLL bundling fix landed in [kspacefirstorder-unified#14](https://github.com/waltsims/kspacefirstorder-unified/issues/14) (awaiting production validation); CUDA bundling tracked in [kspacefirstorder-unified#17](https://github.com/waltsims/kspacefirstorder-unified/issues/17). Pin gets flipped in v0.6.3 once both windows flavors are validated end-to-end.
- **Chmod fix** — independent fix for `download_binaries` not setting the exec bit on Linux/macOS, surfaced during v0.6.2 validation on Colab ([#741](https://github.com/waltsims/k-wave-python/pull/741)).
- **Intel Mac guard** — `kwave/__init__.py` now emits a `RuntimeWarning` and skips the darwin OMP download on `darwin` x86_64, since the v1.4.x darwin binary is arm64-only. Universal2 coverage remains tracked in v0.6.5.

**Closes:** [#622](https://github.com/waltsims/k-wave-python/issues/622), [#656](https://github.com/waltsims/k-wave-python/issues/656), [#661](https://github.com/waltsims/k-wave-python/issues/661), [#738](https://github.com/waltsims/k-wave-python/issues/738), [#740](https://github.com/waltsims/k-wave-python/issues/740).

**Out of scope:** Darwin x86_64 (Intel Mac) coverage — v1.4.x OMP-darwin is arm64-only. Tracked in v0.6.5.

---

### v0.6.3 — Tier 2 Features + Examples + Windows binary pin flips

**Goal:** Add solver features needed by Tier 2 examples, port those examples, and flip the Windows binary pins to v1.4.x once both flavors are validated.

**Windows binary pin flips (carried over from v0.6.2):**
- **Validate `v1.4.1` windows-OMP** end-to-end on a Windows machine (smoke + small simulation). Bundle landed in [kspacefirstorder-unified#14](https://github.com/waltsims/kspacefirstorder-unified/issues/14); CI smoke passes; runtime sim not yet confirmed.
- **Fix `v1.4.x` windows-CUDA DLL packaging** in [kspacefirstorder-unified#17](https://github.com/waltsims/kspacefirstorder-unified/issues/17) — same DLL-bundling pattern as windows-omp, plus CUDA runtime DLLs (`cudart64_*.dll`, `cufft64_*.dll`) from `$env:CUDA_PATH\bin`.
- Once both validated: drop `WINDOWS_OMP_VERSION` + `WINDOWS_CUDA_VERSION` pins in `kwave/__init__.py` (both use `BINARY_VERSION`). Likely also: refactor `WINDOWS_DLLS` into per-architecture lists since OMP and CUDA need slightly different MSVC redist + runtime deps.

**Features to add:**
- **Time-reversal reconstruction** — needed by PR examples (`pr_2D_TR_*`, `pr_3D_TR_*`)
- **Rectangular/corner sensor masks** — needed by 5 examples
- **`sound_speed_ref`** — needed by `tvsp_slit_diffraction`
- **Frequency-response sensor** — needed by `ivp_sensor_frequency_response`
- **Directional sensor** — needed by `sd_sensor_directivity_2D`

**Also:**
- CuPy GPU validation on DigitalOcean (29 examples ready)
- Real-world validation (published paper simulations)
- Consolidate test infrastructure (`tests/integration/` + `test_example_parity.py`)

**Simplification targets:**
- Delete `examples/legacy/` once all examples are ported or confirmed obsolete
- Remove unused MATLAB collector infrastructure if parity tests replace it
- Audit `kWaveSimulation_helper/` — delete helpers superseded by `kspaceFirstOrder()`

### v0.6.4 — Axisymmetric Support

Axisymmetric = dimensionality reduction (3D→2D or 2D→1D). Not a separate solver — wrapper around `kspaceFirstOrder()` with radial symmetry terms added to the wave equation.

---

### v0.6.5 — Broader Darwin coverage (Intel + Apple Silicon)

**Goal:** Restore Intel Mac support for the OpenMP binary. The v1.4.0 release shipped a Mach-O `arm64`-only OMP binary (`kspaceFirstOrder-OMP` in `k-wave-omp-darwin` v1.4.0), which excludes every Intel Mac. Pre-2020 hardware is still the majority of macOS users in academic settings, and even on newer Apple Silicon machines x86_64 coverage is useful for Rosetta-only third-party stacks.

**Decision: ship a universal2 binary (arm64 + x86_64) — not two separate per-arch binaries.** Doubles the file size (~300 KB → ~600 KB, negligible) and avoids a per-arch download-selection step in `kwave/__init__.py`. The build flow already runs on `macos-latest` (Apple Silicon GitHub runners); switching to a universal build is a one-line CMake change plus a `libomp` install for both arches.

**Tasks:**
1. **Build path:** Add `-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"` to the macOS leg of `ci-multi-platform.yml` in the unified repo, and ensure the Homebrew install step pulls a `libomp` that has both slices (Homebrew bottles are per-arch — may need `lipo` to combine, or build OMP from source).
2. **Verify:** `lipo -info kspaceFirstOrder-OMP` should report both `x86_64 arm64`. Smoke test on both an Apple Silicon Mac (native) and an Intel Mac (or `arch -x86_64` on AS via Rosetta).
3. **Release:** Tag `v1.4.1` (or whatever the next binary release is) on `k-wave-omp-darwin` with the universal binary, then bump `BINARY_VERSION` in `kwave/__init__.py`.
4. **No Python code changes** are needed — the existing `PLATFORM == "darwin"` branch in `kwave/__init__.py` doesn't distinguish architecture; a universal binary is consumed exactly the same way.

**Future Mac topics (post-1.0, separate releases):**
- **CUDA on macOS:** Not feasible — Apple dropped NVIDIA driver support after macOS 10.13. The `darwin/cuda` slot in `URL_DICT` will stay empty indefinitely.
- **Metal / MPS backend for the Python solver:** Would unlock Apple Silicon GPU acceleration for the `backend="python"` path via something like `mlx` or PyTorch's MPS. Scope is the Python solver, not the C++ binary — best fit is the v2.x performance phase.

---

## Binary distribution maintenance (no fixed version)

Work on the upstream binary side that doesn't need its own k-wave-python release — it gets picked up by whatever version bump happens to be in flight.

### Mirror consolidation (kspacefirstorder-unified#13)

Collapse the 5 per-platform mirror repos (`kspaceFirstOrder-{CUDA,OMP}-{linux,windows}` + `k-wave-omp-darwin`) into the unified repo. Each cross-cutting change currently lands ~2-3× (proven by v1.4.0 / v1.4.1 taking ~12 PRs across repos); the consolidation retires that overhead. Cross-linked from k-wave-python#738 — flip `URL_DICT` to point at unified once the consolidation lands. Best fit for v1.5.0 or alongside whichever release first benefits.

### Static-link Windows binaries (v1.5 follow-up)

The v0.6.2 ship used the **bundle** approach for Windows OMP (DLLs alongside the .exe) because static-linking on Windows MSVC is more complex (`/MT` for CRT, vcpkg static port for FFTW, `VCOMP140` static linking is awkward, may need LLVM OpenMP). The Linux fix in unified#15 already uses static linking; Windows should follow once the broader v1.5 static-linking discipline is in scope. Track alongside mirror consolidation since both are "tidy up the binary pipeline" work.

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

**Current (v0.6.1):**
- Multi-platform pytest (Windows, Ubuntu, macOS) × Python 3.10-3.13 (12-job matrix)
- MATLAB reference tests via CI (66 collectors → `.mat` fixtures)
- `test_example_parity.py` — 47 table-driven parity tests against MATLAB refs
- `test_native_solver.py` — 33 tests for Python backend
- Weekly example runner (`run-examples.yml`)
- `matlab_parity` marker for selective runs

**v0.6.2 (Tier 2 features):**
- Parity tests for time-reversal, rect sensors, etc.
- CuPy GPU validation

**v0.6.3 (axisymmetric):**
- MATLAB reference tests for `at_circular_piston_AS`, `at_focused_bowl_AS`

---

## Implementation Order

1. ~~v0.5.0~~ ✅ Stabilize master
2. ~~v0.6.0~~ ✅ Python solver + unified API + deprecations
3. ~~v0.6.1~~ ✅ C-order + examples + docs cleanup
4. **Next:** v0.6.2 — Binary refresh (sm_120 / Blackwell)
5. **Then:** v0.6.3 — Tier 2 features + examples
6. **Then:** v0.6.4 — Axisymmetric support
7. **Then:** v0.6.5 — Broader Darwin coverage (universal2 OMP)
8. **Then:** v0.7.0 — CLI (`kwp`)
9. **Then:** v1.0.0 — Clean release (delete deprecated code)
10. **Post-1.0:** Performance & scale based on profiling
