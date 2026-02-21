# k-wave-python Release Strategy

## Overview

This release strategy brings the unified solver architecture to fruition while keeping k-wave-python pure and enabling AI agent access via FastMCP.

## Repository Structure

```
waltsims/
├── k-wave-python      # Core library + optional Devito solver
│   └── pip install k-wave-python[devito]  # Optional MPI/stencil deps
├── k-wave-mcp         # Local MCP server (runs anywhere, no cloud deps)
└── k-wave-cloud       # SaaS infrastructure (billing, workers, DO deployment)
```

**Rationale:**
- `k-wave-python`: Core simulation library with optional `[devito]` extra for HPC
- `k-wave-mcp`: Self-hostable MCP server, users can run locally or on their own infra
- `k-wave-cloud`: Commercial SaaS layer with billing, scales independently

---

## Version Roadmap

| Repo | Version | Milestone | Focus |
|------|---------|-----------|-------|
| k-wave-python | **0.5.0** | Native Solver GA | *(current rc1)* Finalize CuPy solver from k-wave-cupy |
| k-wave-python | **0.6.0** | Unified API | Single `kspaceFirstOrder()` with kwargs |
| k-wave-python | **0.7.0** | Deprecation | Warnings on old API, migration guide |
| k-wave-python | **1.0.0** | Stable Release | Remove deprecated code, clean API |
| k-wave-python | **1.1.0** | Devito Solver | `[devito]` extra for MPI/stencil |
| k-wave-mcp | **0.1.0** | MCP Server | Local FastMCP + optional cloud routing |
| k-wave-cloud | **0.1.0** | SaaS Beta | DO deployment, billing, job queue |

---

## Phase 1: v0.5.0 - Native Solver GA (Finalize RC)

**Goal:** Finalize 0.5.0rc1 by completing the CuPy solver migration from k-wave-cupy

**Source:** `k-wave-cupy/k-Wave/python/kWavePy.py` (469 lines, production-ready)

**Migration Tasks:**
1. Copy `Simulation` class from k-wave-cupy → `kwave/solvers/kspace_solver.py`
2. Verify adapter compatibility in `kwave/solvers/kwave_adapter.py`
3. Update `kwave/solvers/native.py` to use migrated Simulation
4. Port k-wave-cupy tests to k-wave-python
5. Validate against MATLAB references (<2e-15 relative error)

**Key Components to Migrate:**
- PML implementation (exponential absorption, staggered grid)
- K-space operators (kappa correction, gradient/divergence)
- Physics operators (absorption, dispersion, nonlinearity)
- Source operators (dirichlet/additive modes)
- Time stepping loop (leapfrog integration)

**After Migration - k-wave-cupy becomes thin wrapper:**
```matlab
% k-wave-cupy/k-Wave/kspaceFirstOrderPy.m
% Just imports k-wave-python and calls it
kWavePy = py.importlib.import_module('kwave.solvers.kspace_solver');
result = kWavePy.simulate_from_dicts(kgrid, medium, source, sensor);
```

**Tests to add:**
- `tests/test_native_solver.py` - parity with C++ backend
- `tests/test_cupy_backend.py` - GPU acceleration (when CUDA available)

---

## Phase 2: v0.6.0 - Unified API

**Goal:** Single entry point with kwargs, eliminate options classes

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
    backend: str = "cpp",  # "cpp" or "native"
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

---

## Phase 3: v0.7.0 - Deprecation

**Goal:** Clear migration path for users

**Add deprecation warnings:**
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

---

## Phase 4: v1.0.0 - Stable Release

**Goal:** Clean, minimal API

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

---

## Phase 5: v1.1.0 - k-wave-mcp (Local MCP Server)

**Goal:** Self-hostable MCP server for AI agents

**Create new repo:** `k-wave-mcp`

**Repo structure:**
```
k-wave-mcp/
├── pyproject.toml
├── Dockerfile              # Containerized server
├── src/kwave_mcp/
│   ├── server.py           # FastMCP server
│   ├── tools.py            # MCP tool definitions
│   └── data_handlers.py    # Array serialization
└── tests/
```

**Features:**
- Runs locally on user's machine
- No cloud dependencies for local mode
- Uses local k-wave-python installation
- Optional CuPy support for GPU
- **Optional k-wave-cloud integration**: Can dispatch heavy simulations to cloud

**Compute Routing:**
```
┌─────────────┐      ┌─────────────┐      ┌────────────────────┐
│  AI Agent   │─────▶│  k-wave-mcp │─────▶│  Local k-wave-py   │
│  (Claude)   │      │   (FastMCP) │      │  (NumPy/CuPy)      │
└─────────────┘      └──────┬──────┘      └────────────────────┘
                            │
                            │ (optional, for heavy workloads)
                            ▼
                     ┌─────────────┐      ┌────────────────────┐
                     │ k-wave-cloud│─────▶│  DO Droplets       │
                     │   (SaaS)    │      │  (GPU workers)     │
                     └─────────────┘      └────────────────────┘
```

**Usage:**
```python
# Local (default) - runs on user's machine
result = run_simulation(..., compute="local")

# Cloud - dispatches to k-wave-cloud for large sims
result = run_simulation(..., compute="cloud")

# Auto - routes based on grid size/complexity
result = run_simulation(..., compute="auto")
```

**MCP Tools:**
```python
@mcp.tool()
async def create_grid(dimensions, spacing, time_end=None, cfl=0.3): ...

@mcp.tool()
async def configure_medium(sound_speed, density=None, alpha_coeff=None): ...

@mcp.tool()
async def configure_source(source_type, mask, signal=None, p0=None): ...

@mcp.tool()
async def configure_sensor(mask, record=["p"]): ...

@mcp.tool()
async def run_simulation(grid_id, medium_id, source_id, sensor_id,
                         backend="native", use_gpu=False): ...

@mcp.tool()
async def get_sensor_data(result_id, field="p", format="summary"): ...
```

**Large array handling:**
- Return summaries by default (shape, min, max, mean)
- Support `format="base64"` for full array transfer
- Store results server-side, return IDs

---

## Phase 6: k-wave-cloud (SaaS Platform)

**Goal:** Commercial simulation-as-a-service platform

**Create new repo:** `k-wave-cloud`

**Repo structure:**
```
k-wave-cloud/
├── pyproject.toml
├── docker-compose.yml      # Full stack
├── src/kwave_cloud/
│   ├── api/                # REST API for job submission
│   ├── worker/             # Celery workers
│   ├── billing/            # Usage tracking + Stripe
│   └── storage/            # S3 result management
├── deploy/
│   ├── terraform/          # DO infrastructure
│   └── kubernetes/         # Optional K8s scale
└── tests/
```

**Simulation-as-a-Service Architecture:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│   AI Agent      │────▶│   MCP Server    │────▶│  DigitalOcean Droplets  │
│ (Claude, etc.)  │     │   (FastMCP)     │     │  (Run simulations)      │
└─────────────────┘     └─────────────────┘     └─────────────────────────┘
                              │                          │
                              ▼                          ▼
                        ┌─────────────┐          ┌──────────────┐
                        │  Billing /  │          │ Results      │
                        │  Usage API  │          │ Storage (S3) │
                        └─────────────┘          └──────────────┘
```

**Components:**
- **MCP Server:** Lightweight FastMCP service handling tool calls
- **Compute Droplets:** Spin up on-demand for simulations (CPU or GPU)
- **Job Queue:** Redis/Celery for async simulation dispatch
- **Results Storage:** S3-compatible storage for large sensor data
- **Billing:** Track compute time, charge per simulation or per minute

**Droplet Tiers:**
- Basic CPU: Small 2D sims, quick tests (~$0.01/sim)
- Standard CPU: Medium 3D sims (~$0.05/sim)
- GPU Droplet: Large 3D sims with CuPy (~$0.10/sim)

**Workflow:**
1. Agent calls `run_simulation()` via MCP
2. MCP server queues job + estimates cost
3. Worker droplet pulls job, runs k-wave-python
4. Results stored in S3, job marked complete
5. Agent retrieves results via `get_sensor_data()`

---

## Phase 7: k-wave-python v1.1.0 - Devito Solver

**Goal:** MPI-enabled stencil solvers for large domains (optional extra)

**Installation:**
```bash
pip install k-wave-python[devito]  # Adds devito, mpi4py, etc.
```

**Add to pyproject.toml:**
```toml
[project.optional-dependencies]
devito = ["devito>=4.8", "mpi4py>=3.1"]
```

**New files:**
- `kwave/solvers/devito_solver.py` - Stencil-based solver using Devito
- Add `Backend.DEVITO` to enum

**Usage:**
```python
result = kspaceFirstOrder(kgrid, medium, source, sensor,
                          backend="devito",  # Uses Devito stencil solver
                          use_mpi=True)      # Enable MPI distribution
```

**Interface:**
- Same input objects (kWaveGrid, kWaveMedium, kSource, kSensor)
- Same output format (dict with 'p', 'p_final')
- Same kwargs API as other backends

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
# v0.5.0 - Native solver works
uv run pytest tests/test_native_solver.py -v

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

# v0.7.0 - Deprecation warnings
uv run python -W error::DeprecationWarning -c "
from kwave.options.simulation_options import SimulationOptions
"  # Should warn

# v1.0.0 - Full test suite
uv run pytest tests/ -v
```

---

## Implementation Order

1. **Now:** Finalize v0.5.0 (complete native solver, release from rc1)
2. **Next:** Create `kwave/kspaceFirstOrder.py` for v0.6.0
3. **Parallel:** Set up k-wave-mcp repo skeleton (local MCP server)
4. **After v0.6.0:** Add deprecation warnings for v0.7.0
5. **After deprecation period:** Clean delete for v1.0.0
6. **Post-1.0:** Release k-wave-mcp v0.1.0
7. **Post-1.0:** Add `[devito]` extra for k-wave-python v1.1.0
8. **When ready:** Set up k-wave-cloud repo (SaaS infrastructure)

---

## Critical Files

| File | Action | Version |
|------|--------|---------|
| `kwave/solvers/kspace_solver.py` | Finalize (from k-wave-cupy) | 0.5.0 |
| `kwave/solvers/native.py` | Finalize | 0.5.0 |
| `kwave/kspaceFirstOrder.py` | Create unified entry point | 0.6.0 |
| `kwave/solvers/serializer.py` | Create HDF5 writer | 0.6.0 |
| `kwave/kspaceFirstOrder2D.py` | Simplify to wrapper | 0.6.0 |
| `kwave/kspaceFirstOrder3D.py` | Simplify to wrapper | 0.6.0 |
| `kwave/options/simulation_options.py` | Delete | 1.0.0 |
| `kwave/options/simulation_execution_options.py` | Delete | 1.0.0 |
| `kwave/kWaveSimulation.py` | Delete | 1.0.0 |
