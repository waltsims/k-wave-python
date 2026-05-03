# Issue #664: NaN output with `alpha_power` near 1.0 on C++ backend

This document briefs a fresh Claude session continuing work on this branch
(`fix-alpha-mode-near-unity`). Read `plans/release-strategy.md` for the
broader v0.6.x context.

## The bug

User report (https://github.com/waltsims/k-wave-python/issues/664):
`kspaceFirstOrder2D` output contains NaNs when `medium.alpha_power` is in
the range 0.95 to 1.03. The same simulation in MATLAB k-Wave runs cleanly.

Important: the user's repro uses `is_gpu_simulation=True` → C++ CUDA backend
via the legacy `kspaceFirstOrder2D` entry. So this is **not** a bug in the
Python solver's `tan(pi*y/2)` term. **MATLAB doesn't NaN, so the C++ binary
itself is fine.** The defect is in the Python-side HDF5 input we hand to
the binary — some byte we serialize differs from what MATLAB produces.

## Branch history (why this is a fresh start)

The branch went through several attempts that didn't work:

- `bcef90d` — replaced `getattr(self.medium, "alpha_mode", None)` with
  direct attribute access. Broke MATLAB interop (`SimpleNamespace` from
  `simulate_from_dicts` doesn't carry `alpha_mode`).
- `73b5584` — reverted `bcef90d` (commit message: *"moving to separate PR"*).
- `fb0c0e3` — reverted the revert (looks accidental). Interop broken again.
- `1dfb845`, `c5b35b7`, `ba1adea` — changed `p_source_flag` to write the
  signal length instead of a boolean. **This may be the actual fix** for
  #664 but was bundled with unrelated work and not validated.
- A near-unity validation guard in `validate_medium` that just blocked
  the user's exact config without addressing the underlying serialization
  bug.

On 2026-05-03 the branch was reset to match master via
`git read-tree --reset -u master` + a new commit (commit `b888e3a`). No
force-push — the old commits stay in history. The PR (#704) now starts
from a clean slate.

## Current state of the branch

- Tree equals master.
- One new test file: `tests/test_issue_664_alpha_power_near_unity.py`
  (commit `0f9dc04`). Smoke test that currently runs the **OMP** binary
  with `is_gpu_simulation=False` and asserts no NaN for `alpha_power` in
  {0.97, 1.01, 1.03}. Skips cleanly when the binary can't launch.
- No fix yet. The smoke test is expected to fail when the binary is
  reachable.

## Why we moved to a cloud machine

Local development was on an Intel Mac (x86_64) but the only bundled C++
binary in `kwave/bin/darwin/` is `arm64`. Architecture mismatch — we
can't run the binary at all on the dev laptop, even with brew deps.

The user's repro uses GPU (`is_gpu_simulation=True`), but the suspected
bug is in the **Python-side HDF5 serializer**
(`save_to_disk_func.py`) — the same code path feeds both the OMP and
CUDA binaries. So a **Linux x86_64 CPU droplet** is sufficient for the
diagnostic and fix work; we use OMP there. GPU only matters for the
final acceptance test reproducing the user's literal scenario.

## What to do next

### Step 1 — confirm the bug reproduces on the droplet (OMP)

The smoke test already targets OMP (`is_gpu_simulation=False`). Just
run it:

```bash
git checkout fix-alpha-mode-near-unity
uv sync --extra all
uv run pytest tests/test_issue_664_alpha_power_near_unity.py -v
```

All three parametrized cases should fail with NaNs. If they pass, two
possibilities:

- **Minimization went too far.** The user's full scenario uses
  heterogeneous sound speed, density variation, and a ring sensor mask;
  any of those could be the trigger. Walk back toward the verbatim
  repro from the issue body until NaNs appear, then re-minimize.
- **The bug is CUDA-specific.** Less likely (shared serialization
  path), but possible if the CUDA dispatch reads a field the OMP
  dispatch ignores. Park this for the GPU acceptance test (Step 6).

### Step 2 — A/B legacy vs modern API

Add a parallel test that runs the same scenario through the **modern**
entry point:

```python
from kwave.kspaceFirstOrder import kspaceFirstOrder
result = kspaceFirstOrder(kgrid, medium, source, sensor,
                          backend="cpp", device="cpu", pml_inside=True)
```

Two outcomes matter:

- **Modern is NaN-free, legacy NaNs:** the modern serializer
  (`kwave/solvers/cpp_simulation.py`) writes the bytes correctly; the
  legacy serializer (`kwave/kWaveSimulation_helper/save_to_disk_func.py`)
  doesn't. The fix template is "make legacy match modern." We can also
  recommend the modern API to the issue reporter as an immediate
  workaround while the legacy fix lands.
- **Both NaN:** shared bug deeper in `kwave/utils/io.py` or in the data
  the simulation builds. Diagnose with HDF5 diff (Step 3) regardless.

Either way, the legacy path needs fixing — the user-facing API in 0.6.x
still includes `kspaceFirstOrder2D`/`3D` (deprecated but functional).

### Step 3 — three-tier testing strategy

"No NaN" is a low bar. The binary could read garbage and write silent
garbage. We need three layers:

1. **Smoke test** (already present). *"Is the bug present?"* Runs the
   binary, asserts no NaN. Cheap, runs in CI.
2. **HDF5-input diff test** (to add). *"Which byte is wrong?"* Python
   writes its HDF5 input via `save_to_disk_exit=True`; MATLAB writes the
   same scenario via `kspaceFirstOrder2D(..., 'SaveToDisk', filename)`;
   both files are diffed field-by-field with `h5py`. The first differing
   field pinpoints the bug. Doesn't need the binary — works everywhere.
3. **Output parity test** (to add). *"Are the values correct?"* Once the
   smoke test stops NaN'ing, compare numerical output to a MATLAB-run
   reference (`sensor_data.p` from a MATLAB simulation of the same
   scenario, captured to `.mat`). Use the same tolerance you'd use for
   any matlab-parity test in this repo (machine precision on
   well-specified inputs; loosen if needed).

### Step 4 — drop the MATLAB reference into the existing CI flow

Drop `collect_issue_664.m` into
`tests/matlab_test_data_collectors/matlab_collectors/`. CI's
`pytest.yml` job `collect_references` (lines 6–67) automatically picks
up every `.m` file in that directory, runs them on a MATLAB-licensed
GitHub runner, caches the output as `collectedValues.tar.gz`, and
exposes it to every pytest matrix job. **You don't need MATLAB
installed locally.** Push the `.m` file → CI generates the `.h5`
reference → diff test runs.

The collector should write *both* the HDF5 input file (for Step 3.2)
and the simulation output (`sensor_data.p`, for Step 3.3). Pattern
follows `collect_example_parity_2D.m`.

### Step 5 — investigate the failing field

Top suspects in priority order:

- `kwave/kWaveSimulation_helper/save_to_disk_func.py` — legacy HDF5
  writer; this is the user's path.
- `kwave/kWaveSimulation.py:274` — `source_p` property (already returns
  signal length, not boolean — cross-check vs MATLAB's value for this
  field name in the HDF5).
- `kwave/utils/io.py` — low-level HDF5 helpers; check dtype, shape, and
  Fortran-vs-C-order conversions.
- `kwave/solvers/cpp_simulation.py:201` — modern-API writer. The earlier
  branch's `p_source_flag` "fix" was here, **not** in the legacy path,
  which is one reason that attempt didn't close the issue. If A/B (Step
  2) shows modern works and legacy doesn't, the diff between these two
  files is the highest-signal place to start.

### Step 6 — acceptance test from the user's verbatim repro

Once unit tests pass, paste the user's full code from the issue body
(Shepp-Logan phantom, 512-element ring transducer, gausspulse) into a
script and run it. Assert no NaN. Ideally compare a few sensor traces
against MATLAB. Save it as `tests/test_issue_664_user_repro.py` with
`@pytest.mark.slow` so it doesn't run in the default suite but is
reproducible.

The user's original code uses `is_gpu_simulation=True`. Since the CPU
droplet has no GPU, run this acceptance test in two places:

1. **CPU droplet, OMP**: change `is_gpu_simulation=False` to confirm
   the fix works on the OMP dispatch.
2. **GPU machine** (when available — DigitalOcean, Lambda, RunPod,
   etc.): run the user's exact code unchanged (`is_gpu_simulation=True`)
   to close the loop on their literal scenario.

This is the final "yes, the user's actual problem is fixed" gate. Both
must pass.

## Things to avoid

- Don't add a Python-solver `tan(pi*y/2)` validation guard. The bug is
  in C++ serialization, not in Python's dispersion math. A guard would
  block the user's config without fixing anything.
- Don't toggle `getattr(self.medium, ...)` ↔ direct attribute access
  without thinking about MATLAB interop (`SimpleNamespace` from
  `simulate_from_dicts`). The flip-flop already happened twice on this
  branch.
- Don't force-push this branch. The user prefers history-preserving
  resets (a forward commit whose tree matches master, as already done).
- Don't commit large `.h5` reference files directly. Use the existing
  `matlab_collectors/collectedValues/` cache pattern; CI picks them up.

## Things to avoid

- Don't add a Python-solver `tan(pi*y/2)` validation guard. The bug is
  in C++ serialization, not in Python's dispersion math. A guard would
  block the user's config without fixing anything.
- Don't toggle `getattr(self.medium, ...)` ↔ direct attribute access
  without thinking about MATLAB interop (`SimpleNamespace` from
  `simulate_from_dicts`). The flip-flop already happened twice on this
  branch.
- Don't force-push this branch. The user prefers history-preserving
  resets (a forward commit whose tree matches master, as already done).

## Reference

- Issue: https://github.com/waltsims/k-wave-python/issues/664
- PR: https://github.com/waltsims/k-wave-python/pull/704
- Sibling release plan: `plans/release-strategy.md`
- Sibling fix already shipped today: PR #705 (`cleanup-alpha-mode-enum` —
  AlphaMode enum + post-construction string normalization).
