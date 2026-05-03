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
  (commit `0f9dc04`). Smoke test that runs the C++ OMP binary and asserts
  no NaN for `alpha_power` in {0.97, 1.01, 1.03}. Skips cleanly when the
  binary can't launch (missing libs, wrong arch).
- No fix yet. The smoke test is expected to fail in CI on master.

## Why we moved to a cloud machine

Local development was on an Intel Mac (x86_64) but the only bundled C++
binary in `kwave/bin/darwin/` is `arm64`. Architecture mismatch — installing
brew deps doesn't help. We can't reproduce the bug locally.

Linux droplet (with NVIDIA GPU) gives:
- x86_64 OMP binary that runs natively.
- CUDA binary for matching the user's GPU repro literally.
- A real iteration loop.

## What to do next

### Step 1 — confirm the bug reproduces on the droplet

```bash
git checkout fix-alpha-mode-near-unity
uv sync --extra all
uv run pytest tests/test_issue_664_alpha_power_near_unity.py -v
```

All three parametrized cases should fail with NaNs. If they pass, the bug
either depends on a heterogeneous medium / different sensor geometry, or
the minimization went too far. Re-add complexity from the user's original
report (the issue body has the full code) until it reproduces.

### Step 2 — add the HDF5-diff diagnostic test

The smoke test confirms the bug exists but doesn't tell us which byte is
wrong. The diagnostic approach:

1. Have Python write its HDF5 input via `save_to_disk_exit=True`.
2. Have MATLAB write its HDF5 input via `kspaceFirstOrder2D(..., 'SaveToDisk', filename)`.
3. Diff the two HDF5 files field by field. The first differing field is the bug.

Where to put pieces:
- MATLAB collector: `tests/matlab_test_data_collectors/matlab_collectors/collect_issue_664.m`.
  Pattern follows existing `collect_example_parity_2D.m`. CI runs all
  collectors via `run_all_collectors.m` and caches outputs.
- Python test: extend `tests/test_issue_664_alpha_power_near_unity.py`
  with `test_hdf5_input_matches_matlab` that loads both files with `h5py`
  and compares. Use `np.testing.assert_array_equal` per field; the first
  failure points at the bug.

### Step 3 — investigate the failing field

Top suspects (look here first before grep'ing the whole codebase):

- `kwave/kWaveSimulation_helper/save_to_disk_func.py` — main HDF5 writer
  used by legacy `kspaceFirstOrder2D` (the user's path).
- `kwave/kWaveSimulation.py:274` — `source_p` property already returns
  signal *length*, not boolean. Cross-check vs MATLAB's value.
- `kwave/utils/io.py` — low-level HDF5 write helpers; check dtype/shape
  conversions.
- `kwave/solvers/cpp_simulation.py:201` — new-API writer; the previous
  branch's `p_source_flag` "fix" was here, not in the legacy path.

### Step 4 — fix and verify

Once the diff test pinpoints the field, fix the serializer. Both tests
(smoke + diff) should go green together. Verify the user's full repro
from the issue body (Shepp-Logan phantom, ring transducer) also runs
NaN-free.

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
