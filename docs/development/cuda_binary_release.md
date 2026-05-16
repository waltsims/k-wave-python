# CUDA Binary Release Runbook (sm_120 / Blackwell)

This runbook covers the end-to-end process for publishing a new set of pre-compiled CUDA binaries that include support for new GPU architectures. The motivating case is **NVIDIA Blackwell (sm_120, RTX 50xx series)** support, which is currently blocking users on:

- [#656](https://github.com/waltsims/k-wave-python/issues/656) — RTX 5070 Ti / sm_120 segfaults (canonical issue, with full repro and the working build recipe from @aconesac)
- [#622](https://github.com/waltsims/k-wave-python/issues/622) — "Can not use kspaceFirstOrder-CUDA" (same root cause, confirmed by @faberno [in this comment](https://github.com/waltsims/k-wave-python/issues/622#issuecomment-2273106886))

Both issues will close automatically once a v1.4.0 release is published per the steps below.

## Architecture

The CUDA binaries are produced by a multi-repo build pipeline:

```
kspacefirstorder-unified  ─── multi-platform CI ───┐
  ├─ repos/kspaceFirstOrder-cuda-linux  (submodule)│ matrices on CUDA 12.2 + 13.0
  ├─ repos/kspaceFirstOrder-cuda-windows (submodule)│ uploads artifacts per leg
  ├─ repos/kspaceFirstOrder-openmp-linux (submodule)│
  ├─ repos/kspaceFirstOrder-openmp-windows         │
  └─ repos/kspaceFirstOrder-openmp-darwin          │
                                                   │
   Each individual repo (CUDA-linux, CUDA-windows, OMP-*)
   ships a release tag containing the binary, which k-wave-python
   then downloads from a URL pinned in `kwave/__init__.py`.
```

## Release pipeline

### Step 1 — Bump `CUDA_ARCH` in both CUDA repos

Add `sm_100`, `sm_120`, and the forward-compatible PTX `compute_120` entry to `Makefile` on the `cuda-12-support` branch of each repo.

- **Linux**: [waltsims/kspaceFirstOrder-CUDA-linux#5](https://github.com/waltsims/kspaceFirstOrder-CUDA-linux/pull/5) (opened against `cuda-12-support`)
- **Windows**: [waltsims/kspaceFirstOrder-CUDA-windows#1](https://github.com/waltsims/kspaceFirstOrder-CUDA-windows/pull/1) — needs minor style cleanup per [review](https://github.com/waltsims/kspaceFirstOrder-CUDA-windows/pull/1#pullrequestreview-) before merge

Note that CUDA 13 dropped support for `sm_50` / `sm_60` / older. The current `cuda-12-support` branches already omit those, so no extra deletion is needed.

The `sm_103` (Blackwell B300 variant) entry is also valid under CUDA 13 but not added by either PR — it's a niche datacenter SKU, optional follow-up.

### Step 2 — Bump submodule SHAs in `kspacefirstorder-unified`

After both PRs land:

```bash
git clone --recurse-submodules https://github.com/waltsims/kspacefirstorder-unified
cd kspacefirstorder-unified
cd repos/kspaceFirstOrder-cuda-linux && git fetch && git checkout origin/cuda-12-support && cd -
cd repos/kspaceFirstOrder-cuda-windows && git fetch && git checkout origin/cuda-12-support && cd -
git add repos/kspaceFirstOrder-cuda-linux repos/kspaceFirstOrder-cuda-windows
git commit -m "Bump CUDA submodules to sm_120-capable Makefile"
git push
```

CI will then build all 5 binaries across the matrix (CUDA 12.2 + 13.0 for the CUDA legs). Only the CUDA 13.0.0 artifacts will have sm_120 support — that's the one we ship.

CI matrix is defined in `.github/workflows/ci-multi-platform.yml` and uploads per-leg artifacts named like:
- `kspaceFirstOrder-cuda-linux-13.0.0`
- `kspaceFirstOrder-cuda-windows-13.0.0`
- `kspaceFirstOrder-openmp-linux-*`, etc.

### Step 3 — Tag releases on the individual binary repos

Pull the CUDA 13.0.0 artifacts off the unified CI run, then on each binary repo:

```bash
# kspaceFirstOrder-CUDA-linux
gh release create v1.4.0 ./kspaceFirstOrder-CUDA \
  --title "v1.4.0: Blackwell (sm_120) support" \
  --notes "Adds sm_100 (Blackwell datacenter) and sm_120 (consumer RTX 50xx) compute capabilities, plus PTX compute_120 for forward compatibility. Requires CUDA 13 runtime to consume the new code paths."

# kspaceFirstOrder-CUDA-windows
gh release create v1.4.0 ./kspaceFirstOrder-CUDA.exe ./*.dll \
  --title "v1.4.0: Blackwell (sm_120) support" \
  --notes "(same as Linux)"
```

For the OMP repos, refresh against the latest HDF5 to also resolve [#661](https://github.com/waltsims/k-wave-python/issues/661) (macOS HDF5 ABI mismatch) and similar issues:

```bash
# k-wave-omp-darwin, kspaceFirstOrder-OMP-linux, kspaceFirstOrder-OMP-windows
gh release create v0.4.0 ./kspaceFirstOrder-OMP \
  --title "v0.4.0: HDF5 ABI refresh" \
  --notes "Built against current Homebrew HDF5 to resolve libhdf5.310 vs .320 ABI mismatch on macOS (#661)."
```

### Step 4 — Bump version pins in k-wave-python

Edit `kwave/__init__.py`:

```python
URL_DICT = {
    "linux": {
        "cuda": [URL_BASE + f"kspaceFirstOrder-CUDA-{PLATFORM}/releases/download/v1.4.0/{EXECUTABLE_PREFIX}CUDA"],
        "omp":  [URL_BASE + f"kspaceFirstOrder-OMP-{PLATFORM}/releases/download/v0.4.0/{EXECUTABLE_PREFIX}OMP"],
    },
    "darwin": {
        "cuda": [],
        "omp":  [URL_BASE + f"k-wave-omp-{PLATFORM}/releases/download/v0.4.0/{EXECUTABLE_PREFIX}OMP"],
    },
    ...
}
```

Bump `BINARY_VERSION` if defined elsewhere. Open a PR. CI will re-download against the new URLs.

### Step 5 — Verify and close issues

After the new k-wave-python release is out:

- Test on an actual Blackwell GPU (e.g. RTX 5070 Ti) — Brno can confirm, or @aconesac who built the working binary
- Close [#656](https://github.com/waltsims/k-wave-python/issues/656) and [#622](https://github.com/waltsims/k-wave-python/issues/622) with the release version
- If the OMP refresh happened, close [#661](https://github.com/waltsims/k-wave-python/issues/661) too

## Test plan for this PR

This PR adds documentation only — no code change. The runbook will be exercised by the actual binary release work described above.

## Open work tracking

- [ ] Merge waltsims/kspaceFirstOrder-CUDA-linux#5 (Linux sm_120)
- [ ] Merge waltsims/kspaceFirstOrder-CUDA-windows#1 (Windows sm_120)
- [ ] Bump submodules in `kspacefirstorder-unified`, run CI, download artifacts
- [ ] Tag v1.4.0 releases on both CUDA binary repos
- [ ] (optional) Tag v0.4.0 OMP releases with current HDF5 ABI to close #661
- [ ] Open k-wave-python PR bumping version pins in `kwave/__init__.py`
- [ ] Close #656 and #622 once verified on a Blackwell box
