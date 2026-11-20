"""Row order of per-point source signals written by ``kspaceFirstOrder(..., backend="cpp")``.

``source.p`` / ``source.u*`` with one row per source point follow the C-order
(row-major) of the mask's nonzero points, which is how ``backend="python"``
applies them.  The C++ binary pairs row k with entry k of ``*_source_index``,
which is written in F-order, so the rows have to be permuted on serialization.
"""

import subprocess
import sys

import h5py
import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.solvers.cpp_simulation import CppSimulation

NT = 12
PML = 4
C_LOW, C_HIGH = 1500.0, 1800.0
SPACING = (1e-4, 1.5e-4, 2e-4)

# Masks whose C-order and F-order point enumerations differ.
MASKS = {
    "2d-two-points": ((16, 20), [(2, 5), (4, 3)]),
    "2d-L-shape": ((16, 20), [(3, 4), (3, 5), (3, 6), (3, 7), (4, 4), (5, 4), (6, 4)]),
    "3d": ((12, 14, 16), [(2, 5, 9), (4, 3, 9), (4, 6, 2), (7, 3, 5), (7, 4, 5), (9, 1, 12)]),
}
FIELDS = ["p", "ux", "uy", "uz"]


def _setup(case, heterogeneous=False):
    shape, points = MASKS[case]
    ndim = len(shape)
    kgrid = kWaveGrid(shape, SPACING[:ndim])
    kgrid.setTime(NT, 1e-8)
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(np.array(points).T)] = True
    if heterogeneous:
        c = C_LOW * np.ones(shape)
        c[..., shape[-1] // 2 :] = C_HIGH
    else:
        c = C_LOW
    return kgrid, kWaveMedium(sound_speed=c, density=1000.0), mask


def _point_signals(mask):
    """One row per source point in C-order; row k is ``(k + 1) * base`` so it identifies its point."""
    base = np.sin(2 * np.pi * np.arange(NT) / NT) + 0.25
    return (1.0 + np.arange(int(mask.sum())))[:, None] * base[None, :], base


def _write(case, field, mode, heterogeneous, tmp_path):
    kgrid, medium, mask = _setup(case, heterogeneous)
    signals, base = _point_signals(mask)
    source = kSource()
    if field == "p":
        source.p_mask, source.p, source.p_mode = mask, signals, mode
    else:
        source.u_mask, source.u_mode = mask, mode
        setattr(source, field, signals)
    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        kSensor(mask=np.ones(mask.shape, dtype=bool)),
        backend="cpp",
        save_only=True,
        data_path=str(tmp_path),
        pml_inside=True,
        pml_size=PML,
    )
    prefix = "p" if field == "p" else "u"
    with h5py.File(result["input_file"], "r") as f:
        index = np.squeeze(f[f"{prefix}_source_index"][()]).astype(int)
        data = np.squeeze(f[f"{field}_source_input"][()]).astype(float)
        many = int(np.squeeze(f[f"{prefix}_source_many"][()]))
    if data.shape != (index.size, NT):
        data = data.T  # h5py reads the binary's column-major layout with reversed dimensions
    return kgrid, medium, mask, base, index, data, many


def _cases():
    for case, (shape, _) in MASKS.items():
        for field in FIELDS[: 1 + len(shape)]:
            yield case, field


@pytest.mark.parametrize("heterogeneous", [False, True], ids=["homogeneous", "heterogeneous"])
@pytest.mark.parametrize("mode", ["additive", "dirichlet"])
@pytest.mark.parametrize("case, field", list(_cases()))
def test_signal_row_k_belongs_to_source_index_k(case, field, mode, heterogeneous, tmp_path):
    """Row k of ``*_source_input`` is the (scaled) signal of the point at ``*_source_index[k]``."""
    kgrid, medium, mask, base, index, data, many = _write(case, field, mode, heterogeneous, tmp_path)
    shape = mask.shape
    ndim = len(shape)
    assert many == 1

    # the written index is F-order and differs from the C-order enumeration for these masks
    points = np.unravel_index(index - 1, shape, order="F")
    assert np.all(mask[points]) and index.size == mask.sum()
    c_rank = np.searchsorted(np.flatnonzero(mask), np.ravel_multi_index(points, shape))
    assert not np.array_equal(c_rank, np.arange(index.size))

    # scaling the binary expects, evaluated at the point each written row is paired with
    c0 = np.broadcast_to(np.asarray(medium.sound_speed, dtype=float), shape)[points]
    dt = float(kgrid.dt)
    if field == "p":
        scale = 1.0 / (ndim * c0**2) if mode == "dirichlet" else 2.0 * dt / (ndim * c0 * SPACING[0])
    else:
        scale = np.ones_like(c0) if mode == "dirichlet" else 2.0 * c0 * dt / SPACING["xyz".index(field[1])]

    # the user's row for that point is (c_rank + 1) * base
    expected = ((c_rank + 1.0) * scale)[:, None] * base[None, :]
    np.testing.assert_allclose(data, expected, rtol=1e-5)


def test_velocity_components_share_the_permutation(tmp_path):
    """All velocity components are reordered consistently with the single ``u_source_index``."""
    kgrid, medium, mask = _setup("2d-L-shape")
    signals, base = _point_signals(mask)
    source = kSource()
    source.u_mask, source.u_mode = mask, "dirichlet"
    source.ux, source.uy = signals, -2.0 * signals
    result = kspaceFirstOrder(
        kgrid, medium, source, kSensor(mask=mask), backend="cpp", save_only=True, data_path=str(tmp_path), pml_inside=True, pml_size=PML
    )
    with h5py.File(result["input_file"], "r") as f:
        index = np.squeeze(f["u_source_index"][()]).astype(int)
        ux = np.squeeze(f["ux_source_input"][()]).astype(float)
        uy = np.squeeze(f["uy_source_input"][()]).astype(float)
    ux, uy = (d if d.shape == (index.size, NT) else d.T for d in (ux, uy))
    points = np.unravel_index(index - 1, mask.shape, order="F")
    c_rank = np.searchsorted(np.flatnonzero(mask), np.ravel_multi_index(points, mask.shape))
    expected = (c_rank + 1.0)[:, None] * base[None, :]
    np.testing.assert_allclose(ux, expected, rtol=1e-5)
    np.testing.assert_allclose(uy, -2.0 * expected, rtol=1e-5)


def test_source_signal_is_not_mutated(tmp_path):
    kgrid, medium, mask = _setup("2d-L-shape")
    signals, _ = _point_signals(mask)
    source = kSource()
    source.p_mask, source.p = mask, signals.copy()
    kspaceFirstOrder(
        kgrid, medium, source, kSensor(mask=mask), backend="cpp", save_only=True, data_path=str(tmp_path), pml_inside=True, pml_size=PML
    )
    np.testing.assert_array_equal(source.p, signals)


def _omp_binary_runs():
    if sys.platform == "darwin":
        return False
    try:
        binary = CppSimulation._resolve_binary_path("cpu")
        return subprocess.run([str(binary), "--version"], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _omp_binary_runs(), reason="C++ OMP binary is not runnable on this machine")
@pytest.mark.parametrize("driven_row", [0, 1])
@pytest.mark.parametrize("field", ["p", "ux"])
def test_cpp_matches_python_for_per_point_signals(field, driven_row, tmp_path):
    """Both backends drive the same grid point from the same signal row."""
    shape, nt = (32, 32), 60
    kgrid = kWaveGrid(shape, (1e-4, 1e-4))
    kgrid.setTime(nt, 1e-8)
    medium = kWaveMedium(sound_speed=C_LOW, density=1000.0)

    mask = np.zeros(shape, dtype=bool)
    mask[8, 20] = True  # first in C-order, second in F-order
    mask[22, 10] = True
    t = np.arange(nt)
    signals = np.zeros((2, nt))
    signals[driven_row] = np.sin(2 * np.pi * t / 20) * (t < 20)

    # a single sensor point next to the first C-order source point, so sensor ordering plays no role
    sensor_mask = np.zeros(shape, dtype=bool)
    sensor_mask[10, 20] = True
    traces = {}
    for backend in ("python", "cpp"):
        source = kSource()
        if field == "p":
            source.p_mask, source.p = mask, signals.copy()
        else:
            source.u_mask, source.ux = mask, signals.copy()
        result = kspaceFirstOrder(
            kgrid,
            medium,
            source,
            kSensor(mask=sensor_mask),
            backend=backend,
            pml_inside=True,
            pml_size=6,
            quiet=True,
            num_threads=1,
            data_path=str(tmp_path),
        )
        traces[backend] = np.asarray(result["p"], dtype=float).ravel()

    # python: the sensor sees the wave only when the adjacent point (row 0) is the driven one
    peaks = {backend: np.abs(trace).max() for backend, trace in traces.items()}
    np.testing.assert_allclose(traces["cpp"], traces["python"], atol=1e-3 * peaks["python"], rtol=0)
