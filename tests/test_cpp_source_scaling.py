"""Source scaling in the HDF5 input written by ``kspaceFirstOrder(..., backend="cpp")``.

The C++ binary expects time-varying sources to be pre-scaled (as MATLAB's
kspaceFirstOrder_scaleSourceTerms.m and the legacy ``kspaceFirstOrder2D`` path
do before saving).  Regression tests for issue #785, where the unified cpp path
wrote ``source.p`` / ``source.u*`` unscaled, so results were off by
``N*c0*dx/(2*dt)`` relative to ``backend="python"``.
"""

import copy
import os

import h5py
import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

N = (32, 24)
DX, DY = 1e-4, 1.5e-4  # anisotropic so the per-axis velocity scaling is exercised
NT = 10  # deliberately different from the number of source points (12)
PML = 6
C_LOW, C_HIGH = 1500.0, 1800.0
MODES = ["additive", "additive-no-correction", "dirichlet"]
MODE_FLAG = {"dirichlet": 0, "additive-no-correction": 1, "additive": 2}


def _kgrid():
    kgrid = kWaveGrid(N, (DX, DY))
    kgrid.setTime(NT, 1e-8)
    return kgrid


def _medium(heterogeneous):
    if not heterogeneous:
        return kWaveMedium(sound_speed=C_LOW, density=1000.0)
    c = C_LOW * np.ones(N)
    c[:, N[1] // 2 :] = C_HIGH
    return kWaveMedium(sound_speed=c, density=1000.0)


def _mask():
    """Source points on both sides of the sound-speed interface, on several rows and columns
    so that C-order and F-order traversals differ."""
    mask = np.zeros(N, dtype=bool)
    mask[10, 8:16] = True
    mask[14, 10:14] = True
    return mask


def _signal(many):
    t = np.arange(NT)
    base = np.sin(2 * np.pi * t / NT) + 0.25
    if not many:
        return base.reshape(1, -1)
    n_src = int(_mask().sum())
    return (1.0 + np.arange(n_src))[:, None] * base[None, :]


def _signal_f_order(many):
    """``_signal`` with per-point rows (C-order of the mask, the unified API's convention)
    rearranged to the F-order of ``*_source_index``; a single series is unchanged."""
    signal = _signal(many)
    if not many:
        return signal
    mask = _mask()
    f_points = np.unravel_index(np.flatnonzero(mask.flatten(order="F")), N, order="F")
    return signal[np.searchsorted(np.flatnonzero(mask), np.ravel_multi_index(f_points, N))]


def _source(field, mode, many, signal=None):
    signal = _signal(many) if signal is None else signal
    source = kSource()
    if field == "p":
        source.p_mask = _mask()
        source.p = signal
        source.p_mode = mode
    else:
        source.u_mask = _mask()
        for component in field.split("+"):
            setattr(source, component, signal)
        source.u_mode = mode
    return source


def _c0_at_source(heterogeneous):
    """Sound speed per source point, in the F-order of ``*_source_index``."""
    c = np.broadcast_to(np.asarray(_medium(heterogeneous).sound_speed, dtype=float), N)
    return c.flatten(order="F")[_mask().flatten(order="F")]


def _expected_scale(field, mode, heterogeneous):
    c0 = _c0_at_source(heterogeneous)
    dt = float(_kgrid().dt)
    ndim = len(N)
    if field == "p":
        if mode == "dirichlet":
            return 1.0 / (ndim * c0**2)
        return 2.0 * dt / (ndim * c0 * DX)
    if mode == "dirichlet":
        return np.ones_like(c0)
    return 2.0 * c0 * dt / {"ux": DX, "uy": DY}[field]


def _read(path, names):
    with h5py.File(path, "r") as f:
        return {name: np.squeeze(f[name][()]) for name in names if name in f}


def _write_unified(field, mode, heterogeneous, many, tmp_path, source=None):
    source = _source(field, mode, many) if source is None else source
    result = kspaceFirstOrder(
        _kgrid(),
        _medium(heterogeneous),
        source,
        kSensor(mask=np.ones(N, dtype=bool)),
        backend="cpp",
        save_only=True,
        data_path=str(tmp_path / "unified"),
        pml_inside=True,
        pml_size=PML,
    )
    return result["input_file"]


def _write_legacy(field, mode, heterogeneous, many, tmp_path):
    data_path = tmp_path / "legacy"
    os.makedirs(data_path, exist_ok=True)
    kspaceFirstOrder2D(
        kgrid=_kgrid(),
        medium=_medium(heterogeneous),
        # the legacy path takes per-point rows in F-order (MATLAB convention)
        source=_source(field, mode, many, signal=_signal_f_order(many)),
        sensor=kSensor(mask=np.ones(N, dtype=bool)),
        simulation_options=SimulationOptions(
            pml_inside=True,
            pml_size=PML,
            smooth_p0=False,
            save_to_disk=True,
            save_to_disk_exit=True,
            input_filename="legacy_input.h5",
            data_path=str(data_path),
        ),
        execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
    )
    return str(data_path / "legacy_input.h5")


def _per_point(data, n_src):
    """Signal applied at each source point, shape (n_src, Nt), for a single or many series."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 2 and data.shape != (n_src, NT):
        data = data.T  # h5py reads the binary's column-major layout with reversed dimensions
    return np.broadcast_to(data, (n_src, NT))


@pytest.mark.parametrize("many", [False, True], ids=["single", "many"])
@pytest.mark.parametrize("heterogeneous", [False, True], ids=["homogeneous", "heterogeneous"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("field", ["p", "ux", "uy"])
def test_source_input_is_scaled(field, mode, heterogeneous, many, tmp_path):
    """Written source signals carry the scaling the binary expects (per source point)."""
    prefix = "p" if field == "p" else "u"
    written = _read(
        _write_unified(field, mode, heterogeneous, many, tmp_path),
        [f"{field}_source_input", f"{prefix}_source_mode", f"{prefix}_source_many"],
    )

    n_src = int(_mask().sum())
    applied = _per_point(written[f"{field}_source_input"], n_src)
    expected = np.broadcast_to(_signal_f_order(many), (n_src, NT)) * _expected_scale(field, mode, heterogeneous)[:, None]

    np.testing.assert_allclose(applied, expected, rtol=1e-5)
    assert int(written[f"{prefix}_source_mode"]) == MODE_FLAG[mode]
    # many == 1 means "one series per source point" to the binary
    assert int(written[f"{prefix}_source_many"]) == int(written[f"{field}_source_input"].size == n_src * NT)


def test_default_mode_is_additive(tmp_path):
    """source.p_mode=None (the kSource default) is the k-space corrected additive source."""
    source = _source("p", None, many=False)
    written = _read(_write_unified("p", None, False, False, tmp_path, source=source), ["p_source_input", "p_source_mode"])
    assert int(written["p_source_mode"]) == MODE_FLAG["additive"]
    expected = _signal(False).ravel() * 2.0 * float(_kgrid().dt) / (len(N) * C_LOW * DX)
    np.testing.assert_allclose(written["p_source_input"].ravel(), expected, rtol=1e-5)


def test_source_is_not_mutated(tmp_path):
    """Scaling is applied to the serialized copy only."""
    source = _source("p", "additive", many=True)
    before = copy.deepcopy(source.p)
    _write_unified("p", "additive", True, True, tmp_path, source=source)
    np.testing.assert_array_equal(source.p, before)


def test_1d_signal_is_accepted(tmp_path):
    """A flat (Nt,) signal is treated as a single series."""
    source = _source("p", "additive", many=False)
    source.p = source.p.ravel()
    written = _read(_write_unified("p", "additive", False, False, tmp_path, source=source), ["p_source_input", "p_source_many"])
    expected = _signal(False).ravel() * 2.0 * float(_kgrid().dt) / (len(N) * C_LOW * DX)
    np.testing.assert_allclose(written["p_source_input"].ravel(), expected, rtol=1e-5)
    assert int(written["p_source_many"]) == 0


# The legacy path is the reference for what the binary expects.  Combinations the legacy
# path cannot serialize (heterogeneous or dirichlet velocity sources, uy without ux) or
# where it applies one sound speed to a series shared by points with different sound
# speeds (heterogeneous + single series) are covered analytically above.
LEGACY_CASES = [
    ("p", "additive", False, False),
    ("p", "additive", False, True),
    ("p", "additive", True, True),
    ("p", "additive-no-correction", False, True),
    ("p", "dirichlet", False, False),
    ("p", "dirichlet", True, True),
    ("ux", "additive", False, False),
    ("ux", "additive", False, True),
    ("ux+uy", "additive", False, True),  # the legacy path requires ux alongside uy
]


@pytest.mark.parametrize("field, mode, heterogeneous, many", LEGACY_CASES)
def test_source_datasets_match_legacy_path(field, mode, heterogeneous, many, tmp_path):
    """The unified cpp path writes the same source datasets as kspaceFirstOrder2D."""
    prefix = "p" if field == "p" else "u"
    names = [f"{prefix}_source_index", f"{prefix}_source_mode", f"{prefix}_source_many"]
    for component in field.split("+"):
        names += [f"{component}_source_input", f"{component}_source_flag"]
    unified = _read(_write_unified(field, mode, heterogeneous, many, tmp_path), names)
    legacy = _read(_write_legacy(field, mode, heterogeneous, many, tmp_path), names)

    assert set(unified) == set(names) == set(legacy)
    for name in names:
        assert unified[name].shape == legacy[name].shape, name
        np.testing.assert_allclose(unified[name], legacy[name], rtol=1e-5, err_msg=name)
