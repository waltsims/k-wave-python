"""Unit tests for CppSimulation._resolve_binary_path."""
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.solvers.cpp_simulation import CppSimulation


class TestPrepare:
    def test_raises_when_input_already_exists(self, tmp_path):
        # Before fix: h5py raised a cryptic ValueError('name already exists')
        # instead of a clear FileExistsError.
        (tmp_path / "kwave_input.h5").write_bytes(b"")
        sim = CppSimulation.__new__(CppSimulation)
        with pytest.raises(FileExistsError, match="already exists"):
            sim.prepare(data_path=str(tmp_path))


class TestResolveBinaryPath:
    """Tests for CppSimulation._resolve_binary_path().

    chmod is exercised separately in test_execute_makes_binary_executable
    because it lives in _execute(), not the resolver.
    """

    def test_custom_path_existing_file_is_returned(self, tmp_path):
        binary = tmp_path / "my-kwave-binary"
        binary.write_text("#!/bin/sh\n")
        resolved = CppSimulation._resolve_binary_path("cpu", binary_path=str(binary))
        assert isinstance(resolved, Path)
        assert resolved == binary

    def test_custom_path_missing_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="Custom C\\+\\+ binary not found"):
            CppSimulation._resolve_binary_path("cpu", binary_path=str(missing))

    def test_custom_path_used_regardless_of_device(self, tmp_path):
        binary = tmp_path / "my-cuda-binary"
        binary.write_text("#!/bin/sh\n")
        # device="gpu" must not trigger the macOS CUDA guard when a custom path is given.
        resolved = CppSimulation._resolve_binary_path("gpu", binary_path=str(binary))
        assert resolved == binary

    def test_default_cpu_selects_omp_binary(self, tmp_path):
        omp_binary = tmp_path / "kspaceFirstOrder-OMP"
        omp_binary.write_text("#!/bin/sh\n")
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "linux"):
            resolved = CppSimulation._resolve_binary_path("cpu")
        assert resolved.name == "kspaceFirstOrder-OMP"

    def test_default_gpu_selects_cuda_binary(self, tmp_path):
        cuda_binary = tmp_path / "kspaceFirstOrder-CUDA"
        cuda_binary.write_text("#!/bin/sh\n")
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "linux"):
            resolved = CppSimulation._resolve_binary_path("gpu")
        assert resolved.name == "kspaceFirstOrder-CUDA"

    def test_default_missing_omp_raises_file_not_found(self, tmp_path):
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "linux"):
            with pytest.raises(FileNotFoundError, match="pip install k-wave-data"):
                CppSimulation._resolve_binary_path("cpu")

    def test_default_missing_cuda_on_linux_raises_file_not_found(self, tmp_path):
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "linux"):
            with pytest.raises(FileNotFoundError, match="pip install k-wave-data"):
                CppSimulation._resolve_binary_path("gpu")

    def test_default_gpu_on_macos_raises_value_error(self, tmp_path):
        # macOS + gpu must raise before the binary-existence check.
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "darwin"):
            with pytest.raises(ValueError, match="not supported on macOS"):
                CppSimulation._resolve_binary_path("gpu")

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows does not honor Unix executable bits")
    def test_execute_makes_binary_executable(self, tmp_path, monkeypatch):
        binary = tmp_path / "kspaceFirstOrder-OMP"
        binary.write_bytes(b"")
        binary.chmod(binary.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        assert not (binary.stat().st_mode & stat.S_IEXEC)

        monkeypatch.setattr(CppSimulation, "_resolve_binary_path", staticmethod(lambda device, binary_path=None: binary))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)

        sim = CppSimulation.__new__(CppSimulation)
        sim._execute("input.h5", "output.h5", device="cpu", num_threads=None, device_num=None, quiet=False, debug=False)

        assert binary.stat().st_mode & stat.S_IEXEC


@pytest.mark.parametrize("sig", [np.arange(1.0, 6.0), np.ones(1)])  # incl. a one-sample signal, expanded to (n_points, 1)
@pytest.mark.parametrize("field, mode", [(f, m) for f in ("p", "ux", "uy") for m in ("additive", "dirichlet")])
def test_source_scaled_for_binary(field, mode, sig, tmp_path):
    # The binary expects pre-scaled sources (kspaceFirstOrder_scaleSourceTerms), with c0 taken per source point (#785).
    N, (dx, dy), dt = (16, 12), (1e-4, 1.5e-4), 1e-8
    kgrid = kWaveGrid(N, (dx, dy))
    kgrid.setTime(sig.size, dt)
    c0, mask = np.full(N, 1500.0), np.zeros(N, bool)
    c0[:, 6:], mask[4, [3, 9]] = 1800.0, True  # one source point on each side of the interface
    source = kSource()
    setattr(source, f"{field[0]}_mask", mask)
    setattr(source, f"{field[0]}_mode", mode)
    setattr(source, field, sig[None])
    c = np.array([[1500.0], [1800.0]])
    additive = {"p": 2 * dt / (2 * c * dx), "ux": 2 * c * dt / dx, "uy": 2 * c * dt / dy}
    scale = additive[field] if mode == "additive" else 1 / (2 * c**2) if field == "p" else 1
    kwargs = dict(backend="cpp", save_only=True, data_path=str(tmp_path), pml_inside=True, pml_size=2)
    out = kspaceFirstOrder(kgrid, kWaveMedium(sound_speed=c0), source, kSensor(mask=mask), **kwargs)
    with h5py.File(out["input_file"]) as f:
        np.testing.assert_allclose(np.squeeze(f[f"{field}_source_input"]).T, np.squeeze(sig * scale), rtol=1e-6)
        assert f[f"{field[0]}_source_many"][()].item() == (np.size(scale) > 1)  # one row per point once expanded


@pytest.mark.parametrize("field", ["p", "ux"])
def test_source_rows_follow_written_index(field, tmp_path):
    # One row per source point is given in C-order of the mask; the binary pairs row k with entry k of the F-order index.
    N = (8, 8)
    kgrid = kWaveGrid(N, (1e-4, 1e-4))
    kgrid.setTime(4, 1e-8)
    mask = np.zeros(N, bool)
    mask[[2, 4], [5, 3]] = True  # C-order: (2,5), (4,3); F-order: (4,3), (2,5)
    source = kSource()
    setattr(source, f"{field[0]}_mask", mask)
    setattr(source, field, np.flatnonzero(mask)[:, None] * np.ones(4))  # each row holds its point's C linear index
    kwargs = dict(backend="cpp", save_only=True, data_path=str(tmp_path), pml_inside=True, pml_size=2)
    out = kspaceFirstOrder(kgrid, kWaveMedium(sound_speed=1500.0), source, kSensor(mask=mask), **kwargs)
    with h5py.File(out["input_file"]) as f:
        idx, rows = np.ravel(f[f"{field[0]}_source_index"]) - 1, np.squeeze(f[f"{field}_source_input"]).T[:, 0]
    expected = np.ravel_multi_index(np.unravel_index(idx, N, order="F"), N)
    np.testing.assert_allclose(rows / rows[0], expected / expected[0], rtol=1e-6)
