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


def test_staggered_density_keeps_rho0_on_last_plane(tmp_path):
    # Before fix: np.roll averaged the last plane along each axis with the first one.
    rho0 = np.full((16, 12), 1000.0)
    rho0[0], rho0[:, 0] = 2000.0, 3000.0
    kgrid = kWaveGrid(rho0.shape, (1e-4, 1e-4))
    kgrid.makeTime(1500)
    source = kSource()
    source.p0 = np.ones(rho0.shape)
    medium = kWaveMedium(1500, density=rho0)
    kspaceFirstOrder(kgrid, medium, source, backend="cpp", save_only=True, pml_size=4, pml_inside=True, data_path=str(tmp_path))
    with h5py.File(tmp_path / "kwave_input.h5") as f:
        sgx, sgy = f["rho0_sgx"][0].T, f["rho0_sgy"][0].T  # stored transposed
    assert np.array_equal(sgx[-1], rho0[-1]) and np.array_equal(sgx[:-1], (rho0[:-1] + rho0[1:]) / 2)
    assert np.array_equal(sgy[:, -1], rho0[:, -1]) and np.array_equal(sgy[:, :-1], (rho0[:, :-1] + rho0[:, 1:]) / 2)


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
