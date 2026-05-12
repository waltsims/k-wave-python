"""Unit tests for CppSimulation._resolve_binary_path."""
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kwave.solvers.cpp_simulation import CppSimulation


class TestResolveBinaryPath:
    """Tests for CppSimulation._resolve_binary_path().

    Note: _resolve_binary_path() is responsible only for locating the binary.
    Making it executable (chmod +x) is done in _execute() after the path is
    resolved, keeping the two concerns separate.
    """

    # ------------------------------------------------------------------
    # Custom binary_path provided by the caller
    # ------------------------------------------------------------------

    def test_custom_path_existing_file_is_returned(self, tmp_path):
        binary = tmp_path / "my-kwave-binary"
        binary.write_text("#!/bin/sh\n")
        resolved = CppSimulation._resolve_binary_path("cpu", binary_path=str(binary))
        assert resolved == binary

    def test_custom_path_returns_path_object(self, tmp_path):
        binary = tmp_path / "my-kwave-binary"
        binary.write_text("#!/bin/sh\n")
        resolved = CppSimulation._resolve_binary_path("cpu", binary_path=str(binary))
        assert isinstance(resolved, Path)

    def test_custom_path_missing_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="Custom C\\+\\+ binary not found"):
            CppSimulation._resolve_binary_path("cpu", binary_path=str(missing))

    def test_custom_path_used_regardless_of_device(self, tmp_path):
        binary = tmp_path / "my-cuda-binary"
        binary.write_text("#!/bin/sh\n")
        # Even when device="gpu" the custom path should be returned as-is,
        # without triggering the macOS CUDA guard or any other default logic.
        resolved = CppSimulation._resolve_binary_path("gpu", binary_path=str(binary))
        assert resolved == binary

    # ------------------------------------------------------------------
    # Default bundled binary (no custom path)
    # ------------------------------------------------------------------

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
        # No OMP binary in tmp_path
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "linux"):
            with pytest.raises(FileNotFoundError, match="pip install k-wave-data"):
                CppSimulation._resolve_binary_path("cpu")

    def test_default_missing_cuda_on_linux_raises_file_not_found(self, tmp_path):
        # No CUDA binary in tmp_path on a non-macOS platform
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "linux"):
            with pytest.raises(FileNotFoundError, match="pip install k-wave-data"):
                CppSimulation._resolve_binary_path("gpu")

    def test_default_gpu_on_macos_raises_value_error(self, tmp_path):
        # macOS + gpu → ValueError before checking binary existence
        with patch("kwave.BINARY_PATH", tmp_path), patch("kwave.PLATFORM", "darwin"):
            with pytest.raises(ValueError, match="not supported on macOS"):
                CppSimulation._resolve_binary_path("gpu")

    # ------------------------------------------------------------------
    # Executable permission is set by _execute() after path resolution
    # ------------------------------------------------------------------

    def test_execute_makes_binary_executable(self, tmp_path, monkeypatch):
        """_execute() sets the executable bit on the resolved binary."""
        binary = tmp_path / "kspaceFirstOrder-OMP"
        binary.write_bytes(b"")
        # Remove executable bit so we can verify _execute() sets it
        binary.chmod(binary.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        assert not (binary.stat().st_mode & stat.S_IEXEC)

        # Patch _resolve_binary_path so it returns our binary without querying kwave paths
        monkeypatch.setattr(CppSimulation, "_resolve_binary_path", staticmethod(lambda device, binary_path=None: binary))
        # Patch subprocess.run so no real process is spawned
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)

        sim = CppSimulation.__new__(CppSimulation)
        sim._execute("input.h5", "output.h5", device="cpu", num_threads=None, device_num=None, quiet=False, debug=False)

        assert binary.stat().st_mode & stat.S_IEXEC
