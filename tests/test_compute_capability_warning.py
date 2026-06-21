"""Tests for the CUDA compute-capability detection helper and the
RuntimeWarning emitted by ``CppSimulation`` when the host GPU is below
the supported compute capability.
"""

import stat
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from kwave.solvers.cpp_simulation import CppSimulation
from kwave.utils import cuda as cuda_utils
from kwave.utils.cuda import _reset_compute_capability_cache, get_min_compute_capability


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure each test starts with a clean compute-capability cache."""
    _reset_compute_capability_cache()
    yield
    _reset_compute_capability_cache()


def _make_completed_process(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr="")


class TestGetMinComputeCapability:
    def test_returns_parsed_capability_for_single_gpu(self):
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", return_value=_make_completed_process("7.5\n")),
        ):
            assert get_min_compute_capability() == (7, 5)

    def test_returns_minimum_for_multiple_gpus(self):
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", return_value=_make_completed_process("8.0\n6.1\n7.5\n")),
        ):
            assert get_min_compute_capability() == (6, 1)

    def test_returns_none_when_nvidia_smi_missing(self):
        with patch.object(cuda_utils.shutil, "which", return_value=None):
            assert get_min_compute_capability() is None

    def test_returns_none_when_nvidia_smi_fails(self):
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", return_value=_make_completed_process("", returncode=9)),
        ):
            assert get_min_compute_capability() is None

    def test_returns_none_on_timeout(self):
        def _raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", side_effect=_raise_timeout),
        ):
            assert get_min_compute_capability() is None

    def test_returns_none_on_os_error(self):
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", side_effect=OSError("nope")),
        ):
            assert get_min_compute_capability() is None

    def test_returns_none_on_empty_output(self):
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", return_value=_make_completed_process("\n")),
        ):
            assert get_min_compute_capability() is None

    def test_returns_none_on_unparseable_output(self):
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", return_value=_make_completed_process("garbage\n")),
        ):
            assert get_min_compute_capability() is None

    def test_result_is_cached(self):
        mock_run = MagicMock(return_value=_make_completed_process("7.5\n"))
        with (
            patch.object(cuda_utils.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(cuda_utils.subprocess, "run", mock_run),
        ):
            assert get_min_compute_capability() == (7, 5)
            assert get_min_compute_capability() == (7, 5)
            assert get_min_compute_capability() == (7, 5)
        assert mock_run.call_count == 1


def _make_executable_binary(tmp_path, name="kspaceFirstOrder-CUDA"):
    binary = tmp_path / name
    binary.write_bytes(b"")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


class TestExecuteWarnsOnUnsupportedGpu:
    def _run_execute(self, tmp_path, monkeypatch, *, device, binary_name):
        binary = _make_executable_binary(tmp_path, name=binary_name)
        monkeypatch.setattr(CppSimulation, "_resolve_binary_path", staticmethod(lambda d, binary_path=None: binary))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
        sim = CppSimulation.__new__(CppSimulation)
        sim._execute("input.h5", "output.h5", device=device, num_threads=None, device_num=None, quiet=False, debug=False)

    def test_warning_fires_when_gpu_below_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kwave.solvers.cpp_simulation.get_min_compute_capability", lambda: (6, 1))
        with pytest.warns(RuntimeWarning, match="compute capability 6\\.1"):
            self._run_execute(tmp_path, monkeypatch, device="gpu", binary_name="kspaceFirstOrder-CUDA")

    def test_warning_text_mentions_maxwell_pascal_volta(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kwave.solvers.cpp_simulation.get_min_compute_capability", lambda: (7, 0))
        with pytest.warns(RuntimeWarning) as record:
            self._run_execute(tmp_path, monkeypatch, device="gpu", binary_name="kspaceFirstOrder-CUDA")
        msg = str(record[0].message)
        assert "Maxwell" in msg
        assert "Pascal" in msg
        assert "Volta" in msg
        assert "backend='python'" in msg

    def test_warning_does_not_fire_when_gpu_supported(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kwave.solvers.cpp_simulation.get_min_compute_capability", lambda: (7, 5))
        with warnings_caught() as caught:
            self._run_execute(tmp_path, monkeypatch, device="gpu", binary_name="kspaceFirstOrder-CUDA")
        assert not any(issubclass(w.category, RuntimeWarning) and "compute capability" in str(w.message) for w in caught)

    def test_warning_does_not_fire_when_nvidia_smi_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kwave.solvers.cpp_simulation.get_min_compute_capability", lambda: None)
        with warnings_caught() as caught:
            self._run_execute(tmp_path, monkeypatch, device="gpu", binary_name="kspaceFirstOrder-CUDA")
        assert not any(issubclass(w.category, RuntimeWarning) and "compute capability" in str(w.message) for w in caught)

    def test_warning_does_not_fire_on_cpu_device(self, tmp_path, monkeypatch):
        def _explode():
            raise AssertionError("get_min_compute_capability must not be called when device='cpu'")

        monkeypatch.setattr("kwave.solvers.cpp_simulation.get_min_compute_capability", _explode)
        # Should not raise — _warn_if_unsupported_gpu must be skipped entirely.
        self._run_execute(tmp_path, monkeypatch, device="cpu", binary_name="kspaceFirstOrder-OMP")


class warnings_caught:  # noqa: N801 - intentional lower-case context manager
    """Tiny wrapper around warnings.catch_warnings that always records."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._record = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self._record

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)
