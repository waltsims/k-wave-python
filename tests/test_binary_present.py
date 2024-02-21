import os
import os.path

import pytest


# TODO: refactor this for new lazy install strategy


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_linux_afp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "linux", "acousticFieldPropagator-OMP"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_linux_omp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "linux", "kspaceFirstOrder-OMP"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_linux_cuda_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "linux", "kspaceFirstOrder-CUDA"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_afp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "acousticFieldPropagator-OMP.exe"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_cuda_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "kspaceFirstOrder-CUDA.exe"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_omp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "kspaceFirstOrder-OMP.exe"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_hdf5_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "hdf5.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_hdf5hl_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "hdf5_hl.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_cufft64_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "cufft64_10.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_libiomp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "libiomp5md.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_libmmd_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "libmmd.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_msvcp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "msvcp140.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_svmldispmd_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "svml_dispmd.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_szip_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "szip.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_vcruntime140_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "vcruntime140.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_zlib_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "zlib.dll"))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Running in GitHub Workflow.")
def test_windows_cufft6410_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), "kwave", "bin", "windows", "cufft64_10.dll"))
