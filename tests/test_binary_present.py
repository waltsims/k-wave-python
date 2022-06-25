import setup_test
import pytest
import os


@pytest.mark.skipif(os.environ.get("CI"), reason="Running in GitHub Workflow.")
def test_linux_binaries_present():
    assert os.path.exists('kwave/bin/linux/acousticFieldPropagator-OMP')
    assert os.path.exists('kwave/bin/linux/kspaceFirstOrder-OMP')
    assert os.path.exists('kwave/bin/linux/kspaceFirstOrder-CUDA')


@pytest.mark.skipif(os.environ.get("CI"), reason="Running in GitHub Workflow.")
def test_windows_binaries_present():
    assert os.path.exists('kwave/bin/windows/acousticFieldPropagator-OMP.exe')
    assert os.path.exists('kwave/bin/windows/kspaceFirstOrder-CUDA.exe')
    assert os.path.exists('kwave/bin/windows/kspaceFirstOrder-OMP.exe')
    assert os.path.exists('kwave/bin/windows/hdf5.dll')
    assert os.path.exists('kwave/bin/windows/hdf5_hl.dll')
    assert os.path.exists('kwave/bin/windows/cufft64_10.dll')
    assert os.path.exists('kwave/bin/windows/libiomp5md.dll')
    assert os.path.exists('kwave/bin/windows/libmmd.dll')
    assert os.path.exists('kwave/bin/windows/msvcp140.dll')
    assert os.path.exists('kwave/bin/windows/svml_dispmd.dll')
    assert os.path.exists('kwave/bin/windows/szip.dll')
    assert os.path.exists('kwave/bin/windows/vcruntime140.dll')
    assert os.path.exists('kwave/bin/windows/zlib.dll')
    assert os.path.exists('kwave/bin/windows/cufft64_10.dll')
