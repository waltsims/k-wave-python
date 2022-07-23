import os
import os.path

os.chdir('..')


def test_linux_afp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'linux', 'acousticFieldPropagator-OMP'))


def test_linux_omp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'linux', 'kspaceFirstOrder-OMP'))


def test_linux_cuda_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'linux', 'kspaceFirstOrder-CUDA'))


def test_windows_afp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'acousticFieldPropagator-OMP.exe'))


def test_windows_cuda_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'kspaceFirstOrder-CUDA.exe'))


def test_windows_omp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'kspaceFirstOrder-OMP.exe'))


def test_windows_hdf5_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'hdf5.dll'))


def test_windows_hdf5hl_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'hdf5_hl.dll'))


def test_windows_cufft64_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'cufft64_10.dll'))


def test_windows_libiomp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'libiomp5md.dll'))


def test_windows_libmmd_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'libmmd.dll'))


def test_windows_msvcp_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'msvcp140.dll'))


def test_windows_svmldispmd_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'svml_dispmd.dll'))


def test_windows_szip_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'szip.dll'))


def test_windows_vcruntime140_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'vcruntime140.dll'))


def test_windows_zlib_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'zlib.dll'))


def test_windows_cufft6410_binaries_present():
    assert os.path.exists(os.path.join(os.getcwd(), 'kwave', 'bin', 'windows', 'cufft64_10.dll'))
