import logging
import os
import sys
import urllib.request
from os import environ
from pathlib import Path

# Test installation with:
# python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple/ k-Wave-python==0.3.0
VERSION = '0.3.1'
# Set environment variable to binaries to get rid of user warning
# This code is a crutch and should be removed when kspaceFirstOrder
# is refactored

platform = sys.platform

if platform.startswith('linux'):
    system = 'linux'
elif platform.startswith(('win', 'cygwin')):
    system = 'windows'
elif platform.startswith('darwin'):
    system = 'darwin'
    raise NotImplementedError('k-wave-python is currently unsupported on MacOS.')

binary_path = os.path.join(Path(__file__).parent, 'bin', system)
environ['KWAVE_BINARY_PATH'] = binary_path

url_base = "https://github.com/waltsims/"

prefix = "https://github.com/waltsims/kspaceFirstOrder-{0}-{1}/releases/download/v1.3.0/"

common_filenames = ["cufft64_10.dll", "hdf5.dll", "hdf5_hl.dll", "libiomp5md.dll", "libmmd.dll",
                    "msvcp140.dll", "svml_dispmd.dll", "szip.dll", "vcruntime140.dll", "zlib.dll"]

specific_omp_filenames = ["kspaceFirstOrder-OMP.exe"]
specific_cuda_filenames = ["kspaceFirstOrder-CUDA.exe"]


def binaries_present() -> bool:
    """
    Check if binaries are present
    Returns:
        bool, True if binaries are present, False otherwise

    """
    binary_list = {
        "linux": [
            # "acousticFieldPropagator-OMP",
            "kspaceFirstOrder-OMP",
            "kspaceFirstOrder-CUDA"
        ],
        "darwin": [
            # "acousticFieldPropagator-OMP",
            "kspaceFirstOrder-OMP",
            "kspaceFirstOrder-CUDA"
        ],
        "windows": specific_omp_filenames + specific_cuda_filenames + common_filenames
    }

    for binary in binary_list[system]:
        if not os.path.exists(os.path.join(binary_path, binary)):
            logging.log(logging.WARN,  f"{binary} not found")
            return False
    return True


def download_binaries(system_os: str, bin_type: str):
    """
    Download binary from release url
    Args:
        system_os: string, current system type
        bin_type: string of "OMP" or "CUDA"

    Returns:
        None

    """

    # GitHub release URLs
    url_dict = {
        "linux": {
            "cuda": [url_base + "kspaceFirstOrder-CUDA-linux/releases/download/v1.3.1/kspaceFirstOrder-CUDA"],
            "cpu": [url_base + "kspaceFirstOrder-OMP-linux/releases/download/v1.3.0/kspaceFirstOrder-OMP"],
        },
        # "darwin": {
        #     "cuda": [url_base + "kspaceFirstOrder-CUDA-linux/releases/download/v1.3/kspaceFirstOrder-CUDA"],
        #     "cpu": [url_base + "kspaceFirstOrder-OMP-linux/releases/download/v1.3.0/kspaceFirstOrder-OMP"],
        # },
        "windows": {
            "cuda": get_windows_release_urls("CUDA", "windows"),
            "cpu": get_windows_release_urls("OMP", "windows"),
        },
    }
    for url in url_dict[system_os][bin_type]:

        # Extract the file name from the GitHub release URL
        filename = url.split("/")[-1]

        logging.log(logging.INFO, f"Downloading {filename} to {binary_path}...")

        # Create the directory if it does not yet exist
        os.makedirs(binary_path, exist_ok=True)

        # Download the binary file
        urllib.request.urlretrieve(url, os.path.join(binary_path, filename))


def get_windows_release_urls(version: str, system_type: str) -> list:
    if version == "OMP":
        specific_filenames = specific_omp_filenames
    elif version == "CUDA":
        specific_filenames = specific_cuda_filenames
    else:
        specific_filenames = []

    release_urls = []
    for filename in common_filenames + specific_filenames:
        release_urls.append(prefix.format(version, system_type) + filename)
    return release_urls


def install_binaries():
    for binary_type in ["cpu", "cuda"]:
        download_binaries(system, binary_type)


if not binaries_present():
    install_binaries()
