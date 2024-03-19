import logging
import os
import sys
import urllib.request
from os import environ
from pathlib import Path
from typing import List
import hashlib
import json

# Test installation with:
# python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple/ k-Wave-python==0.3.0
VERSION = "0.3.2"
# Set environment variable to binaries to get rid of user warning
# This code is a crutch and should be removed when kspaceFirstOrder
# is refactored

platform = sys.platform

if platform.startswith("linux"):
    system = "linux"
elif platform.startswith(("win", "cygwin")):
    system = "windows"
elif platform.startswith("darwin"):
    system = "darwin"
    raise NotImplementedError("k-wave-python is currently unsupported on MacOS.")

binary_path = os.path.join(Path(__file__).parent, "bin", system)
environ["KWAVE_BINARY_PATH"] = binary_path

url_base = "https://github.com/waltsims/"

prefix = "https://github.com/waltsims/kspaceFirstOrder-{0}-{1}/releases/download/v1.3.0/"

common_filenames = [
    ("cufft64_10.dll", None),
    ("hdf5.dll", None),
    ("hdf5_hl.dll", None),
    ("libiomp5md.dll", None),
    ("libmmd.dll", None),
    ("msvcp140.dll", None),
    ("svml_dispmd.dll", None),
    ("szip.dll", None),
    ("vcruntime140.dll", None),
    ("zlib.dll", None),
]

specific_omp_filenames = [("kspaceFirstOrder-OMP.exe", "cpu")]
specific_cuda_filenames = [("kspaceFirstOrder-CUDA.exe", "cuda")]


def get_windows_release_urls(version: str, system_type: str) -> list:
    if version == "OMP":
        specific_filenames = specific_omp_filenames
    elif version == "CUDA":
        specific_filenames = specific_cuda_filenames
    else:
        specific_filenames = []

    release_urls = []
    for filename, binary_type in common_filenames + specific_filenames:
        release_urls.append(prefix.format(version, system_type) + filename)
    return release_urls


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


def _hash_file(filepath: str) -> str:
    buf_size = 65536  # 64kb chunks
    md5 = hashlib.md5()

    with open(filepath, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def _is_binary_present(binary_name: str, binary_type: str) -> bool:
    binary_filepath = os.path.join(binary_path, binary_name)
    binary_file_exists = os.path.exists(binary_filepath)
    if not binary_file_exists:
        return False
    
    if binary_type is None:
        # this is non-kwave windows binary
        # it already exists according to the check above
        return True
    
    existing_metadata_path = os.path.join(binary_path, f'{binary_name}_metadata.json')
    if not os.path.exists(existing_metadata_path):
        # metadata does not exist => binaries may or may not exist
        # Let's play safe and claim they don't exist
        # This will trigger binary download and generation of binary metadata
        return False
    existing_metadata = json.loads(Path(existing_metadata_path).read_text())

    # If metadata was somehow corrupted
    file_hash = _hash_file(binary_filepath)
    if existing_metadata['file_hash'] != file_hash:
        return False
    
    # If there is a new binary
    latest_url = url_dict[system][binary_type][0]
    if existing_metadata['url'] != latest_url:
        return False
    
    # No need to check `version` field for now
    # because we version is already present in the URL
    return True



def binaries_present() -> bool:
    """
    Check if binaries are present
    Returns:
        bool, True if binaries are present, False otherwise

    """
    binary_list = {
        "linux": [
            # "acousticFieldPropagator-OMP",
            ("kspaceFirstOrder-OMP", "cpu"),
            ("kspaceFirstOrder-CUDA", "cuda")
        ],
        "darwin": [
            # "acousticFieldPropagator-OMP",
            ("kspaceFirstOrder-OMP", "cpu"),
            ("kspaceFirstOrder-CUDA", "cuda")
        ],
        "windows": specific_omp_filenames + specific_cuda_filenames + common_filenames,
    }
    missing_binaries: List[str] = []

    for binary_name, binary_type in binary_list[system]:
        if not _is_binary_present(binary_name, binary_type):
            missing_binaries.append(binary_name)

    if len(missing_binaries) > 0:
        missing_binaries_str = ", ".join(missing_binaries)
        logging.log(
            logging.INFO,
            f"Following binaries were not found: {missing_binaries_str}"
            "If this is first time you're running k-wave-python, "
            "binaries will be downloaded automatically.",
        )

    return len(missing_binaries) == 0


def _record_binary_metadata(binary_version: str, binary_filepath: str, binary_url: str, filename: str) -> None:
    # note: version is not immediately useful at the moment
    # because it is already present in the url and we use url to understand if versions match
    # However, let's record it anyway. Maybe it will be useful in the future.
    metadata = {
        "url": binary_url,
        "version": binary_version,
        "file_hash": _hash_file(binary_filepath)
    }
    metadata_filename = f'{filename}_metadata.json'
    metadata_filepath = os.path.join(binary_path, metadata_filename)
    with open(metadata_filepath, "w") as outfile: 
        json.dump(metadata, outfile, indent=4)


def download_binaries(system_os: str, bin_type: str):
    """
    Download binary from release url
    Args:
        system_os: string, current system type
        bin_type: string of "OMP" or "CUDA"

    Returns:
        None

    """
    for url in url_dict[system_os][bin_type]:
        # Extract the file name from the GitHub release URL
        binary_version, filename = url.split("/")[-2:]

        logging.log(logging.INFO, f"Downloading {filename} to {binary_path}...")

        # Create the directory if it does not yet exist
        os.makedirs(binary_path, exist_ok=True)

        # Download the binary file
        try:
            binary_filepath = os.path.join(binary_path, filename)
            urllib.request.urlretrieve(url, binary_filepath)
            _record_binary_metadata(
                binary_version=binary_version,
                binary_filepath=binary_filepath,
                binary_url=url,
                filename=filename
            )
        except TimeoutError:
            logging.log(
                logging.WARN,
                f"Download of {filename} timed out. "
                "This can be due to slow internet connection. "
                "Partially downloaded files will be removed.",
            )
            try:
                os.remove(binary_path)
            except Exception:
                folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
                logging.warning(
                    "Error occurred while removing partially downloaded binary. "
                    f"Please manually delete the `{folder_path}` folder which "
                    "can be found in your virtual environment."
                )


def install_binaries():
    for binary_type in ["cpu", "cuda"]:
        download_binaries(system, binary_type)


if not binaries_present():
    install_binaries()
