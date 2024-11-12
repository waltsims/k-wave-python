import logging
import os
import platform
from urllib.request import urlretrieve
from pathlib import Path
from typing import List
import hashlib
import json

# Test installation with:
# python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple/ k-Wave-python==0.3.0
VERSION = "0.3.5"

# Constants and Configurations
URL_BASE = "https://github.com/waltsims/"
BINARY_VERSION = "v1.3.0"
PREFIX = f"{URL_BASE}kspaceFirstOrder-{{}}-{{}}/releases/download/{BINARY_VERSION}/"
PLATFORM = platform.system().lower()

if PLATFORM not in ["linux", "windows", "darwin"]:
    raise NotImplementedError(f"k-wave-python is currently unsupported on this operating system: {PLATFORM}.")

# TODO: install directly in to /bin/ directory system directory is no longer needed
BINARY_PATH = Path(__file__).parent / "bin" / PLATFORM


WINDOWS_DLLS = [
    "cufft64_10.dll",
    "hdf5.dll",
    "hdf5_hl.dll",
    "libiomp5md.dll",
    "libmmd.dll",
    "msvcp140.dll",
    "svml_dispmd.dll",
    "szip.dll",
    "vcruntime140.dll",
    "zlib.dll",
]

EXECUTABLE_PREFIX = "kspaceFirstOrder-"
ARCHITECTURES = ["omp", "cuda"]


def get_windows_release_urls(architecture: str) -> list:
    specific_filenames = [EXECUTABLE_PREFIX + architecture + ".exe"]
    if architecture == "omp":
        specific_filenames += WINDOWS_DLLS
    release_urls = [PREFIX.format(architecture.upper(), PLATFORM.lower()) + filename for filename in specific_filenames]
    return release_urls


URL_DICT = {
    "linux": {
        "cuda": [URL_BASE + f"kspaceFirstOrder-CUDA-{PLATFORM}/releases/download/v1.3.1/{EXECUTABLE_PREFIX}CUDA"],
        "omp": [URL_BASE + f"kspaceFirstOrder-OMP-{PLATFORM}/releases/download/{BINARY_VERSION}/{EXECUTABLE_PREFIX}OMP"],
    },
    "darwin": {
        "cuda": [],
        "omp": [URL_BASE + f"k-wave-omp-{PLATFORM}/releases/download/v0.3.0rc2/{EXECUTABLE_PREFIX}OMP"],
    },
    "windows": {architecture: get_windows_release_urls(architecture) for architecture in ARCHITECTURES},
}


def _hash_file(filepath: str) -> str:
    buf_size = 65536  # 64kb chunks
    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def _is_binary_present(binary_name: str, binary_type: str) -> bool:
    binary_filepath = BINARY_PATH / binary_name
    binary_file_exists = os.path.exists(binary_filepath)
    if not binary_file_exists:
        return False

    if binary_type is None:
        # this is non-kwave windows binary
        # it already exists according to the check above
        return True
    existing_metadata_path = BINARY_PATH / f"{binary_name}_metadata.json"

    if not os.path.exists(existing_metadata_path):
        # metadata does not exist => binaries may or may not exist
        # Let's play safe and claim they don't exist
        # This will trigger binary download and generation of binary metadata
        return False
    existing_metadata = json.loads(Path(existing_metadata_path).read_text())

    # If metadata was somehow corrupted
    file_hash = _hash_file(binary_filepath)
    if existing_metadata["file_hash"] != file_hash:
        return False

    # If there is a new binary
    latest_urls = URL_DICT[PLATFORM][binary_type]
    if existing_metadata["url"] not in latest_urls:
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
    binary_list = []
    for binary_type in ARCHITECTURES:
        for binary_name in URL_DICT[PLATFORM][binary_type]:
            binary_list.append((binary_name.split("/")[-1], binary_type))

    missing_binaries: List[str] = []

    for binary_name, binary_type in binary_list:
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
    metadata = {"url": binary_url, "version": binary_version, "file_hash": _hash_file(binary_filepath)}
    metadata_filename = f"{filename}_metadata.json"
    metadata_filepath = BINARY_PATH / metadata_filename
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
    for url in URL_DICT[system_os][bin_type]:
        # Extract the file name from the GitHub release URL
        binary_version, filename = url.split("/")[-2:]

        logging.log(logging.INFO, f"Downloading {filename} to {BINARY_PATH}...")

        # Create the directory if it does not yet exist
        os.makedirs(BINARY_PATH, exist_ok=True)

        # Download the binary file
        try:
            binary_filepath = os.path.join(BINARY_PATH, filename)
            urlretrieve(url, binary_filepath)
            _record_binary_metadata(binary_version=binary_version, binary_filepath=binary_filepath, binary_url=url, filename=filename)

        except TimeoutError:
            logging.log(
                logging.WARN,
                f"Download of {filename} timed out. "
                "This can be due to slow internet connection. "
                "Partially downloaded files will be removed.",
            )
            try:
                os.remove(BINARY_PATH)
            except Exception:
                folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
                logging.warning(
                    "Error occurred while removing partially downloaded binary. "
                    f"Please manually delete the `{folder_path}` folder which "
                    "can be found in your virtual environment."
                )


def install_binaries():
    for binary_type in ARCHITECTURES:
        download_binaries(PLATFORM, binary_type)


if not binaries_present():
    install_binaries()
