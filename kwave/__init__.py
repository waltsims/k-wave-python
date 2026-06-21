import hashlib
import json
import logging
import os
import platform
import stat
import warnings
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

# Test installation with:
# python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple/ k-Wave-python==0.3.0
__version__ = "0.6.3"

# Constants and Configurations
URL_BASE = "https://github.com/waltsims/"
BINARY_VERSION = "v1.4.2"
# Single unified release hosts every platform binary + Windows runtime DLL
# (consolidated from 5 mirror repos in v1.4.2; see kspacefirstorder-unified#13).
# One version pin, one set of assets, one source-tree SHA. CUDA binary covers
# compute capability 7.5+ (Turing through every Blackwell variant: B200/GB200,
# B300/GB300, Jetson Thor, RTX 50xx, RTX PRO 6000 Blackwell, GB10/DGX Spark).
_UNIFIED_RELEASE_URL = f"{URL_BASE}kspacefirstorder-unified/releases/download/{BINARY_VERSION}/"
PLATFORM = platform.system().lower()

if PLATFORM not in ["linux", "windows", "darwin"]:
    raise NotImplementedError(f"k-wave-python is currently unsupported on this operating system: {PLATFORM}.")

# darwin C++ binary is arm64-only; universal2 coverage tracked for v0.6.5
DARWIN_BINARY_ARCH = "arm64"
_darwin_unsupported = PLATFORM == "darwin" and platform.machine() != DARWIN_BINARY_ARCH
if _darwin_unsupported:
    warnings.warn(
        f"k-wave-python's macOS C++ binary is {DARWIN_BINARY_ARCH}-only. "
        f"Detected {platform.machine()} — the C++ backend (backend='cpp') will not run on this machine. "
        "Use backend='python' instead. Universal2 (Intel + Apple Silicon) coverage is tracked for v0.6.5.",
        RuntimeWarning,
        stacklevel=2,
    )

# TODO: install directly in to /bin/ directory system directory is no longer needed
# TODO: deprecate in 0.5.0
BINARY_PATH = Path(__file__).parent / "bin" / PLATFORM
BINARY_DIR = BINARY_PATH  # add alias for BINARY_PATH for now


# Windows runtime DLLs shipped alongside both .exe files in the unified v1.4.2
# release. The full bundle is downloaded for either backend selection because we
# don't know at install time which the user will invoke. Verified against the
# v1.4.2 release asset manifest (21 DLLs).
WINDOWS_DLLS = [
    # CUDA runtime (CUDA 13.0 — used by the CUDA backend)
    "cudart64_13.dll",
    "cufft64_12.dll",
    # FFTW3 (used by the OMP backend)
    "fftw3.dll",
    "fftw3f.dll",
    "fftw3l.dll",
    # HDF5 + szip + zlib (from vcpkg; used by both backends)
    "aec.dll",
    "hdf5.dll",
    "hdf5_hl.dll",
    "szip.dll",
    "zlib1.dll",
    # OpenMP runtime (used by the OMP backend)
    "vcomp140.dll",
    # MSVC CRT (Concurrency Runtime + C++ stdlib + C runtime)
    "concrt140.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vccorlib140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
]

EXECUTABLE_PREFIX = "kspaceFirstOrder-"
ARCHITECTURES = ["omp", "cuda"]


def _platform_binary_url(architecture: str) -> list:
    """Return the URL list for a given backend on the current platform."""
    if PLATFORM == "darwin":
        if _darwin_unsupported or architecture == "cuda":
            return []
        return [_UNIFIED_RELEASE_URL + f"{EXECUTABLE_PREFIX}OMP-darwin"]
    if PLATFORM == "linux":
        suffix = "CUDA-linux" if architecture == "cuda" else "OMP-linux"
        return [_UNIFIED_RELEASE_URL + EXECUTABLE_PREFIX + suffix]
    # Windows: the .exe plus the shared runtime DLL bundle
    exe_suffix = "CUDA-windows.exe" if architecture == "cuda" else "OMP-windows.exe"
    return [_UNIFIED_RELEASE_URL + EXECUTABLE_PREFIX + exe_suffix] + [_UNIFIED_RELEASE_URL + dll for dll in WINDOWS_DLLS]


URL_DICT = {os: {arch: _platform_binary_url(arch) for arch in ARCHITECTURES} for os in ["linux", "darwin", "windows"]}


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


def _ensure_executable(binary_filepath) -> None:
    # Self-heal the executable bit on Linux/macOS. urlretrieve creates files
    # at 0644, and prior versions of this package didn't fix that up, so users
    # upgrading with a cached non-executable binary on disk would otherwise
    # stay stuck (the cache check below returns True and skips re-download).
    # Any OS-level failure here (broken symlink, read-only FS, wrong ownership,
    # TOCTOU race) is degraded to a warning so it never aborts `import kwave`.
    if PLATFORM == "windows":
        return
    try:
        current_mode = os.stat(binary_filepath).st_mode
        desired_mode = current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if current_mode == desired_mode:
            return
        os.chmod(binary_filepath, desired_mode)
    except OSError:  # pragma: no cover - defensive; degrades to warning, never fatal
        # Don't abort import. The user can chmod +x manually or reinstall
        # into a writable location.
        logging.warning(
            "kwave: cannot set executable bit on %s — backend='cpp' may fail with "
            "Permission denied. Run `chmod +x` manually or reinstall.",
            binary_filepath,
        )


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

    _ensure_executable(binary_filepath)

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
            _ensure_executable(binary_filepath)
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
