import logging
import os
import sys
import urllib.request

# Get current system type
system = sys.platform


def download_binary(system: str, type: str):
    """
    Download binary from release url
    Args:
        system: string, current system type
        type: string of "OMP" or "CUDA"

    Returns:
        None

    """
    for url in url_dict[system][type]:
        dirname = f"kwave/bin/{system}/"

        # Extract the file name from the GitHub release URL
        filename = url.split("/")[-1]

        print(f"Downloading {filename}...")

        # Create the directory if it does not yet exist
        os.makedirs(dirname, exist_ok=True)

        # Download the binary file
        urllib.request.urlretrieve(url, f"{dirname}/{filename}")


def get_windows_release_urls(version, system_type):
    """
    Get the release urls for windows
    Args:
        version: string, "OMP" or "CUDA"

    Returns:
        None

    """
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


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

url_base = "https://github.com/waltsims/"

prefix = "https://github.com/waltsims/kspaceFirstOrder-{0}-{1}/releases/download/v1.3.0/"

common_filenames = ["cufft64_10.dll", "hdf5.dll", "hdf5_hl.dll", "libiomp5md.dll", "libmmd.dll",
                    "msvcp140.dll", "svml_dispmd.dll", "szip.dll", "vcruntime140.dll", "zlib.dll"]

specific_omp_filenames = ["kspaceFirstOrder-OMP.exe"]
specific_cuda_filenames = ["kspaceFirstOrder-CUDA.exe"]

# GitHub release URLs
url_dict = {
    "linux": {
        "cuda": [url_base + "kspaceFirstOrder-CUDA-linux/releases/download/v1.3/kspaceFirstOrder-CUDA"],
        "cpu": [url_base + "kspaceFirstOrder-OMP-linux/releases/download/v1.3.0/kspaceFirstOrder-OMP"],
    },
    # "darwin": {
    #     "cuda": url_base + "kspaceFirstOrder-CUDA-linux/releases/download/v1.3/kspaceFirstOrder-CUDA",
    #     "cpu": url_base + "kspaceFirstOrder-OMP-linux/releases/download/v1.3/kspaceFirstOrder-OMP",
    # },
    "win64": {
        "cuda": get_windows_release_urls("CUDA", "windows"),
        "cpu": get_windows_release_urls("OMP", "windows"),
    },
}

# Download CPU binary
download_binary(system, "cpu")

# Download the CUDA binary
download_binary(system, "cuda")
