"""CUDA / GPU detection helpers.

Light-weight runtime introspection of the host's NVIDIA GPUs via
``nvidia-smi``.  Used by the C++ backend to warn users whose GPU
compute capability is below the minimum supported by the bundled
CUDA binary (currently 7.5, since CUDA Toolkit 13.0 dropped
Maxwell/Pascal/Volta).
"""

from __future__ import annotations

import shutil
import subprocess

_MIN_COMPUTE_CAPABILITY_CACHE: tuple[tuple[int, int] | None, bool] = (None, False)


def _query_nvidia_smi() -> list[tuple[int, int]] | None:
    """Run nvidia-smi and return per-GPU compute capabilities.

    Returns None if nvidia-smi cannot be located, fails, times out,
    or returns output we cannot parse.  Returns an empty list if
    nvidia-smi runs but reports no devices.
    """
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    caps: list[tuple[int, int]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            major_str, minor_str = line.split(".", 1)
            caps.append((int(major_str), int(minor_str)))
        except ValueError:
            # Unparseable line -> bail out entirely; we'd rather be silent
            # than warn based on a misread value.
            return None
    return caps


def get_min_compute_capability() -> tuple[int, int] | None:
    """Return the lowest (major, minor) compute capability across visible NVIDIA GPUs.

    Returns None if nvidia-smi is unavailable, returns no devices, or fails to parse.
    Result is cached after first call.
    """
    global _MIN_COMPUTE_CAPABILITY_CACHE
    cached_value, cached = _MIN_COMPUTE_CAPABILITY_CACHE
    if cached:
        return cached_value

    caps = _query_nvidia_smi()
    if not caps:
        # None (failure) or [] (no devices) both collapse to None.
        result: tuple[int, int] | None = None
    else:
        result = min(caps)

    _MIN_COMPUTE_CAPABILITY_CACHE = (result, True)
    return result


def _reset_compute_capability_cache() -> None:
    """Clear the cached compute-capability result.

    Intended for use in tests; not part of the public API.
    """
    global _MIN_COMPUTE_CAPABILITY_CACHE
    _MIN_COMPUTE_CAPABILITY_CACHE = (None, False)
