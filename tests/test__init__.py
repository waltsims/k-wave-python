import importlib
import json
import os
import stat
import sys
from unittest.mock import patch

import pytest


def test__init():
    with pytest.raises(NotImplementedError):
        with patch("platform.system", lambda: "Unknown"):
            import kwave

            importlib.reload(kwave)


def _seed_binary(binary_path, binary_name, url, *, mode=0o644):
    """Write a fake binary at the given path + matching metadata file."""
    binary_path.mkdir(parents=True, exist_ok=True)
    filepath = binary_path / binary_name
    filepath.write_bytes(b"fake-binary-payload")
    filepath.chmod(mode)
    import kwave

    metadata = {
        "url": url,
        "version": url.rsplit("/", 2)[-2],
        "file_hash": kwave._hash_file(str(filepath)),
    }
    (binary_path / f"{binary_name}_metadata.json").write_text(json.dumps(metadata))
    return filepath


@pytest.mark.skipif(sys.platform == "win32", reason="exec bit is meaningless on Windows")
def test_download_sets_executable_bit(tmp_path, monkeypatch):
    """Regression for #740 — urlretrieve creates files at 0644 and the C++
    backend fails with Permission denied (exit 126) when the executor invokes
    the binary."""
    import kwave

    monkeypatch.setattr(kwave, "BINARY_PATH", tmp_path)

    test_url = list(kwave.URL_DICT[kwave.PLATFORM].values())[0][0]
    filename = test_url.rsplit("/", 1)[-1]

    def fake_urlretrieve(url, dest):
        with open(dest, "wb") as f:
            f.write(b"fake-binary-payload")
        os.chmod(dest, 0o644)

    monkeypatch.setattr(kwave, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(kwave, "URL_DICT", {kwave.PLATFORM: {"test": [test_url]}})

    kwave.download_binaries(kwave.PLATFORM, "test")

    binary_filepath = tmp_path / filename
    assert binary_filepath.exists()
    mode = os.stat(binary_filepath).st_mode
    assert mode & stat.S_IXUSR, "owner exec bit not set after download"
    assert mode & stat.S_IXGRP, "group exec bit not set after download"
    assert mode & stat.S_IXOTH, "other exec bit not set after download"


@pytest.mark.skipif(sys.platform == "win32", reason="exec bit is meaningless on Windows")
def test_existing_non_executable_binary_is_healed(tmp_path, monkeypatch):
    """Regression for #740 — users with a cached non-executable binary from a
    pre-fix install would otherwise stay broken across upgrades because
    _is_binary_present skips re-download for valid cached binaries."""
    import kwave

    monkeypatch.setattr(kwave, "BINARY_PATH", tmp_path)

    test_url = list(kwave.URL_DICT[kwave.PLATFORM].values())[0][0]
    filename = test_url.rsplit("/", 1)[-1]
    binary_filepath = _seed_binary(tmp_path, filename, test_url, mode=0o644)
    pre_mode = os.stat(binary_filepath).st_mode
    assert not (pre_mode & stat.S_IXUSR), "test fixture should start without exec bit"

    monkeypatch.setattr(kwave, "URL_DICT", {kwave.PLATFORM: {"test": [test_url]}})

    assert kwave._is_binary_present(filename, "test") is True

    post_mode = os.stat(binary_filepath).st_mode
    assert post_mode & stat.S_IXUSR, "owner exec bit not healed on cache hit"
    assert post_mode & stat.S_IXGRP, "group exec bit not healed on cache hit"
    assert post_mode & stat.S_IXOTH, "other exec bit not healed on cache hit"


if __name__ == "__main__":
    pytest.main([__file__])
