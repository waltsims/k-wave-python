import importlib
import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test__init():
    import kwave

    try:
        with pytest.raises(NotImplementedError):
            with patch("platform.system", lambda: "Unknown"):
                importlib.reload(kwave)
    finally:
        # The failed reload above left kwave.PLATFORM = "unknown" in module
        # state (line where PLATFORM is set ran before the NotImplementedError).
        # Reload once more without the patch so subsequent tests see a valid
        # PLATFORM / URL_DICT pair.
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

    test_url = next(urls[0] for urls in kwave.URL_DICT[kwave.PLATFORM].values() if urls)
    created = {}

    def fake_urlretrieve(url, dest):
        # Capture the actual destination chosen by download_binaries instead
        # of re-deriving it from the URL — the unified release v1.4.2+ tags
        # assets with a platform suffix (e.g. -linux) that the package
        # strips when computing the local install filename, so URL-tail
        # inference is no longer correct (and never should have been).
        created["dest"] = dest
        with open(dest, "wb") as f:
            f.write(b"fake-binary-payload")
        os.chmod(dest, 0o644)

    monkeypatch.setattr(kwave, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(kwave, "URL_DICT", {kwave.PLATFORM: {"test": [test_url]}})

    kwave.download_binaries(kwave.PLATFORM, "test")

    binary_filepath = Path(created["dest"])
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

    test_url = next(urls[0] for urls in kwave.URL_DICT[kwave.PLATFORM].values() if urls)
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


def test_darwin_gpu_urls_point_to_mlx_binary_and_metallib():
    """macOS has no CUDA: its GPU backend is the MLX binary plus the Metal kernel library it loads at run time."""
    import kwave

    mlx_urls = kwave._platform_binary_url("darwin", "mlx")
    if kwave.PLATFORM == "darwin" and kwave._darwin_unsupported:
        assert mlx_urls == []
        return
    assert [url.rsplit("/", 1)[-1] for url in mlx_urls] == ["kspaceFirstOrder-MLX-darwin", "mlx.metallib"]
    assert all(url.startswith(kwave._UNIFIED_RELEASE_URL) for url in mlx_urls)
    assert kwave._platform_binary_url("darwin", "cuda") == []
    assert kwave._platform_binary_url("linux", "mlx") == []
    assert kwave._platform_binary_url("windows", "mlx") == []


def test_local_filename_strips_platform_suffix_for_mlx():
    import kwave

    assert kwave._local_filename("kspaceFirstOrder-MLX-darwin") == "kspaceFirstOrder-MLX"
    assert kwave._local_filename("mlx.metallib") == "mlx.metallib"


def test_failed_optional_mlx_download_does_not_abort(tmp_path, monkeypatch):
    """A release without the MLX assets must not break `import kwave` (the OMP backend stays usable)."""
    from urllib.error import HTTPError

    import kwave

    monkeypatch.setattr(kwave, "BINARY_PATH", tmp_path)

    def fake_urlretrieve(url, dest):
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(kwave, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(kwave, "URL_DICT", {kwave.PLATFORM: {"mlx": ["https://example.invalid/kspaceFirstOrder-MLX-darwin"]}})
    kwave.download_binaries(kwave.PLATFORM, "mlx")  # must not raise

    monkeypatch.setattr(kwave, "URL_DICT", {kwave.PLATFORM: {"omp": ["https://example.invalid/kspaceFirstOrder-OMP-darwin"]}})
    with pytest.raises(HTTPError):
        kwave.download_binaries(kwave.PLATFORM, "omp")


def _fake_release(tmp_path, monkeypatch, fail_assets):
    """Point kwave at a fake release with an OMP binary and a two-file MLX bundle; assets in fail_assets 404."""
    from urllib.error import HTTPError

    import kwave

    base = "https://example.invalid/releases/download/v9.9.9/"
    url_dict = {
        kwave.PLATFORM: {
            "omp": [base + "kspaceFirstOrder-OMP-darwin"],
            "mlx": [base + "kspaceFirstOrder-MLX-darwin", base + "mlx.metallib"],
        }
    }
    downloads = []

    def fake_urlretrieve(url, dest):
        downloads.append(url.rsplit("/", 1)[-1])
        if url.rsplit("/", 1)[-1] in fail_assets:
            raise HTTPError(url, 404, "Not Found", None, None)
        Path(dest).write_bytes(b"payload-" + url.encode())

    monkeypatch.setattr(kwave, "BINARY_PATH", tmp_path)
    monkeypatch.setattr(kwave, "URL_DICT", url_dict)
    monkeypatch.setattr(kwave, "ARCHITECTURES", ["omp", "mlx"])
    monkeypatch.setattr(kwave, "urlretrieve", fake_urlretrieve)
    return downloads


def test_cached_omp_is_not_downloaded_again_when_mlx_is_missing(tmp_path, monkeypatch):
    """A missing optional MLX bundle must not re-download (or require the network for) a valid cached OMP binary."""
    import kwave

    downloads = _fake_release(tmp_path, monkeypatch, fail_assets={"kspaceFirstOrder-MLX-darwin", "mlx.metallib"})
    kwave.install_binaries()
    assert (tmp_path / "kspaceFirstOrder-OMP").exists()
    assert not kwave.binaries_present()  # MLX still missing

    downloads.clear()
    kwave.install_binaries()  # second import: OMP is cached and valid
    assert "kspaceFirstOrder-OMP-darwin" not in downloads
    assert downloads == ["kspaceFirstOrder-MLX-darwin"]


def test_incomplete_mlx_bundle_is_removed(tmp_path, monkeypatch):
    """If mlx.metallib fails after the MLX executable was downloaded, the unusable executable must not stay installed."""
    import kwave

    _fake_release(tmp_path, monkeypatch, fail_assets={"mlx.metallib"})
    kwave.download_binaries(kwave.PLATFORM, "mlx")
    assert not (tmp_path / "kspaceFirstOrder-MLX").exists()
    assert not (tmp_path / "kspaceFirstOrder-MLX_metadata.json").exists()
    assert not (tmp_path / "mlx.metallib").exists()


if __name__ == "__main__":
    pytest.main([__file__])
