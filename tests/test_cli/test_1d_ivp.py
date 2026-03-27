"""Test: replicate ivp_1D_simulation.py via CLI commands.

Exercises: 1D grid, heterogeneous medium (array .npy files),
custom p0, sparse sensor mask, custom CFL.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from kwave.cli.main import cli
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder


def _invoke(runner, args, session_dir):
    result = runner.invoke(cli, ["--session-dir", str(session_dir)] + args, catch_exceptions=False)
    assert result.exit_code == 0, f"Command failed: {args}\n{result.output}"
    output = result.output.strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass
    # For run command: find the last top-level JSON object
    depth = 0
    last_start = None
    for i, ch in enumerate(output):
        if ch == "{" and depth == 0:
            last_start = i
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if last_start is not None:
        return json.loads(output[last_start:])
    raise ValueError(f"Could not parse JSON from output: {output[:200]}")


# --- Build the 1D IVP arrays (same as ivp_1D_simulation.py) ---

Nx = 512
dx = 0.05e-3


def _make_sound_speed():
    c = 1500 * np.ones(Nx)
    c[: Nx // 3] = 2000
    return c


def _make_density():
    rho = 1000 * np.ones(Nx)
    rho[4 * Nx // 5 :] = 1500
    return rho


def _make_p0():
    p0 = np.zeros(Nx)
    x0, width = 280, 100
    pulse = 0.5 * (np.sin(np.arange(width + 1) * np.pi / width - np.pi / 2) + 1)
    p0[x0 : x0 + width + 1] = pulse
    return p0


def _make_sensor_mask():
    mask = np.zeros(Nx)
    mask[Nx // 4] = 1
    mask[3 * Nx // 4] = 1
    return mask


@pytest.fixture
def session_dir(tmp_path):
    return tmp_path / "kwave_test_session"


@pytest.fixture
def data_dir(tmp_path):
    """Directory for pre-built .npy files (simulating what an agent would prepare)."""
    d = tmp_path / "arrays"
    d.mkdir()
    np.save(d / "sound_speed.npy", _make_sound_speed())
    np.save(d / "density.npy", _make_density())
    np.save(d / "p0.npy", _make_p0())
    np.save(d / "sensor_mask.npy", _make_sensor_mask())
    return d


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI1DIVP:
    """Replicate ivp_1D_simulation.py end-to-end via CLI."""

    def test_cli_matches_python_api(self, runner, session_dir, data_dir):
        # -- CLI flow --
        _invoke(runner, ["session", "init"], session_dir)

        _invoke(
            runner,
            [
                "phantom",
                "load",
                "--grid-size",
                "512",
                "--spacing",
                "0.05e-3",
                "--sound-speed",
                str(data_dir / "sound_speed.npy"),
                "--density",
                str(data_dir / "density.npy"),
                "--cfl",
                "0.3",
            ],
            session_dir,
        )

        _invoke(
            runner,
            [
                "source",
                "define",
                "--type",
                "initial-pressure",
                "--p0-file",
                str(data_dir / "p0.npy"),
            ],
            session_dir,
        )

        _invoke(
            runner,
            [
                "sensor",
                "define",
                "--mask",
                str(data_dir / "sensor_mask.npy"),
                "--record",
                "p",
            ],
            session_dir,
        )

        plan_resp = _invoke(runner, ["plan"], session_dir)
        assert plan_resp["status"] == "ok"
        assert plan_resp["result"]["grid"]["N"] == [512]
        assert plan_resp["result"]["grid"]["Nt"] > 0

        run_resp = _invoke(runner, ["run"], session_dir)
        assert run_resp["status"] == "ok"

        # Load CLI results
        cli_p = np.load(run_resp["result"]["outputs"]["p"]["path"])

        # -- Direct Python API (the example) --
        sound_speed = _make_sound_speed()
        density = _make_density()
        kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))
        kgrid.makeTime(sound_speed, cfl=0.3)
        medium = kWaveMedium(sound_speed=sound_speed, density=density)
        source = kSource()
        source.p0 = _make_p0()
        sensor = kSensor(mask=_make_sensor_mask())
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", quiet=True)

        # -- Compare --
        assert cli_p.shape == result["p"].shape, f"Shape mismatch: {cli_p.shape} vs {result['p'].shape}"
        np.testing.assert_allclose(cli_p, result["p"], rtol=0, atol=0)
