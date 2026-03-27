"""Test: replicate ivp_1D_simulation.py via CLI commands.

Exercises: 1D grid, heterogeneous medium (array .npy files),
custom p0, sparse sensor mask, custom CFL.
"""

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from tests.test_cli.conftest import invoke

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
def data_dir(tmp_path):
    """Directory with pre-built .npy files (simulating agent-prepared arrays)."""
    d = tmp_path / "arrays"
    d.mkdir()
    np.save(d / "sound_speed.npy", _make_sound_speed())
    np.save(d / "density.npy", _make_density())
    np.save(d / "p0.npy", _make_p0())
    np.save(d / "sensor_mask.npy", _make_sensor_mask())
    return d


class TestCLI1DIVP:
    """Replicate ivp_1D_simulation.py end-to-end via CLI."""

    def test_cli_matches_python_api(self, runner, session_dir, data_dir):
        invoke(runner, ["session", "init"], session_dir)

        invoke(
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

        invoke(
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

        invoke(
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

        plan_resp = invoke(runner, ["plan"], session_dir)
        assert plan_resp["status"] == "ok"
        assert plan_resp["result"]["grid"]["N"] == [512]
        assert plan_resp["result"]["grid"]["Nt"] > 0

        run_resp = invoke(runner, ["run"], session_dir)
        assert run_resp["status"] == "ok"

        cli_p = np.load(run_resp["result"]["outputs"]["p"]["path"])

        # Direct Python API
        sound_speed = _make_sound_speed()
        kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))
        kgrid.makeTime(sound_speed, cfl=0.3)
        medium = kWaveMedium(sound_speed=sound_speed, density=_make_density())
        source = kSource()
        source.p0 = _make_p0()
        sensor = kSensor(mask=_make_sensor_mask())
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", quiet=True)

        assert cli_p.shape == result["p"].shape
        np.testing.assert_allclose(cli_p, result["p"], rtol=0, atol=0)
