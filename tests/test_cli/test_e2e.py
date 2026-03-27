"""End-to-end CLI test: replicates new_api_ivp_2D.py via CLI commands."""

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
from kwave.utils.mapgen import make_disc
from tests.test_cli.conftest import invoke


class TestSessionLifecycle:
    def test_init(self, runner, session_dir):
        resp = invoke(runner, ["session", "init"], session_dir)
        assert resp["status"] == "ok"
        assert "session_id" in resp["result"]

    def test_status_without_init_fails(self, runner, session_dir):
        result = runner.invoke(cli, ["--session-dir", str(session_dir), "session", "status"])
        assert result.exit_code != 0

    def test_reset(self, runner, session_dir):
        invoke(runner, ["session", "init"], session_dir)
        resp = invoke(runner, ["session", "reset"], session_dir)
        assert resp["result"]["reset"] is True


class TestPhantomGenerate:
    def test_disc_phantom(self, runner, session_dir):
        invoke(runner, ["session", "init"], session_dir)
        resp = invoke(
            runner,
            [
                "phantom",
                "generate",
                "--type",
                "disc",
                "--grid-size",
                "64,64",
                "--spacing",
                "0.1e-3",
                "--sound-speed",
                "1500",
                "--density",
                "1000",
                "--disc-radius",
                "5",
            ],
            session_dir,
        )
        assert resp["status"] == "ok"
        assert resp["result"]["grid_size"] == [64, 64]
        assert resp["result"]["p0_max"] == 1.0

    def test_disc_requires_2d(self, runner, session_dir):
        invoke(runner, ["session", "init"], session_dir)
        result = runner.invoke(
            cli,
            ["--session-dir", str(session_dir), "phantom", "generate", "--type", "disc", "--grid-size", "64,64,64", "--spacing", "0.1e-3"],
        )
        assert result.exit_code != 0
        resp = json.loads(result.output)
        assert resp["errors"][0]["code"] == "DISC_REQUIRES_2D"


class TestSensorDefine:
    def test_full_grid(self, runner, session_dir):
        invoke(runner, ["session", "init"], session_dir)
        resp = invoke(runner, ["sensor", "define", "--mask", "full-grid", "--record", "p,p_final"], session_dir)
        assert resp["result"]["mask_type"] == "full-grid"
        assert resp["result"]["record"] == ["p", "p_final"]


class TestPlan:
    def test_plan_incomplete_session(self, runner, session_dir):
        invoke(runner, ["session", "init"], session_dir)
        result = runner.invoke(cli, ["--session-dir", str(session_dir), "plan"])
        assert result.exit_code != 0

    def test_plan_complete_session(self, runner, session_dir):
        invoke(runner, ["session", "init"], session_dir)
        invoke(
            runner,
            [
                "phantom",
                "generate",
                "--type",
                "disc",
                "--grid-size",
                "64,64",
                "--spacing",
                "0.1e-3",
                "--sound-speed",
                "1500",
                "--density",
                "1000",
            ],
            session_dir,
        )
        invoke(runner, ["sensor", "define", "--mask", "full-grid", "--record", "p,p_final"], session_dir)
        resp = invoke(runner, ["plan"], session_dir)
        assert resp["status"] == "ok"
        assert resp["result"]["grid"]["Nt"] > 0
        assert resp["derived"]["cfl"] > 0


class TestEndToEnd:
    """Replicate new_api_ivp_2D.py via CLI and compare results."""

    N = 48  # small grid for fast CI (must be > 2*pml_size=40)

    def test_cli_matches_python_api(self, runner, session_dir):
        N = self.N
        invoke(runner, ["session", "init"], session_dir)
        invoke(
            runner,
            [
                "phantom",
                "generate",
                "--type",
                "disc",
                "--grid-size",
                f"{N},{N}",
                "--spacing",
                "0.1e-3",
                "--sound-speed",
                "1500",
                "--density",
                "1000",
                "--disc-center",
                f"{N // 2},{N // 2}",
                "--disc-radius",
                "3",
            ],
            session_dir,
        )
        invoke(runner, ["sensor", "define", "--mask", "full-grid", "--record", "p,p_final"], session_dir)
        resp = invoke(runner, ["run"], session_dir)
        assert resp["status"] == "ok"

        cli_p = np.load(resp["result"]["outputs"]["p"]["path"])
        cli_p_final = np.load(resp["result"]["outputs"]["p_final"]["path"])

        kgrid = kWaveGrid([N, N], [0.1e-3, 0.1e-3])
        kgrid.makeTime(1500)
        medium = kWaveMedium(sound_speed=1500, density=1000)
        source = kSource()
        source.p0 = make_disc(Vector([N, N]), Vector([N // 2, N // 2]), 3).astype(float)
        sensor = kSensor(mask=np.ones((N, N), dtype=bool))
        result = kspaceFirstOrder(kgrid, medium, source, sensor, quiet=True)

        assert cli_p.shape == result["p"].shape
        assert cli_p_final.shape == result["p_final"].shape
        np.testing.assert_allclose(cli_p, result["p"], rtol=0, atol=0)
        np.testing.assert_allclose(cli_p_final, result["p_final"], rtol=0, atol=0)
