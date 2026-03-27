"""Shared fixtures and helpers for CLI tests."""

import json

import pytest
from click.testing import CliRunner

from kwave.cli.main import cli


def invoke(runner, args, session_dir):
    """Invoke a CLI command and parse the final JSON response."""
    result = runner.invoke(cli, ["--session-dir", str(session_dir)] + args, catch_exceptions=False)
    assert result.exit_code == 0, f"Command failed: {args}\n{result.output}"
    output = result.output.strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass
    # For run command: progress events precede the final JSON response.
    # Find the last top-level JSON object.
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


@pytest.fixture
def session_dir(tmp_path):
    return tmp_path / "kwave_test_session"


@pytest.fixture
def runner():
    return CliRunner()
