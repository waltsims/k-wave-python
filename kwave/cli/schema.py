"""Response envelope and error types for the agent-first CLI."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Literal

import click

# Exit codes
EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_SESSION = 2
EXIT_SIMULATION = 3
EXIT_IO = 4


@dataclass
class CLIError:
    code: str
    field: str = ""
    value: Any = None
    constraint: str = ""
    suggestion: str = ""


@dataclass
class CLIResponse:
    status: Literal["ok", "error", "warning"] = "ok"
    step: str = ""
    result: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def json_command(step_name: str):
    """Decorator that wraps a Click command to always return the JSON envelope."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                resp = fn(*args, **kwargs)
                if not isinstance(resp, CLIResponse):
                    resp = CLIResponse(step=step_name, result=resp or {})
                resp.step = step_name
                click.echo(resp.to_json())
                sys.exit(EXIT_OK)
            except click.exceptions.Exit:
                raise
            except SystemExit:
                raise
            except ValidationError as e:
                resp = CLIResponse(
                    status="error",
                    step=step_name,
                    errors=[asdict(e.error)],
                )
                click.echo(resp.to_json())
                sys.exit(EXIT_VALIDATION)
            except SessionError as e:
                resp = CLIResponse(
                    status="error",
                    step=step_name,
                    errors=[asdict(CLIError(code="SESSION_ERROR", suggestion=str(e)))],
                )
                click.echo(resp.to_json())
                sys.exit(EXIT_SESSION)
            except Exception as e:
                resp = CLIResponse(
                    status="error",
                    step=step_name,
                    errors=[asdict(CLIError(code="UNEXPECTED_ERROR", suggestion=str(e)))],
                )
                click.echo(resp.to_json())
                sys.exit(EXIT_SIMULATION)

        return wrapper

    return decorator


class ValidationError(Exception):
    def __init__(self, error: CLIError):
        self.error = error
        super().__init__(error.code)


class SessionError(Exception):
    pass
