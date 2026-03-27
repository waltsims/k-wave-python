"""Agent-first CLI for k-Wave simulations. All commands return structured JSON."""

from pathlib import Path
from typing import Optional

import click

from kwave.cli.session import Session

pass_session = click.make_pass_decorator(Session, ensure=True)


@click.group()
@click.option("--session-dir", type=click.Path(), default=None, envvar="KWAVE_SESSION_DIR", help="Session directory (default: ~/.kwave)")
@click.pass_context
def cli(ctx, session_dir):
    """k-Wave agent-first CLI. All commands return structured JSON."""
    base_dir = Path(session_dir) if session_dir else None
    ctx.obj = Session(base_dir=base_dir)


# Register command groups
from kwave.cli.commands.phantom import phantom  # noqa: E402
from kwave.cli.commands.plan import plan  # noqa: E402
from kwave.cli.commands.run import run  # noqa: E402
from kwave.cli.commands.sensor import sensor  # noqa: E402
from kwave.cli.commands.session_cmd import session  # noqa: E402

cli.add_command(session)
cli.add_command(phantom)
cli.add_command(sensor)
cli.add_command(plan)
cli.add_command(run)
