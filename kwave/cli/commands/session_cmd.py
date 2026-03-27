"""Session management commands."""

import click

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIResponse, json_command


@click.group("session")
def session():
    """Manage simulation session."""
    pass


@session.command()
@pass_session
@json_command("session.init")
def init(sess):
    """Create a new session."""
    info = sess.init()
    return CLIResponse(result=info)


@session.command()
@pass_session
@json_command("session.status")
def status(sess):
    """Return full current session state."""
    sess.load()
    return CLIResponse(result=sess.status())


@session.command()
@pass_session
@json_command("session.reset")
def reset(sess):
    """Clear session state."""
    info = sess.reset()
    return CLIResponse(result=info)
