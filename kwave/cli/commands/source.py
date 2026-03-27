"""Source definition command."""

import click
import numpy as np

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIResponse, json_command


@click.group("source")
def source():
    """Define simulation source."""
    pass


@source.command()
@click.option("--type", "source_type", required=True, type=click.Choice(["initial-pressure"]))
@click.option("--p0-file", required=True, type=click.Path(exists=True), help="Path to .npy file with initial pressure distribution")
@pass_session
@json_command("source.define")
def define(sess, source_type, p0_file):
    """Define source from file."""
    sess.load()

    p0 = np.load(p0_file)
    p0_path = sess.save_array("p0", p0)

    sess.update(
        "source",
        {
            "type": source_type,
            "p0_path": p0_path,
        },
    )

    return CLIResponse(
        result={
            "type": source_type,
            "p0_shape": list(p0.shape),
            "p0_max": float(p0.max()),
            "p0_min": float(p0.min()),
        }
    )
