"""Sensor definition command."""

import click

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIResponse, json_command


@click.group("sensor")
def sensor():
    """Define sensor configuration."""
    pass


@sensor.command()
@click.option("--mask", required=True, help="Sensor mask: 'full-grid' or path to .npy file")
@click.option("--record", default="p,p_final", help="Comma-separated fields to record, e.g. p,p_final,ux")
@pass_session
@json_command("sensor.define")
def define(sess, mask, record):
    """Define what and where to record."""
    sess.load()

    record_fields = [r.strip() for r in record.split(",")]

    sensor_config = {"record": record_fields}
    if mask == "full-grid":
        sensor_config["mask_type"] = "full-grid"
    else:
        sensor_config["mask_type"] = "file"
        sensor_config["mask_path"] = mask

    sess.update("sensor", sensor_config)

    return CLIResponse(
        result={
            "mask_type": sensor_config["mask_type"],
            "record": record_fields,
        }
    )
