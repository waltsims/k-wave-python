"""Run command: execute simulation with structured JSON progress."""

import json
import time

import click
import numpy as np

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIResponse, json_command


def _emit_event(event: dict):
    """Write a JSON event to stdout and flush."""
    click.echo(json.dumps(event, default=str))


@click.command("run")
@click.option("--backend", default="python", type=click.Choice(["python", "cpp"]))
@click.option("--device", default="cpu", type=click.Choice(["cpu", "gpu"]))
@pass_session
@json_command("run")
def run(sess, backend, device):
    """Execute the simulation."""
    sess.load()
    sess.assert_ready("run")

    kgrid = sess.make_grid()
    medium = sess.make_medium()
    source = sess.make_source()
    sensor = sess.make_sensor()

    Nt = int(kgrid.Nt)

    _emit_event({"event": "started", "backend": backend, "device": device, "Nt": Nt})

    t_start = time.time()
    last_pct = -5  # emit at most every 5%

    def progress_callback(step, total):
        nonlocal last_pct
        pct = round(100 * step / total, 1)
        if pct - last_pct >= 5 or step == total:
            last_pct = pct
            _emit_event(
                {
                    "event": "progress",
                    "step": step,
                    "total": total,
                    "pct": pct,
                    "elapsed_s": round(time.time() - t_start, 2),
                }
            )

    from kwave.kspaceFirstOrder import kspaceFirstOrder

    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=True,
        progress_callback=progress_callback,
    )

    elapsed = round(time.time() - t_start, 2)

    # Save results
    result_info = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            path = sess.save_array(f"result_{key}", val)
            result_info[key] = {"shape": list(val.shape), "path": path}
        else:
            result_info[key] = val

    sess.update("result_path", str(sess.data_dir))

    _emit_event({"event": "completed", "elapsed_s": elapsed, "output_keys": list(result.keys())})

    return CLIResponse(
        result={
            "elapsed_s": elapsed,
            "outputs": result_info,
        },
    )
