"""Plan command: derive full simulation config, validate, estimate cost."""

import click
import numpy as np

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIResponse, json_command


@click.command("plan")
@pass_session
@json_command("plan")
def plan(sess):
    """Derive full simulation config and validate before running."""
    sess.load()
    sess.assert_ready("plan")

    kgrid = sess.make_grid()
    medium = sess.make_medium()

    grid_n = tuple(int(n) for n in kgrid.N)
    spacing = tuple(float(d) for d in kgrid.spacing)
    ndim = len(grid_n)
    grid_points = int(np.prod(grid_n))
    dt = float(kgrid.dt)
    Nt = int(kgrid.Nt)

    c_max = float(np.max(medium.sound_speed)) if hasattr(medium.sound_speed, "__len__") else float(medium.sound_speed)
    c_min = float(np.min(medium.sound_speed)) if hasattr(medium.sound_speed, "__len__") else float(medium.sound_speed)
    cfl = c_max * dt / min(spacing)

    n_fields = 3 + 2 * ndim
    memory_mb = grid_points * n_fields * 8 / (1024 * 1024)
    estimated_runtime_s = grid_points * Nt * 50e-9  # ~50ns per grid point per step on CPU

    pml_size = 20

    warnings = []
    if cfl > 0.5:
        warnings.append(
            {
                "code": "HIGH_CFL",
                "detail": f"CFL={cfl:.3f} exceeds 0.5, simulation may be unstable",
                "suggestion": "Reduce time step or increase grid spacing",
            }
        )

    result = {
        "grid": {
            "N": list(grid_n),
            "spacing": list(spacing),
            "ndim": ndim,
            "dt": dt,
            "Nt": Nt,
        },
        "pml": {"size": pml_size},
        "medium": {
            "sound_speed": c_min if c_min == c_max else f"{c_min}-{c_max}",
        },
        "source": sess.state["source"],
        "sensor": sess.state["sensor"],
        "backend": "python",
        "device": "cpu",
    }

    derived = {
        "cfl": round(cfl, 4),
        "grid_points": grid_points,
        "estimated_memory_mb": round(memory_mb, 1),
        "estimated_runtime_s": round(estimated_runtime_s, 1),
    }

    return CLIResponse(
        result=result,
        derived=derived,
        warnings=warnings,
    )
