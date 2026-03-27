"""Plan command: derive full simulation config, validate, estimate cost."""

import click
import numpy as np

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIResponse, SessionError, json_command


@click.command("plan")
@pass_session
@json_command("plan")
def plan(sess):
    """Derive full simulation config and validate before running."""
    sess.load()

    # Check completeness
    comp = sess._completeness()
    missing = [k for k, v in comp.items() if not v]
    if missing:
        raise SessionError(f"Cannot plan: missing {', '.join(missing)}. Complete setup first.")

    kgrid = sess.make_grid()
    medium = sess.make_medium()

    g = sess.state["grid"]
    grid_n = tuple(g["N"])
    spacing = tuple(g["spacing"])
    ndim = len(grid_n)
    grid_points = int(np.prod(grid_n))

    # Time stepping
    dt = float(kgrid.dt)
    Nt = int(kgrid.Nt)

    # PPW check
    c_min = float(np.min(medium.sound_speed)) if hasattr(medium.sound_speed, "__len__") else float(medium.sound_speed)
    max_spacing = max(spacing)
    # For IVP problems, use grid-based wavelength estimate
    min_wavelength = c_min * dt * Nt / 2  # rough estimate
    ppw = c_min / (max_spacing * (1 / (2 * dt)))  # Nyquist-based PPW

    # CFL
    c_max = float(np.max(medium.sound_speed)) if hasattr(medium.sound_speed, "__len__") else float(medium.sound_speed)
    cfl = c_max * dt / min(spacing)

    # Memory estimate: ~(3 + 2*ndim) fields of float64
    n_fields = 3 + 2 * ndim
    memory_bytes = grid_points * n_fields * 8
    memory_mb = memory_bytes / (1024 * 1024)

    # Runtime estimate
    cost_per_point_step_ns = 50  # ~50ns per grid point per step on CPU
    estimated_runtime_s = grid_points * Nt * cost_per_point_step_ns / 1e9

    # PML
    pml_size = 20

    # Warnings
    warnings = []
    if ppw < 4:
        warnings.append(
            {
                "code": "LOW_PPW",
                "detail": f"PPW={ppw:.1f} is below recommended minimum of 4",
                "suggestion": "Increase grid resolution or reduce maximum frequency",
            }
        )
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
        "ppw": round(ppw, 2),
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
