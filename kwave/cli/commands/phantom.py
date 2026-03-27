"""Phantom generation and loading commands."""

import click
import numpy as np

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIError, CLIResponse, ValidationError, json_command


@click.group("phantom")
def phantom():
    """Define the simulation phantom (medium + initial pressure)."""
    pass


@phantom.command()
@click.option("--type", "phantom_type", required=True, type=click.Choice(["disc", "spherical", "layered"]))
@click.option("--grid-size", required=True, help="Grid dimensions, e.g. 128,128")
@click.option("--spacing", required=True, type=float, help="Grid spacing in meters, e.g. 0.1e-3")
@click.option("--sound-speed", type=float, default=1500, help="Medium sound speed (m/s)")
@click.option("--density", type=float, default=1000, help="Medium density (kg/m^3)")
@click.option("--disc-center", default=None, help="Disc center, e.g. 64,64")
@click.option("--disc-radius", type=int, default=5, help="Disc radius in grid points")
@pass_session
@json_command("phantom.generate")
def generate(sess, phantom_type, grid_size, spacing, sound_speed, density, disc_center, disc_radius):
    """Generate an analytical phantom."""
    sess.load()

    grid_n = tuple(int(x) for x in grid_size.split(","))
    ndim = len(grid_n)
    grid_spacing = (spacing,) * ndim

    if phantom_type == "disc":
        if ndim != 2:
            raise ValidationError(
                CLIError(
                    code="DISC_REQUIRES_2D",
                    field="grid_size",
                    value=grid_size,
                    constraint="disc phantom requires 2D grid",
                    suggestion="Use --grid-size Nx,Ny (two dimensions)",
                )
            )
        from kwave.data import Vector
        from kwave.utils.mapgen import make_disc

        if disc_center is None:
            center = Vector([n // 2 for n in grid_n])
        else:
            center = Vector([int(x) for x in disc_center.split(",")])

        p0 = make_disc(Vector(list(grid_n)), center, disc_radius).astype(float)

    elif phantom_type == "spherical":
        # Simple spherical inclusion centered in grid
        center = np.array([n // 2 for n in grid_n])
        coords = np.mgrid[tuple(slice(0, n) for n in grid_n)]
        dist = np.sqrt(sum((c - cn) ** 2 for c, cn in zip(coords, center)))
        p0 = (dist <= disc_radius).astype(float)

    elif phantom_type == "layered":
        p0 = np.zeros(grid_n)
        layer_pos = grid_n[0] // 4
        p0[layer_pos, ...] = 1.0

    # Save arrays
    p0_path = sess.save_array("p0", p0)

    # Update session
    sess.update(
        "grid",
        {
            "N": list(grid_n),
            "spacing": list(grid_spacing),
            "sound_speed_for_time": sound_speed,
        },
    )
    sess.update(
        "medium",
        {
            "sound_speed": sound_speed,
            "density": density,
        },
    )
    sess.update(
        "source",
        {
            "type": "initial-pressure",
            "p0_path": p0_path,
        },
    )

    return CLIResponse(
        result={
            "phantom_type": phantom_type,
            "grid_size": list(grid_n),
            "spacing": list(grid_spacing),
            "p0_shape": list(p0.shape),
            "p0_max": float(p0.max()),
            "sound_speed": sound_speed,
            "density": density,
        },
        derived={
            "ndim": ndim,
            "grid_points": int(np.prod(grid_n)),
        },
    )
