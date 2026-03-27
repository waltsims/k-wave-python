"""Phantom generation and loading commands."""

import click
import numpy as np

from kwave.cli.main import pass_session
from kwave.cli.schema import CLIError, CLIResponse, ValidationError, json_command


@click.group("phantom")
def phantom():
    """Define the simulation phantom (medium + initial pressure)."""
    pass


@phantom.command("load")
@click.option("--grid-size", required=True, help="Grid dimensions, e.g. 512 or 128,128")
@click.option("--spacing", required=True, type=float, help="Grid spacing in meters")
@click.option("--sound-speed", required=True, help="Scalar value (m/s) or path to .npy file")
@click.option("--density", default=None, help="Scalar value (kg/m^3) or path to .npy file")
@click.option("--cfl", type=float, default=None, help="CFL number for time step calculation")
@pass_session
@json_command("phantom.load")
def load(sess, grid_size, spacing, sound_speed, density, cfl):
    """Load medium properties from scalar values or .npy files."""
    sess.load()

    grid_n = tuple(int(x) for x in grid_size.split(","))
    ndim = len(grid_n)
    grid_spacing = (spacing,) * ndim

    medium_state = {}

    # Sound speed: scalar or .npy path
    if sound_speed.endswith(".npy"):
        arr = np.load(sound_speed)
        path = sess.save_array("sound_speed", arr)
        medium_state["sound_speed_path"] = path
        medium_state["sound_speed_scalar"] = None
        # Store path so makeTime gets the full array (not just max)
        sound_speed_for_time_path = path
        sound_speed_for_time_scalar = None
    else:
        medium_state["sound_speed_scalar"] = float(sound_speed)
        medium_state["sound_speed_path"] = None
        sound_speed_for_time_path = None
        sound_speed_for_time_scalar = float(sound_speed)

    # Density: scalar, .npy path, or None
    if density is not None:
        if density.endswith(".npy"):
            arr = np.load(density)
            path = sess.save_array("density", arr)
            medium_state["density_path"] = path
            medium_state["density_scalar"] = None
        else:
            medium_state["density_scalar"] = float(density)
            medium_state["density_path"] = None

    grid_state = {
        "N": list(grid_n),
        "spacing": list(grid_spacing),
        "sound_speed_for_time_path": sound_speed_for_time_path,
        "sound_speed_for_time_scalar": sound_speed_for_time_scalar,
    }
    if cfl is not None:
        grid_state["cfl"] = cfl

    sess.update("grid", grid_state)
    sess.update("medium", medium_state)

    return CLIResponse(
        result={
            "grid_size": list(grid_n),
            "spacing": list(grid_spacing),
            "medium": medium_state,
        },
        derived={
            "ndim": ndim,
            "grid_points": int(np.prod(grid_n)),
        },
    )


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
            "sound_speed_for_time_scalar": sound_speed,
            "sound_speed_for_time_path": None,
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
