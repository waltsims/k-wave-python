"""
Validation functions for kspaceFirstOrder().

Pure validation — no side effects, no modifications to inputs.
Raises ValueError/warnings for invalid simulation parameters.
"""
import warnings

import numpy as np


def validate_simulation(kgrid, medium, source, sensor, *, pml_size):
    """Run all validation checks before simulation dispatch."""
    validate_time_stepping(kgrid)
    validate_medium(medium, kgrid)
    validate_pml(pml_size, kgrid)
    validate_cfl(kgrid, medium)
    validate_source(source, kgrid)
    validate_sensor(sensor, kgrid)


def validate_time_stepping(kgrid):
    """Check that time stepping is configured."""
    if kgrid.Nt == "auto" or kgrid.dt == "auto":
        raise ValueError(
            "kgrid.Nt and kgrid.dt must be set before calling kspaceFirstOrder. "
            "Call kgrid.makeTime(medium.sound_speed) or kgrid.setTime(Nt, dt) first."
        )
    if int(kgrid.Nt) < 1:
        raise ValueError(f"kgrid.Nt must be >= 1, got {kgrid.Nt}")
    if float(kgrid.dt) <= 0:
        raise ValueError(f"kgrid.dt must be > 0, got {kgrid.dt}")


def validate_medium(medium, kgrid):
    """Check medium properties match grid dimensions."""
    c = np.atleast_1d(np.asarray(medium.sound_speed))
    grid_size = int(np.prod(kgrid.N))

    if c.size > 1 and c.size != grid_size:
        raise ValueError(
            f"medium.sound_speed has {c.size} elements but grid has {grid_size} points. " f"Must be scalar or match grid size."
        )
    if np.any(c <= 0):
        raise ValueError("medium.sound_speed must be positive everywhere.")

    if medium.density is not None:
        rho = np.atleast_1d(np.asarray(medium.density))
        if rho.size > 1 and rho.size != grid_size:
            raise ValueError(f"medium.density has {rho.size} elements but grid has {grid_size} points.")
        if np.any(rho <= 0):
            raise ValueError("medium.density must be positive everywhere.")

    if medium.alpha_coeff is not None:
        alpha = np.atleast_1d(np.asarray(medium.alpha_coeff))
        if np.any(alpha < 0):
            raise ValueError("medium.alpha_coeff must be non-negative.")
        if medium.alpha_power is not None:
            power = float(np.asarray(medium.alpha_power).flat[0])
            if power < 0 or power > 3:
                warnings.warn(
                    f"medium.alpha_power={power} is outside typical range [0, 3].",
                    stacklevel=3,
                )


def validate_pml(pml_size, kgrid):
    """Check PML sizes are valid relative to grid."""
    ndim = kgrid.dim
    for i in range(ndim):
        N = int(kgrid.N[i])
        pml = int(pml_size[i])
        if pml < 0:
            raise ValueError(f"pml_size[{i}] must be non-negative, got {pml}")
        if 2 * pml >= N:
            raise ValueError(f"pml_size[{i}]={pml} is too large for grid size N[{i}]={N} " f"(2*pml must be < N).")


def validate_cfl(kgrid, medium):
    """Warn if CFL condition suggests instability.

    Uses the pseudospectral stability limit: c*dt/dx <= 2/(pi*sqrt(ndim))
    (Tabei, Mast & Waag, JASA 2002, Eq. 11).
    """
    c_max = float(np.max(np.asarray(medium.sound_speed)))
    dt = float(kgrid.dt)
    dx_min = float(np.min(kgrid.spacing))
    ndim = kgrid.dim
    cfl = c_max * dt / dx_min
    cfl_limit = 2 / (np.pi * np.sqrt(ndim))
    if cfl > cfl_limit:
        warnings.warn(
            f"CFL number = {cfl:.3f} exceeds pseudospectral stability limit "
            f"{cfl_limit:.3f} for {ndim}D. Simulation may be unstable. "
            f"Reduce dt or increase grid spacing.",
            stacklevel=3,
        )


def validate_source(source, kgrid):
    """Check source masks and data shapes."""
    grid_size = int(np.prod(kgrid.N))

    if source.p0 is not None:
        p0 = np.asarray(source.p0)
        if p0.size != grid_size and p0.size != 0:
            raise ValueError(f"source.p0 has {p0.size} elements but grid has {grid_size} points.")

    if source.p is not None and source.p_mask is None:
        raise ValueError("source.p requires source.p_mask to be set.")

    if source.p_mask is not None:
        mask = np.asarray(source.p_mask)
        if mask.size != grid_size:
            raise ValueError(f"source.p_mask has {mask.size} elements but grid has {grid_size} points.")

    for vel in ["ux", "uy", "uz"]:
        if getattr(source, vel, None) is not None and source.u_mask is None:
            raise ValueError(f"source.{vel} requires source.u_mask to be set.")

    if source.u_mask is not None:
        mask = np.asarray(source.u_mask)
        if mask.size != grid_size:
            raise ValueError(f"source.u_mask has {mask.size} elements but grid has {grid_size} points.")


def validate_sensor(sensor, kgrid):
    """Check sensor mask shape."""
    if sensor is None:
        return

    if hasattr(sensor, "mask") and sensor.mask is not None:
        mask = np.asarray(sensor.mask)
        grid_size = int(np.prod(kgrid.N))
        if mask.size != grid_size:
            raise ValueError(f"sensor.mask has {mask.size} elements but grid has {grid_size} points.")
