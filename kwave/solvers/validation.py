import warnings

import numpy as np


def validate_simulation(kgrid, medium, source, sensor, *, pml_size):
    validate_time_stepping(kgrid)
    validate_medium(medium, kgrid)
    validate_pml(pml_size, kgrid)
    validate_cfl(kgrid, medium)
    validate_source(source, kgrid)
    validate_sensor(sensor, kgrid)


def validate_time_stepping(kgrid):
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
    c = np.atleast_1d(np.asarray(medium.sound_speed))
    grid_size = int(np.prod(kgrid.N))
    if c.size > 1 and c.size != grid_size:
        raise ValueError(f"medium.sound_speed has {c.size} elements but grid has {grid_size} points.")
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
                warnings.warn(f"medium.alpha_power={power} is outside typical range [0, 3].", stacklevel=3)


def validate_pml(pml_size, kgrid):
    for i in range(kgrid.dim):
        N, pml = int(kgrid.N[i]), int(pml_size[i])
        if pml < 0:
            raise ValueError(f"pml_size[{i}] must be non-negative, got {pml}")
        if 2 * pml >= N:
            raise ValueError(f"pml_size[{i}]={pml} is too large for grid size N[{i}]={N} (2*pml must be < N).")


def validate_cfl(kgrid, medium):
    """Warn if CFL condition suggests instability (pseudospectral limit: Tabei et al., JASA 2002, Eq. 11)."""
    cfl = float(np.max(np.asarray(medium.sound_speed))) * float(kgrid.dt) / float(np.min(kgrid.spacing))
    cfl_limit = 2 / (np.pi * np.sqrt(kgrid.dim))
    if cfl > cfl_limit:
        warnings.warn(
            f"CFL number = {cfl:.3f} exceeds pseudospectral stability limit "
            f"{cfl_limit:.3f} for {kgrid.dim}D. Reduce dt or increase grid spacing.",
            stacklevel=3,
        )


def validate_source(source, kgrid):
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
    if sensor is None or not hasattr(sensor, "mask") or sensor.mask is None:
        return
    mask = np.asarray(sensor.mask)
    grid_size = int(np.prod(kgrid.N))
    # Cartesian mask: shape (ndim, N_pts) — skip binary-size check
    if not (mask.ndim == 2 and mask.shape[0] == kgrid.dim) and mask.size != grid_size:
        raise ValueError(f"sensor.mask has {mask.size} elements but grid has {grid_size} points.")
