import logging
import math

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.ksource import kSource
from kwave.utils.dotdictionary import dotdict


def scale_source_terms_func(
    c0, dt, kgrid: kWaveGrid, source, p_source_pos_index, s_source_pos_index, u_source_pos_index, transducer_input_signal, flags: dotdict
):
    """
    Subscript for the first-order k-Wave simulation functions to scale source terms to the correct units.
    Args:
    Returns:

    """
    dx, dy, dz = (
        kgrid.dx,
        kgrid.dy,
        kgrid.dz,
    )

    # get the dimension size
    N = kgrid.dim

    if not check_conditions(flags.nonuniform_grid, flags.source_uy, flags.source_uz, flags.transducer_source):
        return

    # =========================================================================
    # PRESSURE SOURCES
    # =========================================================================

    apply_pressure_source_correction(flags.source_p, flags.use_w_source_correction_p, source, dt)

    scale_pressure_source(flags.source_p, source, kgrid, N, c0, dx, dt, p_source_pos_index, flags.nonuniform_grid)

    # =========================================================================
    # STRESS SOURCES
    # =========================================================================

    scale_stress_sources(source, c0, flags, dt, dx, N, s_source_pos_index)

    # =========================================================================
    # VELOCITY SOURCES
    # =========================================================================

    apply_velocity_source_corrections(flags.use_w_source_correction_u, flags.source_ux, flags.source_uy, flags.source_uz, source, dt)

    scale_velocity_sources(flags, source, kgrid, c0, dt, dx, dy, dz, u_source_pos_index)

    # =========================================================================
    # TRANSDUCER SOURCE
    # =========================================================================
    transducer_input_signal = scale_transducer_source(flags.transducer_source, transducer_input_signal, c0, dt, dx, u_source_pos_index)
    return transducer_input_signal


def check_conditions(is_nonuniform_grid, is_source_uy, is_source_uz, is_transducer_source):
    """
    check for non-uniform grid and give error for source terms that haven't yet been implemented
    Returns:

    """
    if is_nonuniform_grid and (is_source_uy or is_source_uz or is_transducer_source):
        logging.log(logging.WARN, "source scaling not implemented for non-uniform grids with given source condition")
        return False
    return True


def apply_pressure_source_correction(is_source_p, use_w_source_correction_p, source, dt):
    """
    apply k-space source correction expressed as a function of w
    Args:
        is_source_p:
        use_w_source_correction_p:
        source:
        dt:

    Returns:

    """
    if is_source_p and use_w_source_correction_p:
        source.p = apply_source_correction(source.p, source.p_frequency_ref, dt)


def scale_pressure_source(is_source_p, source, kgrid, N, c0, dx, dt, p_source_pos_index, is_nonuniform_grid):
    """
    scale the input pressure by 1/c0^2 (to convert to units of density), then
    by 1/N (to split the input across the split density field). If the
    pressure is injected as a mass source, also scale the pressure by
    2*dt*c0/dx to account for the time step and convert to units of [kg/(m^3 s)]
    Args:
        is_source_p:
        source:
        kgrid:
        N:
        c0:
        dx:
        dt:
        p_source_pos_index:
        is_nonuniform_grid:

    Returns:

    """
    if not is_source_p:
        return

    if source.p_mode == "dirichlet":
        source.p = scale_pressure_source_dirichlet(source.p, c0, N, p_source_pos_index)
    else:
        if is_nonuniform_grid:
            source.p = scale_pressure_source_nonuniform_grid(source.p, kgrid, c0, N, dt, p_source_pos_index)

        else:
            source.p = scale_pressure_source_uniform_grid(source.p, c0, N, dx, dt, p_source_pos_index)


def scale_pressure_source_dirichlet(source_p, c0, N, p_source_pos_index):
    if c0.size == 1:
        # compute the scale parameter based on the homogeneous
        # sound speed
        source_p = source_p / (N * (c0**2))

    else:
        # compute the scale parameter seperately for each source
        # position based on the sound speed at that position
        ind = range(source_p[:, 0].size)
        mask = p_source_pos_index.flatten("F")[ind]
        scale = 1.0 / (N * np.expand_dims(c0.ravel(order="F")[mask.ravel(order="F")] ** 2, axis=-1))
        source_p[ind, :] *= scale

    return source_p


def scale_pressure_source_nonuniform_grid(source_p, kgrid, c0, N, dt, p_source_pos_index):
    x = kgrid.x
    xn = kgrid.xn
    yn = kgrid.yn
    zn = kgrid.zn
    x_size, y_size, z_size = kgrid.size

    # create empty matrix
    grid_point_sep = np.zeros(x.size)

    # compute averaged grid point seperation map, the interior
    # points are calculated using the average distance to all
    # connected grid points (the edge values are not calculated
    # assuming there are no source points in the PML)
    if kgrid.dim == 1:
        grid_point_sep[1:-1] = x_size * (xn[2:, 0] - xn[0:-2, 0]) / 2
    elif kgrid.dim == 2:
        grid_point_sep[1:-1, 1:-1] = x_size * (xn[2:, 1:-1] - xn[0:-2, 1:-1]) / 4 + y_size * (yn[1:-1, 2:] - yn[1:-1, 0:-2]) / 4
    elif kgrid.dim == 3:
        grid_point_sep[1:-1, 1:-1, 1:-1] = (
            x_size * (xn[2:, 1:-1, 1:-1] - xn[0:-2, 1:-1, 1:-1]) / 6
            + y_size * (yn[1:-1, 2:, 1:-1] - yn[1:-1, 0:-2, 1:-1]) / 6
            + z_size * (zn[1:-1, 1:-1, 2:] - zn[1:-1, 1:-1, 0:-2]) / 6
        )

    # compute and apply scale parameter
    for p_index in range(source_p.size[0]):
        if c0.size == 1:
            # compute the scale parameter based on the homogeneous sound speed
            source_p[p_index, :] = source_p[p_index, :] * (2 * dt / (N * c0 * grid_point_sep[p_source_pos_index[p_index]]))

        else:
            # compute the scale parameter based on the sound speed at that position
            source_p[p_index, :] = source_p[p_index, :] * (
                2 * dt / (N * c0[p_source_pos_index[p_index]] * grid_point_sep[p_source_pos_index[p_index]])
            )
    return source_p


def scale_pressure_source_uniform_grid(source_p, c0, N, dx, dt, p_source_pos_index):
    if c0.size == 1:
        # compute the scale parameter based on the homogeneous
        # sound speed
        source_p = source_p * (2 * dt / (N * c0 * dx))

    else:
        # compute the scale parameter seperately for each source
        # position based on the sound speed at that position
        ind = range(source_p[:, 0].size)
        mask = p_source_pos_index.flatten("F")[ind]
        scale = (2.0 * dt) / (N * np.expand_dims(c0.ravel(order="F")[mask.ravel(order="F")], axis=-1) * dx)
        source_p[ind, :] *= scale
    return source_p


def scale_stress_sources(source, c0, flags, dt, dx, N, s_source_pos_index):
    """
    scale the stress source by 1/N to divide amoungst the split field
    components, and if source.s_mode is not set to 'dirichlet', also scale by
    2*dt*c0/dx to account for the time step and convert to units of
    [kg/(m^3 s)] (note dx is used in all dimensions)
    Args:
        source:
        c0:
        flags:
        dt:
        dx:
        N:
        s_source_pos_index:

    Returns:

    """
    source.sxx = scale_stress_source(source, c0, flags.source_sxx, flags.source_p0, source.sxx, dt, N, dx, s_source_pos_index)
    source.syy = scale_stress_source(source, c0, flags.source_syy, flags.source_p0, source.syy, dt, N, dx, s_source_pos_index)
    source.szz = scale_stress_source(source, c0, flags.source_szz, flags.source_p0, source.szz, dt, N, dx, s_source_pos_index)
    source.sxy = scale_stress_source(source, c0, flags.source_sxy, True, source.sxy, dt, N, dx, s_source_pos_index)
    source.sxz = scale_stress_source(source, c0, flags.source_sxz, True, source.sxz, dt, N, dx, s_source_pos_index)
    source.syz = scale_stress_source(source, c0, flags.source_syz, True, source.syz, dt, N, dx, s_source_pos_index)


def scale_stress_source(source, c0, is_source_exists, is_p0_exists, source_val, dt, N, dx, s_source_pos_index):
    if is_source_exists:
        if source.s_mode == "dirichlet" or is_p0_exists:
            source_val = source_val / N
        else:
            if c0.size == 1:
                # compute the scale parameter based on the homogeneous sound
                # speed
                source_val = source_val * (2 * dt * c0 / (N * dx))

            else:
                # compute the scale parameter seperately for each source
                # position based on the sound speed at that position
                s_index = range(source_val.size[0])
                source_val[s_index, :] = source_val[s_index, :] * (2 * dt * c0[s_source_pos_index[s_index]] / (N * dx))
    return source_val


def apply_velocity_source_corrections(
    use_w_source_correction_u: bool, is_ux_exists: bool, is_uy_exists: bool, is_uz_exists: bool, source: kSource, dt: float
):
    """
    apply k-space source correction expressed as a function of w
    Args:
        use_w_source_correction_u:
        is_ux_exists:
        is_uy_exists:
        is_uz_exists:
        source:
        dt:

    Returns:

    """
    if not use_w_source_correction_u:
        return

    if is_ux_exists:
        source.ux = apply_source_correction(source.ux, source.u_frequency_ref, dt)

    if is_uy_exists:
        source.uy = apply_source_correction(source.uy, source.u_frequency_ref, dt)

    if is_uz_exists:
        source.uz = apply_source_correction(source.uz, source.u_frequency_ref, dt)


def apply_source_correction(source_val, frequency_ref, dt):
    return source_val * math.cos(2 * math.pi * frequency_ref * dt / 2)


def scale_velocity_sources(flags, source, kgrid, c0, dt, dx, dy, dz, u_source_pos_index):
    source.ux = scale_velocity_source_x(
        flags.source_ux, source.u_mode, source.ux, kgrid, c0, dt, dx, u_source_pos_index, flags.nonuniform_grid
    )
    source.uy = scale_velocity_source(flags.source_uy, source.u_mode, source.uy, c0, dt, u_source_pos_index, dy)
    source.uz = scale_velocity_source(flags.source_uz, source.u_mode, source.uz, c0, dt, u_source_pos_index, dz)


def scale_velocity_source_x(is_source_ux, source_u_mode, source_val, kgrid, c0, dt, dx, u_source_pos_index, is_nonuniform_grid):
    """
    if source.u_mode is not set to 'dirichlet', scale the x-direction
    velocity source terms by 2*dt*c0/dx to account for the time step and
    convert to units of [m/s^2]
    Returns:

    """
    if not is_source_ux or source_u_mode == "dirichlet":
        return

    if is_nonuniform_grid:
        source_val = scale_velocity_source_nonuniform(is_source_ux, source_u_mode, kgrid, source_val, c0, dt, u_source_pos_index)
    else:
        source_val = scale_velocity_source(is_source_ux, source_u_mode, source_val, c0, dt, u_source_pos_index, dx)
    return source_val


def scale_velocity_source(is_source, source_u_mode, source_val, c0, dt, u_source_pos_index, d_direction):
    """
    if source.u_mode is not set to 'dirichlet', scale the d_direction
    velocity source terms by 2*dt*c0/dz to account for the time step and
    convert to units of [m/s^2]
    Args:
        is_source:
        source_u_mode:
        source_val:
        c0:
        dt:
        u_source_pos_index:
        d_direction:

    Returns:

    """
    if not is_source or source_u_mode == "dirichlet":
        return source_val

    if c0.size == 1:
        # compute the scale parameter based on the homogeneous sound speed
        source_val = source_val * (2 * c0 * dt / d_direction)
    else:
        # compute the scale parameter seperately for each source position
        # based on the sound speed at that position
        u_index = range(source_val.size[0])
        source_val[u_index, :] = source_val[u_index, :] * (2 * c0[u_source_pos_index[u_index]] * dt / d_direction)
    return source_val


def scale_velocity_source_nonuniform(is_source, source_u_mode, kgrid, source_val, c0, dt, u_source_pos_index):
    """
    if source.u_mode is not set to 'dirichlet', scale the d_direction
    velocity source terms by 2*dt*c0/dz to account for the time step and
    convert to units of [m/s^2]
    Args:
        is_source:
        source_u_mode:
        kgrid:
        source_val:
        c0:
        dt:
        u_source_pos_index:

    Returns:

    """
    if not is_source or source_u_mode == "dirichlet":
        return source_val

    # create empty matrix
    x = kgrid.x
    xn = kgrid.xn
    x_size = kgrid.size[0]
    grid_point_sep = np.zeros_like(x)

    # compute averaged grid point seperation map, the interior
    # points are calculated using the average distance to all
    # connected grid points (the edge values are not calculated
    # assuming there are no source points in the PML)
    grid_point_sep[1:-1, :, :] = x_size * (xn[2:, :, :] - xn[1:-2, :, :]) / 2

    # compute and apply scale parameter
    for u_index in range(source_val.size[0]):
        if c0.size == 1:
            # compute the scale parameter based on the homogeneous sound speed
            source_val[u_index, :] = source_val[u_index, :] * (2 * c0 * dt / (grid_point_sep[u_source_pos_index[u_index]]))
        else:
            # compute the scale parameter based on the sound speed at that position
            source_val[u_index, :] = source_val[u_index, :] * (
                2 * c0[u_source_pos_index[u_index]] * dt / (grid_point_sep[u_source_pos_index[u_index]])
            )
    return source_val


def scale_transducer_source(is_transducer_source, transducer_input_signal, c0, dt, dx, u_source_pos_index):
    """
    scale the transducer source term by 2*dt*c0/dx to account for the time
    step and convert to units of [m/s^2]
    Args:
        is_transducer_source:
        transducer_input_signal:
        c0:
        dt:
        dx:
        u_source_pos_index:

    Returns:

    """
    if is_transducer_source:
        if c0.size == 1:
            transducer_input_signal = transducer_input_signal * (2 * c0 * dt / dx)
        else:
            # compute the scale parameter based on the average sound speed at the
            # transducer positions (only one input signal is used to drive the transducer)
            transducer_input_signal = transducer_input_signal * (2 * (np.mean(c0.flatten(order="F")[u_source_pos_index - 1])) * dt / dx)
    return transducer_input_signal
