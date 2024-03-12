import logging
import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationOptions
from kwave.ktransducer import NotATransducer
from kwave.data import Vector
from kwave.utils.data import get_smallest_possible_type
from kwave.utils.dotdictionary import dotdict
from kwave.utils.matlab import matlab_find
from kwave.utils.matrix import expand_matrix


def expand_grid_matrices(kgrid: kWaveGrid, medium: kWaveMedium, source, sensor, opt: SimulationOptions, values: dotdict, flags: dotdict):
    # update command line status
    logging.log(logging.INFO, "  expanding computational grid...")

    #####################
    # Grab values
    #####################

    # retaining the values for kgrid time array
    pml_size = [opt.pml_x_size, opt.pml_y_size, opt.pml_z_size]
    pml_size = Vector(pml_size[: kgrid.dim])

    p_source_pos_index = values.p_source_pos_index
    u_source_pos_index = values.u_source_pos_index
    s_source_pos_index = values.s_source_pos_index

    is_source_sensor_same = isinstance(sensor, NotATransducer) and sensor == source

    #####################
    # Expand Structures
    #####################

    # expand the computational grid, replacing the original grid
    kgrid = expand_kgrid(kgrid, flags.axisymmetric, pml_size)

    expand_size = calculate_expand_size(kgrid, flags.axisymmetric, pml_size)

    # update the data type in case adding the PML requires additional index precision
    total_grid_points = kgrid.total_grid_points
    index_data_type = get_smallest_possible_type(total_grid_points, "uint", default="double")

    expand_sensor(sensor, expand_size, flags.use_sensor, flags.blank_sensor)

    # TODO why it is not self.record ? "self"
    record = expand_cuboid_corner_list(flags.cuboid_corners, kgrid, pml_size)  # noqa: F841

    expand_medium(medium, expand_size)

    p_source_pos_index, u_source_pos_index, s_source_pos_index = expand_source(
        source, is_source_sensor_same, flags, expand_size, index_data_type, p_source_pos_index, u_source_pos_index, s_source_pos_index
    )

    expand_directivity_angle(kgrid, sensor, expand_size, flags.use_sensor, flags.compute_directivity)

    print_grid_size(kgrid)

    return kgrid, index_data_type, p_source_pos_index, u_source_pos_index, s_source_pos_index


def expand_kgrid(kgrid, is_axisymmetric, pml_size):
    Nt_temp, dt_temp = kgrid.Nt, kgrid.dt

    pml_size = pml_size.squeeze()

    if kgrid.dim == 1:
        new_size = kgrid.N + 2 * pml_size
    elif kgrid.dim == 2:
        if is_axisymmetric:
            new_size = [kgrid.Nx + 2 * pml_size[0], kgrid.Ny + pml_size[1]]
        else:
            new_size = kgrid.N + 2 * pml_size
    elif kgrid.dim == 3:
        new_size = kgrid.N + 2 * pml_size
    else:
        raise NotImplementedError

    kgrid = kWaveGrid(new_size, kgrid.spacing)
    # re-assign original time array
    kgrid.setTime(Nt_temp, dt_temp)

    return kgrid


def calculate_expand_size(kgrid, is_axisymmetric, pml_size):
    # set the PML size for use with expandMatrix, don't expand the inner radial
    # dimension if using the axisymmetric code
    if kgrid.dim == 1:
        expand_size = pml_size[0]
    elif kgrid.dim == 2:
        if is_axisymmetric:
            expand_size = [pml_size[0], pml_size[0], 0, pml_size[1]]
        else:
            expand_size = pml_size
    elif kgrid.dim == 3:
        expand_size = pml_size
    else:
        raise NotImplementedError
    return np.array(expand_size)


def expand_medium(medium: kWaveMedium, expand_size):
    # enlarge the sound speed grids by exting the edge values into the expanded grid
    medium.sound_speed = np.atleast_1d(medium.sound_speed)
    if medium.sound_speed.size > 1:
        medium.sound_speed = expand_matrix(medium.sound_speed, expand_size)

    # enlarge the grid of density by exting the edge values into the expanded grid
    medium.density = np.atleast_1d(medium.density)
    if medium.density.size > 1:
        medium.density = expand_matrix(medium.density, expand_size)

    # for key in ['alpha_coeff', 'alpha_coeff_compression', 'alpha_coeff_shear', 'BonA']:
    for key in ["alpha_coeff", "BonA"]:
        # enlarge the grid of medium[key] if given
        attr = getattr(medium, key)
        if attr is not None and np.atleast_1d(attr).size > 1:
            attr = expand_matrix(np.atleast_1d(attr), expand_size)
            setattr(medium, key, attr)

    # enlarge the absorption filter mask if given
    if medium.alpha_filter is not None:
        medium.alpha_filter = expand_matrix(medium.alpha_filter, expand_size, 0)


def expand_source(
    source, is_source_sensor_same, flags, expand_size, index_data_type, p_source_pos_index, u_source_pos_index, s_source_pos_index
):
    p_source_pos_index = expand_pressure_sources(source, expand_size, flags.source_p0, flags.source_p, index_data_type, p_source_pos_index)

    u_source_pos_index = expand_velocity_sources(
        source,
        expand_size,
        is_source_sensor_same,
        index_data_type,
        u_source_pos_index,
        flags.source_ux,
        flags.source_uy,
        flags.source_uz,
        flags.transducer_source,
    )

    s_source_pos_index = expand_stress_sources(source, expand_size, flags, index_data_type, s_source_pos_index)

    return p_source_pos_index, u_source_pos_index, s_source_pos_index


def expand_pressure_sources(source, expand_size, is_source_p0, is_source_p, index_data_type, p_source_pos_index):
    # enlarge the initial pressure if given
    if is_source_p0:
        source.p0 = expand_matrix(source.p0, expand_size, 0)

    # enlarge the pressure source mask if given
    if is_source_p:
        # enlarge the pressure source mask
        source.p_mask = expand_matrix(source.p_mask, expand_size, 0)

        # create an indexing variable corresponding to the source elements
        # and convert the data type deping on the number of indices
        p_source_pos_index = matlab_find(source.p_mask).astype(index_data_type)
    return p_source_pos_index


def expand_velocity_sources(
    source,
    expand_size,
    is_source_sensor_same,
    index_data_type,
    u_source_pos_index,
    is_source_ux,
    is_source_uy,
    is_source_uz,
    is_transducer_source,
):
    """
        enlarge the velocity source mask if given
    Args:
        source:
        expand_size:
        is_source_sensor_same:
        index_data_type:
        u_source_pos_index:
        is_source_ux:
        is_source_uy:
        is_source_uz:
        is_transducer_source:

    Returns:

    """
    if is_source_ux or is_source_uy or is_source_uz or is_transducer_source:
        # update the source indexing variable
        if isinstance(source, NotATransducer):
            # check if the sensor is also the same transducer, if so, don't expand the grid again
            if not is_source_sensor_same:
                # expand the transducer mask
                source.expand_grid(expand_size)

            # get the new active elements mask
            active_elements_mask = source.active_elements_mask

            # update the indexing variable corresponding to the active elements
            u_source_pos_index = matlab_find(active_elements_mask)
        else:
            # enlarge the velocity source mask
            source.u_mask = expand_matrix(source.u_mask, expand_size, 0)

            # create an indexing variable corresponding to the source elements
            u_source_pos_index = matlab_find(source.u_mask)

        # convert the data type deping on the number of indices
        u_source_pos_index = u_source_pos_index.astype(index_data_type)
    return u_source_pos_index


def expand_stress_sources(source, expand_size, flags, index_data_type, s_source_pos_index):
    # enlarge the stress source mask if given
    if flags.source_sxx or flags.source_syy or flags.source_szz or flags.source_sxy or flags.source_sxz or flags.source_syz:
        # enlarge the velocity source mask
        source.s_mask = expand_matrix(source.s_mask, expand_size, 0)

        # create an indexing variable corresponding to the source elements
        s_source_pos_index = matlab_find(source.s_mask != 0)

        # convert the data type deping on the number of indices
        s_source_pos_index = s_source_pos_index.astype(index_data_type)
    return s_source_pos_index


def expand_directivity_angle(kgrid, sensor, expand_size, is_use_sensor, is_compute_directivity):
    """
        enlarge the directivity angle if given (2D only)
    Args:
        kgrid:
        sensor:
        expand_size:
        is_use_sensor:
        is_compute_directivity:

    Returns:

    """
    if is_use_sensor and kgrid.dim == 2 and is_compute_directivity:
        # enlarge the directivity angle
        sensor.directivity.angle = expand_matrix(sensor.directivity.angle, expand_size, 0)
        # re-assign the wavenumber vectors
        sensor.directivity.wavenumbers = np.hstack((kgrid.ky.T, kgrid.kx.T))


def print_grid_size(kgrid):
    """
        update command line status
    Args:
        kgrid:

    Returns:

    """
    k_Nx, k_Ny, k_Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    if kgrid.dim == 1:
        logging.log(logging.INFO, "  computational grid size:", int(k_Nx), "grid points")
    elif kgrid.dim == 2:
        logging.log(logging.INFO, "  computational grid size:", int(k_Nx), "by", int(k_Ny), "grid points")
    elif kgrid.dim == 3:
        logging.log(logging.INFO, "  computational grid size:", int(k_Nx), "by", int(k_Ny), "by", int(k_Nz), "grid points")


def expand_cuboid_corner_list(is_cuboid_list, kgrid, pml_size: Vector):
    """
        add the PML size to cuboid corner indices if using a cuboid sensor mask
    Args:
        is_cuboid_list:
        kgrid:

    Returns:

    """
    if not is_cuboid_list:
        return

    record = dotdict()
    if kgrid.dim == 1:
        record.cuboid_corners_list = record.cuboid_corners_list + pml_size.x
    elif kgrid.dim == 2:
        record.cuboid_corners_list[[0, 2], :] = record.cuboid_corners_list[[0, 2], :] + pml_size.x
        record.cuboid_corners_list[[1, 3], :] = record.cuboid_corners_list[[1, 3], :] + pml_size.y
    elif kgrid.dim == 3:
        record.cuboid_corners_list[[0, 3], :] = record.cuboid_corners_list[[0, 3], :] + pml_size.x
        record.cuboid_corners_list[[1, 4], :] = record.cuboid_corners_list[[1, 4], :] + pml_size.y
        record.cuboid_corners_list[[2, 5], :] = record.cuboid_corners_list[[2, 5], :] + pml_size.z
    return record


def expand_sensor(sensor, expand_size, is_use_sensor, is_blank_sensor):
    """
        enlarge the sensor mask (for Cartesian sensor masks and cuboid corners,
        this has already been converted to a binary mask for display in inputChecking)
    Args:
        sensor:
        expand_size:
        is_use_sensor:
        is_blank_sensor:

    Returns:

    """
    if is_use_sensor and not is_blank_sensor:
        sensor.expand_grid(expand_size)
