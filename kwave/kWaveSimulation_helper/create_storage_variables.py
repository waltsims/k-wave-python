import numpy as np
from numpy.fft import ifftshift
from copy import deepcopy
from typing import Union, List

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.recorder import Recorder
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.dotdictionary import dotdict


from scipy.spatial import Delaunay


def gridDataFast2D(x, y, xi, yi):
    """
    Delauney triangulation in 2D
    """
    x = np.ravel(x)
    y = np.ravel(y)
    xi = np.ravel(xi)
    yi = np.ravel(yi)

    points = np.squeeze(np.dstack((x, y)))
    interpolation_points = np.squeeze(np.dstack((xi, yi)))

    tri = Delaunay(points)

    indices = tri.find_simplex(interpolation_points)

    bc = tri.transform[indices, :2].dot(np.transpose(tri.points[indices, :] - tri.transform[indices, 2]))

    return tri.points[indices, :], bc


def gridDataFast3D(x, y, z, xi, yi, zi):
    """
    Delauney triangulation in 3D
    """
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)
    xi = np.ravel(xi)
    yi = np.ravel(yi)
    zi = np.ravel(zi)

    grid_points = np.squeeze(np.dstack((x, y, z)))
    interpolation_points = np.squeeze(np.dstack((xi, yi, zi)))

    tri = Delaunay(grid_points)

    simplex_indices = tri.find_simplex(interpolation_points)

    print("----------->", tri.simplices[simplex_indices])

    # barycentric coordinates
    bc = tri.transform[simplex_indices, :2].dot(np.transpose(tri.points[simplex_indices, :] - tri.transform[simplex_indices, 2]))

    print("----------->", bc)

    return tri.points[simplex_indices, :], bc


class OutputSensor(object):
    """
    Class which holds information about which spatial locations are used to save data
    """
    flags = None
    x_shift_neg = None
    p = None


def create_storage_variables(kgrid: kWaveGrid, sensor, opt: SimulationOptions,
                             values: dotdict, flags: dotdict, record: Recorder):
    """
    Creates the storage variable sensor
    """

    # =========================================================================
    # PREPARE DATA MASKS AND STORAGE VARIABLES
    # =========================================================================

    sensor_data = OutputSensor()

    # print("unset flags:")
    # for k, v in flags.items():
    #     print("\t", k, v)
    flags = set_flags(flags, values.sensor_x, sensor.mask, opt.cartesian_interp)
    # print("set flags:")
    # for k, v in flags.items():
    #     print("\t", k, v)

    # preallocate output variables
    if flags.time_rev:
        return flags

    num_sensor_points = get_num_of_sensor_points(flags.blank_sensor,
                                                 flags.binary_sensor_mask,
                                                 kgrid.k,
                                                 values.sensor_mask_index,
                                                 values.sensor_x)

    num_recorded_time_points, _ = \
        get_num_recorded_time_points(kgrid.dim, kgrid.Nt, opt.stream_to_disk, sensor.record_start_index)

    record = create_shift_operators(record, values.record, kgrid, opt.use_sg)

    create_normalized_wavenumber_vectors(record, kgrid, flags.record_u_split_field)

    pml_size = [opt.pml_x_size, opt.pml_y_size, opt.pml_z_size]
    pml_size = Vector(pml_size[:kgrid.dim])
    all_vars_size = calculate_all_vars_size(kgrid, opt.pml_inside, pml_size)

    sensor_data = create_sensor_variables(values.record, kgrid, num_sensor_points, num_recorded_time_points,
                                          all_vars_size, values.sensor_mask_index, flags.use_cuboid_corners)

    create_transducer_buffer(flags.transducer_sensor, values.transducer_receive_elevation_focus, sensor,
                             num_sensor_points, num_recorded_time_points, values.sensor_data_buffer_size,
                             flags, sensor_data)

    record = compute_triangulation_points(flags, kgrid, record, sensor.mask)

    return flags, record, sensor_data, num_recorded_time_points


def set_flags(flags: dotdict, sensor_x, sensor_mask, is_cartesian_interp):
    """
    check sensor mask based on the Cartesian interpolation setting
    """

    if not flags.binary_sensor_mask and is_cartesian_interp == 'nearest':

        # extract the data using the binary sensor mask created in
        # input_checking, but switch on Cartesian reorder flag so that the
        # final data is returned in the correct order (not in time
        # reversal mode).
        flags.binary_sensor_mask = True
        if not flags.time_rev:
            flags.reorder_data = True

        # check if any duplicate points have been discarded in the
        # conversion from a Cartesian to binary mask
        num_discarded_points = len(sensor_x) - sensor_mask.sum()
        if num_discarded_points != 0:
            print(f'  WARNING: {num_discarded_points} duplicated sensor points discarded (nearest neighbour interpolation)')

    return flags


def get_num_of_sensor_points(is_blank_sensor, is_binary_sensor_mask, kgrid_k, sensor_mask_index, sensor_x):
    """
    Returns the number of sensor points for a given set of sensor parameters.

    Args:
        is_blank_sensor (bool): Whether the sensor is blank or not.
        is_binary_sensor_mask (bool): Whether the sensor mask is binary or not.
        kgrid_k (ndarray): An array of k-values for the k-Wave grid.
        sensor_mask_index (list): List of sensor mask indices.
        sensor_x (list): List of sensor x-coordinates.

    Returns:
        int: The number of sensor points.
    """
    if is_blank_sensor:
        num_sensor_points = kgrid_k.size
    elif is_binary_sensor_mask:
        num_sensor_points = len(sensor_mask_index)
    else:
        num_sensor_points = len(sensor_x)
    return num_sensor_points


def get_num_recorded_time_points(kgrid_dim, Nt, stream_to_disk, record_start_index):
    """
    calculate the number of time points that are stored
    - if streaming data to disk, reduce to the size of the
        sensor_data matrix based on the value of self.options.stream_to_disk
    - if a user input for sensor.record_start_index is given, reduce
        the size of the sensor_data matrix based on the value given
    Args:
        kgrid_dim:
        Nt:
        stream_to_disk:
        record_start_index:

    Returns:

    """
    if kgrid_dim == 3 and stream_to_disk:

        # set the number of points
        num_recorded_time_points = stream_to_disk

        # initialise the file index variable
        stream_data_index = 1

    else:
        num_recorded_time_points = Nt - record_start_index + 1
        stream_data_index = None  # ???

    return num_recorded_time_points, stream_data_index


def create_shift_operators(record: Recorder, record_old: Recorder, kgrid: kWaveGrid, is_use_sg: bool):
    """
    create shift operators used for calculating the components of the
    particle velocity field on the non-staggered grids (these are used
    for both binary and cartesian sensor masks)
    """

    if (record_old.u_non_staggered or record_old.u_split_field or record_old.I or record_old.I_avg):
        if is_use_sg:
            if kgrid.dim == 1:
                record.x_shift_neg = ifftshift(np.exp(-1j * kgrid.k_vec.x * kgrid.dx / 2))
            elif kgrid.dim == 2:
                record.x_shift_neg = ifftshift(np.exp(-1j * kgrid.k_vec.x * kgrid.dx / 2))
                record.y_shift_neg = ifftshift(np.exp(-1j * kgrid.k_vec.y * kgrid.dy / 2)).T
            elif kgrid.dim == 3:
                record.x_shift_neg = ifftshift(np.exp(-1j * kgrid.k_vec.x * kgrid.dx / 2))
                record.y_shift_neg = ifftshift(np.exp(-1j * kgrid.k_vec.y * kgrid.dy / 2))
                record.z_shift_neg = ifftshift(np.exp(-1j * kgrid.k_vec.z * kgrid.dz / 2))

                record.x_shift_neg = np.expand_dims(record.x_shift_neg, axis=-1)

                record.y_shift_neg = np.expand_dims(record.y_shift_neg, axis=0)

                record.z_shift_neg = np.squeeze(record.z_shift_neg)
                record.z_shift_neg = np.expand_dims(record.z_shift_neg, axis=0)
                record.z_shift_neg = np.expand_dims(record.z_shift_neg, axis=0)

        else:
            if kgrid.dim == 1:
                record.x_shift_neg = 1
            elif kgrid.dim == 2:
                record.x_shift_neg = 1
                record.y_shift_neg = 1
            elif kgrid.dim == 3:
                record.x_shift_neg = 1
                record.y_shift_neg = 1
                record.z_shift_neg = 1
    return record


def create_normalized_wavenumber_vectors(record: Recorder, kgrid: kWaveGrid, is_record_u_split_field):
    """
    create normalised wavenumber vectors for k-space dyadics used to
    split the particule velocity into compressional and shear components
    """
    if not is_record_u_split_field:
        return record

    # x-dimension
    record.kx_norm = kgrid.kx / kgrid.k
    record.kx_norm[kgrid.k == 0] = 0
    record.kx_norm = ifftshift(record.kx_norm)

    # y-dimension
    record.ky_norm = kgrid.ky / kgrid.k
    record.ky_norm[kgrid.k == 0] = 0
    record.ky_norm = ifftshift(record.ky_norm)

    # z-dimension
    if kgrid.dim == 3:
        record.kz_norm = kgrid.kz / kgrid.k
        record.kz_norm[kgrid.k == 0] = 0
        record.kz_norm = ifftshift(record.kz_norm)

    return record


def create_sensor_variables(record_old: Recorder, kgrid, num_sensor_points, num_recorded_time_points,
                            all_vars_size, sensor_mask_index, use_cuboid_corners) -> Union[dotdict, List[dotdict]]:
    """
    create storage and scaling variables - all variables are saved as fields of
    a container called sensor_data. If cuboid corners are used this is a list, else a dictionary-like container
    """

    print(record_old)

    if use_cuboid_corners:

        # as a list
        sensor_data = []

        # get number of doctdicts in the list for each set of cuboid corners
        n_cuboids: int = np.shape(record_old.cuboid_corners_list)[1]

        # for each set of cuboid corners
        for cuboid_index in np.arange(n_cuboids, dtype=int):

            # add an entry to the list
            sensor_data.append(dotdict())

            # get size of cuboid for indexing regions of computational grid
            if kgrid.dim == 1:
                cuboid_size_x = [record_old.cuboid_corners_list[1, cuboid_index] - record_old.cuboid_corners_list[0, cuboid_index] + 1, 1]
            elif kgrid.dim == 2:
                cuboid_size_x = [record_old.cuboid_corners_list[2, cuboid_index] - record_old.cuboid_corners_list[0, cuboid_index] + 1,
                                 record_old.cuboid_corners_list[3, cuboid_index] - record_old.cuboid_corners_list[1, cuboid_index] + 1]
            elif kgrid.dim == 3:
                cuboid_size_x = [record_old.cuboid_corners_list[3, cuboid_index] - record_old.cuboid_corners_list[0, cuboid_index] + 1,
                                 record_old.cuboid_corners_list[4, cuboid_index] - record_old.cuboid_corners_list[1, cuboid_index] + 1,
                                 record_old.cuboid_corners_list[5, cuboid_index] - record_old.cuboid_corners_list[2, cuboid_index] + 1]

            cuboid_size_xt = deepcopy(cuboid_size_x)
            cuboid_size_xt.append(num_recorded_time_points)

            # time history of the acoustic pressure
            if record_old.p or record_old.I or record_old.I_avg:
                sensor_data[cuboid_index].p = np.zeros(cuboid_size_xt)

            # maximum pressure
            if record_old.p_max:
                sensor_data[cuboid_index].p_max = np.zeros(cuboid_size_x)

            # minimum pressure
            if record_old.p_min:
                sensor_data[cuboid_index].p_min = np.zeros(cuboid_size_x)

            # rms pressure
            if record_old.p_rms:
                sensor_data[cuboid_index].p_rms = np.zeros(cuboid_size_x)

            # time history of the acoustic particle velocity
            if record_old.u:
                # pre-allocate the velocity fields based on the number of dimensions in the simulation
                if kgrid.dim == 1:
                    sensor_data[cuboid_index].ux = np.zeros(cuboid_size_xt)
                elif kgrid.dim == 2:
                    sensor_data[cuboid_index].ux = np.zeros(cuboid_size_xt)
                    sensor_data[cuboid_index].uy = np.zeros(cuboid_size_xt)
                elif kgrid.dim == 3:
                    sensor_data[cuboid_index].ux = np.zeros(cuboid_size_xt)
                    sensor_data[cuboid_index].uy = np.zeros(cuboid_size_xt)
                    sensor_data[cuboid_index].uz = np.zeros(cuboid_size_xt)

            # store the time history of the particle velocity on staggered grid
            if record_old.u_non_staggered or record_old.I or record_old.I_avg:
                print("record_old is correct")
                # pre-allocate the velocity fields based on the number of dimensions in the simulation
                if kgrid.dim == 1:
                    sensor_data[cuboid_index].ux_non_staggered = np.zeros(cuboid_size_xt)
                elif kgrid.dim == 2:
                    sensor_data[cuboid_index].ux_non_staggered = np.zeros(cuboid_size_xt)
                    sensor_data[cuboid_index].uy_non_staggered = np.zeros(cuboid_size_xt)
                elif kgrid.dim == 3:
                    print("THIS MUST BE SET")
                    sensor_data[cuboid_index].ux_non_staggered = np.zeros(cuboid_size_xt)
                    sensor_data[cuboid_index].uy_non_staggered = np.zeros(cuboid_size_xt)
                    sensor_data[cuboid_index].uz_non_staggered = np.zeros(cuboid_size_xt)

            # time history of the acoustic particle velocity split into compressional and shear components
            if record_old.u_split_field:
                # pre-allocate the velocity fields based on the number of dimensions in the simulation
                if kgrid.dim == 2:
                    sensor_data[cuboid_index].ux_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].ux_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].uy_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].uy_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                if kgrid.dim == 3:
                    sensor_data[cuboid_index].ux_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].ux_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].uy_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].uy_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].uz_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                    sensor_data[cuboid_index].uz_split_s = np.zeros([num_sensor_points, num_recorded_time_points])

    else:

        # allocate empty sensor
        sensor_data = dotdict()

        # if only p is being stored (i.e., if no user input is given for
        # sensor.record), then sensor_data.p is copied to sensor_data at the
        # end of the simulation

        # time history of the acoustic pressure
        if record_old.p or record_old.I or record_old.I_avg:
            # print("create storage:", num_sensor_points, num_recorded_time_points, np.shape(sensor_data.p) )
            sensor_data.p = np.zeros([num_sensor_points, num_recorded_time_points])

        # maximum pressure
        if record_old.p_max:
            sensor_data.p_max = np.zeros([num_sensor_points,])

        # minimum pressure
        if record_old.p_min:
            sensor_data.p_min = np.zeros([num_sensor_points,])

        # rms pressure
        if record_old.p_rms:
            sensor_data.p_rms = np.zeros([num_sensor_points,])

        # maximum pressure over all grid points
        if record_old.p_max_all:
            sensor_data.p_max_all = np.zeros(all_vars_size)

        # minimum pressure over all grid points
        if record_old.p_min_all:
            sensor_data.p_min_all = np.zeros(all_vars_size)

        # time history of the acoustic particle velocity
        if record_old.u:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux = np.zeros([num_sensor_points, num_recorded_time_points])
            elif kgrid.dim == 2:
                sensor_data.ux = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy = np.zeros([num_sensor_points, num_recorded_time_points])
            elif kgrid.dim == 3:
                sensor_data.ux = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uz = np.zeros([num_sensor_points, num_recorded_time_points])

        # maximum particle velocity
        if record_old.u_max:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux_max = np.zeros([num_sensor_points,])
            if kgrid.dim == 2:
                sensor_data.ux_max = np.zeros([num_sensor_points,])
                sensor_data.uy_max = np.zeros([num_sensor_points,])
            if kgrid.dim == 3:
                sensor_data.ux_max = np.zeros([num_sensor_points,])
                sensor_data.uy_max = np.zeros([num_sensor_points,])
                sensor_data.uz_max = np.zeros([num_sensor_points,])

        # minimum particle velocity
        if record_old.u_min:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux_min = np.zeros([num_sensor_points,])
            if kgrid.dim == 2:
                sensor_data.ux_min = np.zeros([num_sensor_points,])
                sensor_data.uy_min = np.zeros([num_sensor_points,])
            if kgrid.dim == 3:
                sensor_data.ux_min = np.zeros([num_sensor_points,])
                sensor_data.uy_min = np.zeros([num_sensor_points,])
                sensor_data.uz_min = np.zeros([num_sensor_points,])

        # rms particle velocity
        if record_old.u_rms:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux_rms = np.zeros([num_sensor_points,])
            if kgrid.dim == 2:
                sensor_data.ux_rms = np.zeros([num_sensor_points,])
                sensor_data.uy_rms = np.zeros([num_sensor_points,])
            if kgrid.dim == 3:
                sensor_data.ux_rms = np.zeros([num_sensor_points,])
                sensor_data.uy_rms = np.zeros([num_sensor_points,])
                sensor_data.uz_rms = np.zeros([num_sensor_points,])

        # maximum particle velocity over all grid points
        if record_old.u_max_all:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux_max_all = np.zeros(all_vars_size)
            if kgrid.dim == 2:
                sensor_data.ux_max_all = np.zeros(all_vars_size)
                sensor_data.uy_max_all = np.zeros(all_vars_size)
            if kgrid.dim == 3:
                sensor_data.ux_max_all = np.zeros(all_vars_size)
                sensor_data.uy_max_all = np.zeros(all_vars_size)
                sensor_data.uz_max_all = np.zeros(all_vars_size)

        # minimum particle velocity over all grid points
        if record_old.u_min_all:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux_min_all = np.zeros(all_vars_size)
            if kgrid.dim == 2:
                sensor_data.ux_min_all = np.zeros(all_vars_size)
                sensor_data.uy_min_all = np.zeros(all_vars_size)
            if kgrid.dim == 3:
                sensor_data.ux_min_all = np.zeros(all_vars_size)
                sensor_data.uy_min_all = np.zeros(all_vars_size)
                sensor_data.uz_min_all = np.zeros(all_vars_size)

        # time history of the acoustic particle velocity on the non-staggered grid points
        if record_old.u_non_staggered or record_old.I or record_old.I_avg:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 1:
                sensor_data.ux_non_staggered = np.zeros([num_sensor_points, num_recorded_time_points])
            if kgrid.dim == 2:
                sensor_data.ux_non_staggered = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy_non_staggered = np.zeros([num_sensor_points, num_recorded_time_points])
            if kgrid.dim == 3:
                sensor_data.ux_non_staggered = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy_non_staggered = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uz_non_staggered = np.zeros([num_sensor_points, num_recorded_time_points])

        # time history of the acoustic particle velocity split into compressional and shear components
        if record_old.u_split_field:
            # pre-allocate the velocity fields based on the number of dimensions in the simulation
            if kgrid.dim == 2:
                sensor_data.ux_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.ux_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
            if kgrid.dim == 3:
                sensor_data.ux_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.ux_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uy_split_s = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uz_split_p = np.zeros([num_sensor_points, num_recorded_time_points])
                sensor_data.uz_split_s = np.zeros([num_sensor_points, num_recorded_time_points])

    if use_cuboid_corners:
        info = "using cuboid_corners," + str(len(sensor_data)) + ", " + str(np.shape(sensor_data[0].p))
    else:
        info = "binary_mask, ", np.shape(sensor_data.p)
    print("end here", info)

    return sensor_data


def create_transducer_buffer(is_transducer_sensor, is_transducer_receive_elevation_focus, sensor,
                             num_sensor_points, num_recorded_time_points, sensor_data_buffer_size,
                             flags, sensor_data):
    # object of the kWaveTransducer class is being used as a sensor

    if is_transducer_sensor:
        if is_transducer_receive_elevation_focus:

            # if there is elevation focusing, a buffer is
            # needed to store a short time history at each
            # sensor point before averaging
            # ???
            sensor_data_buffer_size = sensor.elevation_beamforming_delays.max() + 1
            if sensor_data_buffer_size > 1:
                sensor_data_buffer = np.zeros([num_sensor_points, sensor_data_buffer_size])  # noqa: F841
            else:
                del sensor_data_buffer_size
                flags.transducer_receive_elevation_focus = False

        # the grid points can be summed on the fly and so the
        # sensor is the size of the number of active elements
        sensor_data.transducer = np.zeros([int(sensor.number_active_elements), num_recorded_time_points])
    else:
        pass


def compute_triangulation_points(flags, kgrid, record, mask):
    """
    precomputate the triangulation points if a Cartesian sensor mask is used
    with linear interpolation (tri and bc are the Delaunay TRIangulation and
    Barycentric Coordinates)
    """

    if not flags.binary_sensor_mask:

        if kgrid.dim == 1:

            # assign pseudonym for Cartesain grid points in 1D (this is later used for data casting)
            record.grid_x = kgrid.x_vec

        else:

            if kgrid.dim == 1:
              # align sensor data as a column vector to be the same as kgrid.x_vec
              # so that calls to interp return data in the correct dimension
              sensor_x = np.reshape((mask, (-1, 1)))
            elif kgrid.dim == 2:
                sensor_x = mask[0, :]
                sensor_y = mask[1, :]
            elif kgrid.dim == 3:
                sensor_x = mask[0, :]
                sensor_y = mask[1, :]
                sensor_z = mask[2, :]

            # update command line status
            print('  calculating Delaunay triangulation...')

            # compute triangulation
            if kgrid.dim == 2:
                if flags.axisymmetric:
                    record.tri, record.bc = gridDataFast2D(kgrid.x, kgrid.y - kgrid.y_vec.min(), sensor_x, sensor_y)
                else:
                    record.tri, record.bc = gridDataFast2D(kgrid.x, kgrid.y, sensor_x, sensor_y)
            elif kgrid.dim == 3:
                record.tri, record.bc = gridDataFast3D(kgrid.x, kgrid.y, kgrid.z, sensor_x, sensor_y, sensor_z)

            print("done")

    return record


def calculate_all_vars_size(kgrid, is_pml_inside, pml_size):
    """
    calculate the size of the _all and _final output variables - if the
    PML is set to be outside the grid, these will be the same size as the
    user input, rather than the expanded grid
    """
    if is_pml_inside:
        all_vars_size = kgrid.k.shape
    else:
        if kgrid.dim == 1:
            all_vars_size = [kgrid.Nx - 2 * pml_size.x, 1]
        elif kgrid.dim == 2:
            all_vars_size = [kgrid.Nx - 2 * pml_size.x, kgrid.Ny - 2 * pml_size.y]
        elif kgrid.dim == 3:
            all_vars_size = [kgrid.Nx - 2 * pml_size.x, kgrid.Ny - 2 * pml_size.y, kgrid.Nz - 2 * pml_size.z]
        else:
            raise NotImplementedError
    return all_vars_size
