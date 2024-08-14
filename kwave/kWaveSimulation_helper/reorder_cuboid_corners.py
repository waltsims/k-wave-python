import numpy as np

from kwave.utils.dotdictionary import dotdict

def reorder_cuboid_corners(kgrid, record, sensor_data, time_info, flags, verbose: bool = False):

    """DESCRIPTION:
    Method to reassign the sensor data belonging to each set of cuboid corners
    from the indexed sensor mask data.

    ABOUT:
        author      - Bradley Treeby
        date        - 8th July 2014
        last update - 15th May 2018

    This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    Copyright (C) 2014-2018 Bradley Treeby

    This file is part of k-Wave. k-Wave is free software: you can
    redistribute it and/or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
    more details.

    You should have received a copy of the GNU Lesser General Public License
    along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
    """

    # update command line status
    if verbose:
        print('  reordering cuboid corners data...')

    # set cuboid index variable
    cuboid_start_pos: int = 0

    # create list of cuboid corners
    sensor_data_temp = []

    # set number of time points from dotdict container
    if time_info.stream_to_disk:
        cuboid_num_time_points = time_info.num_stream_time_points
    else:
        cuboid_num_time_points = time_info.num_recorded_time_points

    # loop through cuboid corners and for each recorded variable, reshape to
    # [X, Y, Z, Y] or [X, Y, Z] instead of [sensor_index, T] or [sensor_index]
    for cuboid_index in np.arange(np.shape(record.cuboid_corners_list)[1]):

        # get size of cuboid
        if kgrid.dim == 1:
            cuboid_size_x = [record.cuboid_corners_list[1, cuboid_index] - record.cuboid_corners_list[0, cuboid_index], 1]
            cuboid_size_xt = [cuboid_size_x[0], cuboid_num_time_points]
        elif kgrid.dim == 2:
            cuboid_size_x = [record.cuboid_corners_list[2, cuboid_index] - record.cuboid_corners_list[0, cuboid_index],
                             record.cuboid_corners_list[3, cuboid_index] - record.cuboid_corners_list[1, cuboid_index]]
            cuboid_size_xt = [cuboid_size_x, cuboid_num_time_points]
        elif kgrid.dim == 3:
            cuboid_size_x = [record.cuboid_corners_list[3, cuboid_index] - record.cuboid_corners_list[0, cuboid_index],
                             record.cuboid_corners_list[4, cuboid_index] - record.cuboid_corners_list[1, cuboid_index],
                             record.cuboid_corners_list[5, cuboid_index] - record.cuboid_corners_list[2, cuboid_index]]
            cuboid_size_xt = [cuboid_size_x, cuboid_num_time_points]

        # set index and size variables
        cuboid_num_points = np.prod(cuboid_size_x)

        sensor_data_temp.append(dotdict())

        # print(cuboid_size_x)
        # print(cuboid_size_xt)
        # print(cuboid_num_points)
        # print(flags)
        # if flags.record_p_max:
        #     print(np.shape(sensor_data.p_max) )
        #     print(cuboid_start_pos, cuboid_start_pos + cuboid_num_points )
        # if record.p_max:
        #     print(np.shape(sensor_data.p_max) )
        #     print(cuboid_start_pos, cuboid_start_pos + cuboid_num_points)
        #     print(cuboid_start_pos, cuboid_start_pos + cuboid_num_points)
        #     x = np.arange(cuboid_start_pos, cuboid_start_pos + cuboid_num_points )
        #     print(np.shape(x))
        #     x[cuboid_start_pos:cuboid_start_pos + cuboid_num_points]


        if flags.record_p:
            sensor_data_temp[cuboid_index].p =  np.reshape(
                sensor_data.p[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)

        if flags.record_p_max:
            sensor_data_temp[cuboid_index].p_max =  np.reshape(
                sensor_data.p_max[cuboid_start_pos:cuboid_start_pos + cuboid_num_points], cuboid_size_x)

        if flags.record_p_min:
            sensor_data_temp[cuboid_index].p_min =  np.reshape(
                sensor_data.p_min[cuboid_start_pos:cuboid_start_pos + cuboid_num_points], cuboid_size_x)

        if flags.record_p_rms:
            sensor_data_temp[cuboid_index].p_rms =  np.reshape(
                sensor_data.p_rms[cuboid_start_pos:cuboid_start_pos + cuboid_num_points], cuboid_size_x)

        if flags.record_u:
            # x-dimension
            sensor_data_temp[cuboid_index].ux =  np.reshape(
                sensor_data.ux[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].uy =  np.reshape(
                    sensor_data.uy[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].uz =  np.reshape(
                    sensor_data.uz[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)

        if flags.record_u_non_staggered:
            # x-dimension
            sensor_data_temp[cuboid_index].ux_non_staggered =  np.reshape(
                sensor_data.ux_non_staggered[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].uy_non_staggered =  np.reshape(
                    sensor_data.uy_non_staggered[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].uz_non_staggered =  np.reshape(
                    sensor_data.uz_non_staggered[cuboid_start_pos:cuboid_start_pos + cuboid_num_points, :], cuboid_size_xt)

        if flags.record_u_max:
            # x-dimension
            sensor_data_temp[cuboid_index].ux_max =  np.reshape(
                sensor_data.ux_max[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].uy_max =  np.reshape(
                    sensor_data.uy_max[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].uz_max =  np.reshape(
                    sensor_data.uz_max[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)

        if flags.record_u_min:
            # x-dimension
            sensor_data_temp[cuboid_index].ux_min =  np.reshape(
                sensor_data.ux_min[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].uy_min =  np.reshape(
                    sensor_data.uy_min[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].uz_min =  np.reshape(
                    sensor_data.uz_min[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)

        if flags.record_u_rms:
            # x-dimension
            sensor_data_temp[cuboid_index].ux_rms =  np.reshape(
                sensor_data.ux_rms[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].uy_rms =  np.reshape(
                    sensor_data.uy_rms[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].uz_rms =  np.reshape(
                    sensor_data.uz_rms[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)

        if flags.record_I:
            # x-dimension
            sensor_data_temp[cuboid_index].Ix =  np.reshape(
                sensor_data.Ix[cuboid_start_pos:cuboid_start_pos + cuboid_num_points , :], cuboid_size_xt)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].Iy =  np.reshape(
                    sensor_data.Iy[cuboid_start_pos:cuboid_start_pos + cuboid_num_points , :], cuboid_size_xt)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].Iz =  np.reshape(
                    sensor_data.Iz[cuboid_start_pos:cuboid_start_pos + cuboid_num_points , :], cuboid_size_xt)

        if flags.record_I_avg:
            # x-dimension
            sensor_data_temp[cuboid_index].Ix_avg =  np.reshape(
                sensor_data.Ix_avg[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # y-dimension if 2D or 3D
            if kgrid.dim > 1:
                sensor_data_temp[cuboid_index].Iy_avg =  np.reshape(
                    sensor_data.Iy_avg[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)
            # z-dimension if 3D
            if kgrid.dim > 2:
                sensor_data_temp[cuboid_index].Iz_avg =  np.reshape(
                    sensor_data.Iz_avg[cuboid_start_pos:cuboid_start_pos + cuboid_num_points ], cuboid_size_x)

        # update cuboid index variable
        cuboid_start_pos = cuboid_start_pos + cuboid_num_points


    # assign max and final variables
    if flags.record_p_final:
        sensor_data_temp[0].p_final = sensor_data.p_final

    if flags.record_p_max_all:
        sensor_data_temp[0].p_max_all = sensor_data.p_max_all

    if flags.record_p_min_all:
        sensor_data_temp[0].p_min_all = sensor_data.p_min_all

    if flags.record_u_final:
        # x-dimension
        sensor_data_temp[0].ux_final = sensor_data.ux_final
        # y-dimension if 2D or 3D
        if kgrid.dim > 1:
            sensor_data_temp[0].uy_final = sensor_data.uy_final
        # z-dimension if 3D
        if kgrid.dim > 2:
            sensor_data_temp[0].uz_final = sensor_data.uz_final

    if flags.record_u_max_all:
        # x-dimension
        sensor_data_temp[0].ux_max_all = sensor_data.ux_max_all
        # y-dimension if 2D or 3D
        if kgrid.dim > 1:
            sensor_data_temp[0].uy_max_all = sensor_data.uy_max_all
        # z-dimension if 3D
        if kgrid.dim > 2:
            sensor_data_temp[0].uz_max_all = sensor_data.uz_max_all

    if flags.record_u_min_all:
        # x-dimension
        sensor_data_temp[0].ux_min_all = sensor_data.ux_min_all
        # y-dimension if 2D or 3D
        if kgrid.dim > 1:
            sensor_data_temp[0].uy_min_all = sensor_data.uy_min_all
        # z-dimension if 3D
        if kgrid.dim > 2:
            sensor_data_temp[0].uz_min_all = sensor_data.uz_min_all

    # assign new sensor data to old
    sensor_data = sensor_data_temp

    return sensor_data
