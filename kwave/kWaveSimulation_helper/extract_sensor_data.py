import numpy as np

def extract_sensor_data(dim: int, sensor_data, file_index, sensor_mask_index,
                        flags, record, p, ux_sgx, uy_sgy=None, uz_sgz=None):
    """
    extract_sensor_data Sample field variables at the sensor locations.
    """

    # =========================================================================
    # GRID STAGGERING
    # =========================================================================

    # shift the components of the velocity field onto the non-staggered
    # grid if required for output
    if (flags.record_u_non_staggered or flags.record_I or flags.record_I_avg):
        if (dim == 1):
            ux_shifted = np.real(np.fft.ifft(record.x_shift_neg * np.fft.fft(ux_sgx, axis=0), axis=0))
        elif (dim == 2):
            ux_shifted = np.real(np.fft.ifft(record.x_shift_neg * np.fft.fft(ux_sgx, axis=0), axis=0))
            uy_shifted = np.real(np.fft.ifft(record.y_shift_neg * np.fft.fft(uy_sgy, axis=1), axis=1))
        elif (dim == 3):
            ux_shifted = np.real(np.fft.ifft(record.x_shift_neg * np.fft.fft(ux_sgx, axis=0), axis=0))
            uy_shifted = np.real(np.fft.ifft(record.y_shift_neg * np.fft.fft(uy_sgy, axis=1), axis=1))
            uz_shifted = np.real(np.fft.ifft(record.z_shift_neg * np.fft.fft(uz_sgz, axis=2), axis=2))
        else:
            raise RuntimeError("Wrong dimensions")

    # =========================================================================
    # BINARY SENSOR MASK
    # =========================================================================

    if flags.binary_sensor_mask and not flags.use_cuboid_corners:

        # store the time history of the acoustic pressure
        if (flags.record_p or flags.record_I or flags.record_I_avg):
            if not flags.compute_directivity:
                sensor_data.p[:, file_index] = np.squeeze(p[np.unravel_index(sensor_mask_index, np.shape(p), order='F')])
            else:
              raise NotImplementedError('directivity not used at the moment')

        # store the maximum acoustic pressure
        if flags.record_p_max:
            if file_index == 0:
                sensor_data.p_max = p[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(p), order='F')]
            else:
                sensor_data.p_max = np.maximum(sensor_data.p_max,
                                               p[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(p), order='F')])

        # store the minimum acoustic pressure
        if flags.record_p_min:
            if file_index == 0:
                sensor_data.p_min = p[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(p), order='F')]
            else:
                sensor_data.p_min = np.minimum(sensor_data.p_min,
                                               p[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(p), order='F')])

        # store the rms acoustic pressure
        if flags.record_p_rms:
            if file_index == 0:
                sensor_data.p_rms = p[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(p), order='F')]**2
            else:
                sensor_data.p_rms = np.sqrt((sensor_data.p_rms**2 * file_index +
                                         p[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(p), order='F')]**2) / (file_index + 1) )

        # store the time history of the particle velocity on the staggered grid
        if flags.record_u:
            if (dim == 1):
                sensor_data.ux[:, file_index] = ux_sgx[sensor_mask_index]
            elif (dim == 2):
                sensor_data.ux[:, file_index] = ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]
                sensor_data.uy[:, file_index] = uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]
            elif (dim == 3):
                sensor_data.ux[:, file_index] = ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(ux_sgx), order='F')]
                sensor_data.uy[:, file_index] = uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(uy_sgy), order='F')]
                sensor_data.uz[:, file_index] = uz_sgz[np.unravel_index(np.squeeze(sensor_mask_index), np.shape(uz_sgz), order='F')]
            else:
                raise RuntimeError("Wrong dimensions")

        # store the time history of the particle velocity
        if flags.record_u_non_staggered or flags.record_I or flags.record_I_avg:
            if (dim == 1):
                sensor_data.ux_non_staggered[:, file_index] = ux_shifted[sensor_mask_index]
            elif (dim == 2):
                sensor_data.ux_non_staggered[:, file_index] = ux_shifted[np.unravel_index(np.squeeze(sensor_mask_index), ux_shifted.shape, order='F')]
                sensor_data.uy_non_staggered[:, file_index] = uy_shifted[np.unravel_index(np.squeeze(sensor_mask_index), uy_shifted.shape, order='F')]
            elif (dim == 3):
                sensor_data.ux_non_staggered[:, file_index] = ux_shifted[np.unravel_index(np.squeeze(sensor_mask_index), ux_shifted.shape, order='F')]
                sensor_data.uy_non_staggered[:, file_index] = uy_shifted[np.unravel_index(np.squeeze(sensor_mask_index), uy_shifted.shape, order='F')]
                sensor_data.uz_non_staggered[:, file_index] = uz_shifted[np.unravel_index(np.squeeze(sensor_mask_index), uz_shifted.shape, order='F')]
            else:
                raise RuntimeError("Wrong dimensions")

        # store the split components of the particle velocity
        if flags.record_u_split_field:
            if (dim == 2):

                # compute forward FFTs
                ux_k = record.x_shift_neg * np.fft.fftn(ux_sgx)
                uy_k = record.y_shift_neg * np.fft.fftn(uy_sgy)

                # ux compressional
                split_field = np.real(np.fft.ifftn(record.kx_norm**2 * ux_k + record.kx_norm * record.ky_norm * uy_k))
                sensor_data.ux_split_p[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # ux shear
                split_field = np.real(np.fft.ifftn((1.0 - record.kx_norm**2) * ux_k - record.kx_norm * record.ky_norm * uy_k))
                sensor_data.ux_split_s[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # uy compressional
                split_field = np.real(np.fft.ifftn(record.ky_norm * record.kx_norm * ux_k + record.ky_norm **2 * uy_k))
                sensor_data.uy_split_p[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # uy shear
                split_field = np.real(np.fft.ifftn(-record.ky_norm * record.kx_norm * ux_k + (1.0 - record.ky_norm**2) * uy_k))
                sensor_data.uy_split_s[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

            elif (dim == 3):

                # compute forward FFTs
                ux_k = np.multiply(record.x_shift_neg, np.fft.fftn(ux_sgx), order='F')
                uy_k = np.multiply(record.y_shift_neg, np.fft.fftn(uy_sgy), order='F')
                uz_k = np.multiply(record.z_shift_neg, np.fft.fftn(uz_sgz), order='F')

                # ux compressional
                split_field = np.real(np.fft.ifftn(record.kx_norm**2 * ux_k +
                                                   record.kx_norm * record.ky_norm * uy_k +
                                                   record.kx_norm * record.kz_norm * uz_k))
                sensor_data.ux_split_p[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # ux shear
                split_field = np.real(np.fft.ifftn((1.0 - record.kx_norm**2) * ux_k -
                                                   record.kx_norm * record.ky_norm * uy_k -
                                                   record.kx_norm * record.kz_norm * uz_k))
                sensor_data.ux_split_s[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # uy compressional
                split_field = np.real(np.fft.ifftn(record.ky_norm * record.kx_norm * ux_k +
                                                   record.ky_norm**2 * uy_k +
                                                   record.ky_norm * record.kz_norm * uz_k))
                sensor_data.uy_split_p[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # uy shear
                split_field = np.real(np.fft.ifftn(- record.ky_norm * record.kx_norm * ux_k +
                                                   (1.0 - record.ky_norm**2) * uy_k -
                                                   record.ky_norm * record.kz_norm * uz_k))
                sensor_data.uy_split_s[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # uz compressional
                split_field = np.real(np.fft.ifftn(record.kz_norm * record.kx_norm * ux_k +
                                                   record.kz_norm * record.ky_norm * uy_k +
                                                   record.kz_norm**2 * uz_k))
                sensor_data.uz_split_p[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                # uz shear
                split_field = np.real(np.fft.ifftn( -record.kz_norm * record.kx_norm * ux_k -
                                                   record.kz_norm * record.ky_norm * uy_k +
                                                   (1.0 - record.kz_norm**2) * uz_k))
                sensor_data.uz_split_s[:, file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]
            else:
                raise RuntimeError("Wrong dimensions")

        # store the maximum particle velocity
        if flags.record_u_max:
            if file_index == 0:
                if (dim == 1):
                    sensor_data.ux_max = ux_sgx[sensor_mask_index]
                elif (dim == 2):
                    sensor_data.ux_max = ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]
                    sensor_data.uy_max = uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]
                elif (dim == 3):
                    sensor_data.ux_max = ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]
                    sensor_data.uy_max = uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]
                    sensor_data.uz_max = uz_sgz[np.unravel_index(np.squeeze(sensor_mask_index), uz_sgz.shape, order='F')]
                else:
                    raise RuntimeError("Wrong dimensions")
            else:
                if (dim == 1):
                    sensor_data.ux_max = np.maximum(sensor_data.ux_max, ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')])
                elif (dim == 2):
                    sensor_data.ux_max = np.maximum(sensor_data.ux_max, ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')])
                    sensor_data.uy_max = np.maximum(sensor_data.uy_max, uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')])
                elif (dim == 3):
                    sensor_data.ux_max = np.maximum(sensor_data.ux_max, ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')])
                    sensor_data.uy_max = np.maximum(sensor_data.uy_max, uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')])
                    sensor_data.uz_max = np.maximum(sensor_data.uz_max, uz_sgz[np.unravel_index(np.squeeze(sensor_mask_index), uz_sgz.shape, order='F')])
                else:
                    raise RuntimeError("Wrong dimensions")

        # store the minimum particle velocity
        if flags.record_u_min:
            if file_index == 0:
                if (dim == 1):
                    sensor_data.ux_min = ux_sgx[sensor_mask_index]
                elif (dim == 2):
                    sensor_data.ux_min = ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]
                    sensor_data.uy_min = uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]
                elif (dim == 3):
                    sensor_data.ux_min = ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]
                    sensor_data.uy_min = uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]
                    sensor_data.uz_min = uz_sgz[np.unravel_index(np.squeeze(sensor_mask_index), uz_sgz.shape, order='F')]
                else:
                    raise RuntimeError("Wrong dimensions")
            else:
                if (dim == 1):
                    sensor_data.ux_min = np.minimum(sensor_data.ux_min, ux_sgx[sensor_mask_index])
                elif (dim == 2):
                    sensor_data.ux_min = np.minimum(sensor_data.ux_min, ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')])
                    sensor_data.uy_min = np.minimum(sensor_data.uy_min, uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')])
                elif (dim == 3):
                    sensor_data.ux_min = np.minimum(sensor_data.ux_min, ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')])
                    sensor_data.uy_min = np.minimum(sensor_data.uy_min, uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')])
                    sensor_data.uz_min = np.minimum(sensor_data.uz_min, uz_sgz[np.unravel_index(np.squeeze(sensor_mask_index), uz_sgz.shape, order='F')])
                else:
                    raise RuntimeError("Wrong dimensions")

        # store the rms particle velocity
        if flags.record_u_rms:
            if (dim == 1):
                sensor_data.ux_rms = np.sqrt((sensor_data.ux_rms**2 * (file_index - 0) +
                                                 ux_sgx[sensor_mask_index]**2) / (file_index +1))
            elif (dim == 2):
                sensor_data.ux_rms = np.sqrt((sensor_data.ux_rms**2 * (file_index - 0) +
                                                 ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]**2) / (file_index +1))
                sensor_data.uy_rms = np.sqrt((sensor_data.uy_rms**2 * (file_index - 0) +
                                                 uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]**2) / (file_index +1))
            elif (dim == 3):
                sensor_data.ux_rms = np.sqrt((sensor_data.ux_rms**2 * (file_index - 0) +
                                                 ux_sgx[np.unravel_index(np.squeeze(sensor_mask_index), ux_sgx.shape, order='F')]**2) / (file_index +1))
                sensor_data.uy_rms = np.sqrt((sensor_data.uy_rms**2 * (file_index - 0) +
                                                 uy_sgy[np.unravel_index(np.squeeze(sensor_mask_index), uy_sgy.shape, order='F')]**2) / (file_index +1))
                sensor_data.uz_rms = np.sqrt((sensor_data.uz_rms**2 * (file_index - 0) +
                                                 uz_sgz[np.unravel_index(np.squeeze(sensor_mask_index), uz_sgz.shape, order='F')]**2) / (file_index +1))


    # =========================================================================
    # CARTESIAN SENSOR MASK
    # =========================================================================

    elif flags.use_cuboid_corners:

        n_cuboids: int = np.shape(record.cuboid_corners_list)[1]

        # s_start: int = 0

        # for each cuboid
        for cuboid_index in np.arange(n_cuboids):

          # get size of cuboid for indexing regions of computational grid
            if dim == 1:
                cuboid_size_x = [record.cuboid_corners_list[1, cuboid_index] - record.cuboid_corners_list[0, cuboid_index], 1]
            elif dim == 2:
                cuboid_size_x = [record.cuboid_corners_list[2, cuboid_index] - record.cuboid_corners_list[0, cuboid_index],
                                 record.cuboid_corners_list[3, cuboid_index] - record.cuboid_corners_list[1, cuboid_index]]
            elif dim == 3:
                cuboid_size_x = [record.cuboid_corners_list[3, cuboid_index] + 1 - record.cuboid_corners_list[0, cuboid_index],
                                 record.cuboid_corners_list[4, cuboid_index] + 1 - record.cuboid_corners_list[1, cuboid_index],
                                 record.cuboid_corners_list[5, cuboid_index] + 1 - record.cuboid_corners_list[2, cuboid_index]]

            # cuboid_num_points: int = np.prod(cuboid_size_x)

            # s_end: int = s_start + cuboid_num_points

            # sensor_mask_sub_index = sensor_mask_index[s_start:s_end]

            # s_start = s_end

            x_indices = np.arange(record.cuboid_corners_list[0, cuboid_index], record.cuboid_corners_list[3, cuboid_index]+1, dtype=int)
            y_indices = np.arange(record.cuboid_corners_list[1, cuboid_index], record.cuboid_corners_list[4, cuboid_index]+1, dtype=int)
            z_indices = np.arange(record.cuboid_corners_list[2, cuboid_index], record.cuboid_corners_list[5, cuboid_index]+1, dtype=int)

            # Create a meshgrid
            xx, yy, zz = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')

            # Combine into a list of indices
            cuboid_indices = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

            result = p[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)

            # store the time history of the acoustic pressure
            if (flags.record_p or flags.record_I or flags.record_I_avg):
                if not flags.compute_directivity:
                    # Use the indices to index into p
                    sensor_data[cuboid_index].p[..., file_index] = result
                else:
                    raise NotImplementedError('directivity not implemented at the moment')

            # store the maximum acoustic pressure
            if flags.record_p_max:
                if file_index == 0:
                    sensor_data[cuboid_index].p_max = result
                else:
                    sensor_data[cuboid_index].p_max = np.maximum(sensor_data[cuboid_index].p_max, result)

            # store the minimum acoustic pressure
            if flags.record_p_min:
                if file_index == 0:
                    sensor_data[cuboid_index].p_min = result
                else:
                    sensor_data[cuboid_index].p_min = np.maximum(sensor_data[cuboid_index].p_min, result)

            # store the rms acoustic pressure
            if flags.record_p_rms:
                if file_index == 0:
                    sensor_data[cuboid_index].p_rms = result**2
                else:
                    sensor_data[cuboid_index].p_rms = np.sqrt((sensor_data[cuboid_index].p_rms**2 * file_index + result**2) / (file_index + 1) )

            # store the time history of the particle velocity on the staggered grid
            if flags.record_u:
                if (dim == 1):
                    sensor_data[cuboid_index].ux[..., file_index] = ux_sgx[cuboid_indices]
                elif (dim == 2):
                    sensor_data[cuboid_index].ux[..., file_index] = ux_sgx[cuboid_indices]
                    sensor_data[cuboid_index].uy[..., file_index] = uy_sgy[cuboid_indices]
                elif (dim == 3):
                    sensor_data[cuboid_index].ux[..., file_index] = ux_sgx[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                    sensor_data[cuboid_index].uy[..., file_index] = uy_sgy[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                    sensor_data[cuboid_index].uz[..., file_index] = uz_sgz[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                else:
                    raise RuntimeError("Wrong dimensions")

            # store the time history of the particle velocity
            if flags.record_u_non_staggered or flags.record_I or flags.record_I_avg:
                if (dim == 1):
                    sensor_data[cuboid_index].ux_non_staggered[..., file_index] = ux_shifted[cuboid_indices]
                elif (dim == 2):
                    sensor_data[cuboid_index].ux_non_staggered[..., file_index] = ux_shifted[cuboid_indices]
                    sensor_data[cuboid_index].uy_non_staggered[..., file_index] = uy_shifted[cuboid_indices]
                elif (dim == 3):
                    sensor_data[cuboid_index].ux_non_staggered[..., file_index] = ux_shifted[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                    sensor_data[cuboid_index].uy_non_staggered[..., file_index] = uy_shifted[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                    sensor_data[cuboid_index].uz_non_staggered[..., file_index] = uz_shifted[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                else:
                    raise RuntimeError("Wrong dimensions")

            # store the split components of the particle velocity
            if flags.record_u_split_field:
                if (dim == 2):

                    # compute forward FFTs
                    ux_k = record.x_shift_neg * np.fft.fftn(ux_sgx)
                    uy_k = record.y_shift_neg * np.fft.fftn(uy_sgy)

                    # ux compressional
                    split_field = np.real(np.fft.ifftn(record.kx_norm**2 * ux_k + record.kx_norm * record.ky_norm * uy_k))
                    sensor_data.ux_split_p[..., file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                    # ux shear
                    split_field = np.real(np.fft.ifftn((1.0 - record.kx_norm**2) * ux_k - record.kx_norm * record.ky_norm * uy_k))
                    sensor_data[cuboid_index].ux_split_s[..., file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                    # uy compressional
                    split_field = np.real(np.fft.ifftn(record.ky_norm * record.kx_norm * ux_k + record.ky_norm **2 * uy_k))
                    sensor_data[cuboid_index].uy_split_p[..., file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                    # uy shear
                    split_field = np.real(np.fft.ifftn(-record.ky_norm * record.kx_norm * ux_k + (1.0 - record.ky_norm**2) * uy_k))
                    sensor_data[cuboid_index].uy_split_s[..., file_index] = split_field[np.unravel_index(np.squeeze(sensor_mask_index), split_field.shape, order='F')]

                elif (dim == 3):

                    # compute forward FFTs
                    ux_k = np.multiply(record.x_shift_neg, np.fft.fftn(ux_sgx), order='F')
                    uy_k = np.multiply(record.y_shift_neg, np.fft.fftn(uy_sgy), order='F')
                    uz_k = np.multiply(record.z_shift_neg, np.fft.fftn(uz_sgz), order='F')

                    # ux compressional
                    split_field = np.real(np.fft.ifftn(record.kx_norm**2 * ux_k +
                                                      record.kx_norm * record.ky_norm * uy_k +
                                                      record.kx_norm * record.kz_norm * uz_k))
                    sensor_data[cuboid_index].ux_split_p[..., file_index] = split_field[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                    # ux shear
                    split_field = np.real(np.fft.ifftn((1.0 - record.kx_norm**2) * ux_k -
                                                      record.kx_norm * record.ky_norm * uy_k -
                                                      record.kx_norm * record.kz_norm * uz_k))
                    sensor_data[cuboid_index].ux_split_s[..., file_index] = split_field[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)

                    # uy compressional
                    split_field = np.real(np.fft.ifftn(record.ky_norm * record.kx_norm * ux_k +
                                                      record.ky_norm**2 * uy_k +
                                                      record.ky_norm * record.kz_norm * uz_k))
                    sensor_data[cuboid_index].uy_split_p[..., file_index] = split_field[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)

                    # uy shear
                    split_field = np.real(np.fft.ifftn(- record.ky_norm * record.kx_norm * ux_k +
                                                      (1.0 - record.ky_norm**2) * uy_k -
                                                      record.ky_norm * record.kz_norm * uz_k))
                    sensor_data[cuboid_index].uy_split_s[..., file_index] = split_field[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)

                    # uz compressional
                    split_field = np.real(np.fft.ifftn(record.kz_norm * record.kx_norm * ux_k +
                                                      record.kz_norm * record.ky_norm * uy_k +
                                                      record.kz_norm**2 * uz_k))
                    sensor_data[cuboid_index].uz_split_p[..., file_index] = split_field[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)

                    # uz shear
                    split_field = np.real(np.fft.ifftn( -record.kz_norm * record.kx_norm * ux_k -
                                                      record.kz_norm * record.ky_norm * uy_k +
                                                      (1.0 - record.kz_norm**2) * uz_k))
                    sensor_data[cuboid_index].uz_split_s[..., file_index] = split_field[cuboid_indices[:, 0], cuboid_indices[:, 1], cuboid_indices[:, 2]].reshape(cuboid_size_x)
                else:
                    raise RuntimeError("Wrong dimensions")









    # =========================================================================
    # CARTESIAN SENSOR MASK
    # =========================================================================

    # extract data from specified Cartesian coordinates using interpolation
    # (record.tri and record.bc are the Delaunay triangulation and Barycentric coordinates
    # returned by gridDataFast3D)
    else:

        # store the time history of the acoustic pressure
        if flags.record_p or flags.record_I or flags.record_I_avg:
            if dim == 1:
                sensor_data.p[:, file_index] = np.interp(np.squeeze(record.sensor_x), np.squeeze(record.grid_x), p)
            else:
                sensor_data.p[:, file_index] = np.sum(p[record.tri] * record.bc, axis=1)

        # store the maximum acoustic pressure
        if flags.record_p_max:
            if dim == 1:
                if file_index == 0:
                    sensor_data.p_max = np.interp(record.grid_x, p, record.sensor_x)
                else:
                    sensor_data.p_max = np.maximum(sensor_data.p_max, np.interp(record.grid_x, p, record.sensor_x))
            else:
                if file_index == 0:
                    sensor_data.p_max = np.sum(p[record.tri] * record.bc, axis=1)
                else:
                    sensor_data.p_max = np.maximum(sensor_data.p_max, np.sum(p[record.tri] * record.bc, axis=1))

        # store the minimum acoustic pressure
        if flags.record_p_min:
            if dim == 1:
                if file_index == 0:
                    sensor_data.p_min = np.interp(record.grid_x, p, record.sensor_x)
                else:
                    sensor_data.p_min = np.minimum(sensor_data.p_min, np.interp(record.grid_x, p, record.sensor_x))
            else:
                if file_index == 0:
                    sensor_data.p_min = np.sum(p[record.tri] * record.bc, axis=1)
                else:
                    sensor_data.p_min = np.minimum(sensor_data.p_min, np.sum(p[record.tri] * record.bc, axis=1))

        # store the rms acoustic pressure
        if flags.record_p_rms:
            if dim == 1:
                sensor_data.p_rms = np.sqrt((sensor_data.p_rms**2 * (file_index - 0) + (np.interp(record.grid_x, p, record.sensor_x))**2) / (file_index +1))
            else:
                sensor_data.p_rms[:] = np.sqrt((sensor_data.p_rms[:]**2 * (file_index - 0) +
                                                (np.sum(p[record.tri] * record.bc, axis=1))**2) / (file_index +1))

        # store the time history of the particle velocity on the staggered grid
        if flags.record_u:
            if (dim == 1):
                sensor_data.ux[:, file_index] = np.interp(record.grid_x, ux_sgx, record.sensor_x)
            elif (dim == 2):
                sensor_data.ux[:, file_index] = np.sum(ux_sgx[record.tri] * record.bc, axis=1)
                sensor_data.uy[:, file_index] = np.sum(uy_sgy[record.tri] * record.bc, axis=1)
            elif (dim == 3):
                sensor_data.ux[:, file_index] = np.sum(ux_sgx[record.tri] * record.bc, axis=1)
                sensor_data.uy[:, file_index] = np.sum(uy_sgy[record.tri] * record.bc, axis=1)
                sensor_data.uz[:, file_index] = np.sum(uz_sgz[record.tri] * record.bc, axis=1)
            else:
                raise RuntimeError("Wrong dimensions")

        # store the time history of the particle velocity
        if flags.record_u_non_staggered or flags.record_I or flags.record_I_avg:
            if (dim == 1):
                sensor_data.ux_non_staggered[:, file_index] = np.interp(record.grid_x, ux_shifted, record.sensor_x)
            elif (dim == 2):
                sensor_data.ux_non_staggered[:, file_index] = np.sum(ux_shifted[record.tri] * record.bc, axis=1)
                sensor_data.uy_non_staggered[:, file_index] = np.sum(uy_shifted[record.tri] * record.bc, axis=1)
            elif (dim == 3):
                sensor_data.ux_non_staggered[:, file_index] = np.sum(ux_shifted[record.tri] * record.bc, axis=1)
                sensor_data.uy_non_staggered[:, file_index] = np.sum(uy_shifted[record.tri] * record.bc, axis=1)
                sensor_data.uz_non_staggered[:, file_index] = np.sum(uz_shifted[record.tri] * record.bc, axis=1)
            else:
                raise RuntimeError("Wrong dimensions")

        # store the maximum particle velocity
        if flags.record_u_max:
            if file_index == 0:
                if (dim == 1):
                    sensor_data.ux_max = np.interp(record.grid_x, ux_sgx, record.sensor_x)
                elif (dim == 2):
                    sensor_data.ux_max = np.sum(ux_sgx[record.tri] * record.bc, axis=1)
                    sensor_data.uy_max = np.sum(uy_sgy[record.tri] * record.bc, axis=1)
                elif (dim == 3):
                    sensor_data.ux_max = np.sum(ux_sgx[record.tri] * record.bc, axis=1)
                    sensor_data.uy_max = np.sum(uy_sgy[record.tri] * record.bc, axis=1)
                    sensor_data.uz_max = np.sum(uz_sgz[record.tri] * record.bc, axis=1)
                else:
                    raise RuntimeError("Wrong dimensions")
            else:
                if (dim == 1):
                    sensor_data.ux_max = np.maximum(sensor_data.ux_max, np.interp(record.grid_x, ux_sgx, record.sensor_x))
                elif (dim == 2):
                    sensor_data.ux_max = np.maximum(sensor_data.ux_max, np.sum(ux_sgx[record.tri] * record.bc, axis=1))
                    sensor_data.uy_max = np.maximum(sensor_data.uy_max, np.sum(uy_sgy[record.tri] * record.bc, axis=1))
                elif (dim == 3):
                    sensor_data.ux_max = np.maximum(sensor_data.ux_max, np.sum(ux_sgx[record.tri] * record.bc, axis=1))
                    sensor_data.uy_max = np.maximum(sensor_data.uy_max, np.sum(uy_sgy[record.tri] * record.bc, axis=1))
                    sensor_data.uz_max = np.maximum(sensor_data.uz_max, np.sum(uz_sgz[record.tri] * record.bc, axis=1))
                else:
                    raise RuntimeError("Wrong dimensions")

        # store the minimum particle velocity
        if flags.record_u_min:
            if file_index == 0:
                if (dim == 1):
                    sensor_data.ux_min = np.interp(record.grid_x, ux_sgx, record.sensor_x)
                elif (dim == 2):
                    sensor_data.ux_min = np.sum(ux_sgx[record.tri] * record.bc, axis=1)
                    sensor_data.uy_min = np.sum(uy_sgy[record.tri] * record.bc, axis=1)
                elif (dim == 3):
                    sensor_data.ux_min = np.sum(ux_sgx[record.tri] * record.bc, axis=1)
                    sensor_data.uy_min = np.sum(uy_sgy[record.tri] * record.bc, axis=1)
                    sensor_data.uz_min = np.sum(uz_sgz[record.tri] * record.bc, axis=1)
                else:
                    raise RuntimeError("Wrong dimensions")

            else:
                if (dim == 1):
                    sensor_data.ux_min = np.minimum(sensor_data.ux_min, np.interp(record.grid_x, ux_sgx, record.sensor_x))
                elif (dim == 2):
                    sensor_data.ux_min = np.minimum(sensor_data.ux_min, np.sum(ux_sgx[record.tri] * record.bc, axis=1))
                    sensor_data.uy_min = np.minimum(sensor_data.uy_min, np.sum(uy_sgy[record.tri] * record.bc, axis=1))
                elif (dim == 3):
                    sensor_data.ux_min = np.minimum(sensor_data.ux_min, np.sum(ux_sgx[record.tri] * record.bc, axis=1))
                    sensor_data.uy_min = np.minimum(sensor_data.uy_min, np.sum(uy_sgy[record.tri] * record.bc, axis=1))
                else:
                    raise RuntimeError("Wrong dimensions")

        # store the rms particle velocity
        if flags.record_u_rms:
            if (dim == 1):
                sensor_data.ux_rms = np.sqrt((sensor_data.ux_rms**2 * (file_index - 0) + (np.interp(record.grid_x, ux_sgx, record.sensor_x))**2) / (file_index +1))
            elif (dim == 2):
                sensor_data.ux_rms[:] = np.sqrt((sensor_data.ux_rms[:]**2 * (file_index - 0) + (np.sum(ux_sgx[record.tri] * record.bc, axis=1))**2) / (file_index +1))
                sensor_data.uy_rms[:] = np.sqrt((sensor_data.uy_rms[:]**2 * (file_index - 0) + (np.sum(uy_sgy[record.tri] * record.bc, axis=1))**2) / (file_index +1))
            elif (dim == 3):
                sensor_data.ux_rms[:] = np.sqrt((sensor_data.ux_rms[:]**2 * (file_index - 0) + (np.sum(ux_sgx[record.tri] * record.bc, axis=1))**2) / (file_index +1))
                sensor_data.uy_rms[:] = np.sqrt((sensor_data.uy_rms[:]**2 * (file_index - 0) + (np.sum(uy_sgy[record.tri] * record.bc, axis=1))**2) / (file_index +1))
                sensor_data.uz_rms[:] = np.sqrt((sensor_data.uz_rms[:]**2 * (file_index - 0) + (np.sum(uz_sgz[record.tri] * record.bc, axis=1))**2) / (file_index +1))
            else:
                raise RuntimeError("Wrong dimensions")

    # =========================================================================
    # RECORDED VARIABLES OVER ENTIRE GRID
    # =========================================================================

    # store the maximum acoustic pressure over all the grid elements
    if flags.record_p_max_all and (not flags.use_cuboid_corners):
        if (dim == 1):
            if file_index == 0:
                sensor_data.p_max_all = p[record.x1_inside:record.x2_inside]
            else:
                sensor_data.p_max_all = np.maximum(sensor_data.p_max_all, p[record.x1_inside:record.x2_inside])

        elif (dim == 2):
            if file_index == 0:
                sensor_data.p_max_all = p[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
            else:
                sensor_data.p_max_all = np.maximum(sensor_data.p_max_all, p[record.x1_inside:record.x2_inside,
                                                                            record.y1_inside:record.y2_inside])

        elif (dim == 3):
            if file_index == 0:
                sensor_data.p_max_all = p[record.x1_inside:record.x2_inside,
                                          record.y1_inside:record.y2_inside,
                                          record.z1_inside:record.z2_inside]
            else:
                sensor_data.p_max_all = np.maximum(sensor_data.p_max_all,
                                                p[record.x1_inside:record.x2_inside,
                                                  record.y1_inside:record.y2_inside,
                                                  record.z1_inside:record.z2_inside])
        else:
            raise RuntimeError("Wrong dimensions")

    # store the minimum acoustic pressure over all the grid elements
    if flags.record_p_min_all and (not flags.use_cuboid_corners):
        if (dim == 1):
            if file_index == 0:
                sensor_data.p_min_all = p[record.x1_inside:record.x2_inside]
            else:
                sensor_data.p_min_all = np.minimum(sensor_data.p_min_all, p[record.x1_inside:record.x2_inside])

        elif (dim == 2):
            if file_index == 0:
                sensor_data.p_min_all = p[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
            else:
                sensor_data.p_min_all = np.minimum(sensor_data.p_min_all, p[record.x1_inside:record.x2_inside,
                                                                            record.y1_inside:record.y2_inside])

        elif (dim == 3):
            if file_index == 0:
                sensor_data.p_min_all = p[record.x1_inside:record.x2_inside,
                                          record.y1_inside:record.y2_inside,
                                          record.z1_inside:record.z2_inside]
            else:
                sensor_data.p_min_all = np.minimum(sensor_data.p_min_all,
                                                p[record.x1_inside:record.x2_inside,
                                                  record.y1_inside:record.y2_inside,
                                                  record.z1_inside:record.z2_inside])
        else:
            raise RuntimeError("Wrong dimensions")

    # store the maximum particle velocity over all the grid elements
    if flags.record_u_max_all and (not flags.use_cuboid_corners):
        if (dim == 1):
            if file_index == 0:
                sensor_data.ux_max_all = ux_sgx[record.x1_inside:record.x2_inside]
            else:
                sensor_data.ux_max_all = np.maximum(sensor_data.ux_max_all, ux_sgx[record.x1_inside:record.x2_inside])

        elif (dim == 2):
            if file_index == 0:
                sensor_data.ux_max_all = ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
                sensor_data.uy_max_all = uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
            else:
                sensor_data.ux_max_all = np.maximum(sensor_data.ux_max_all,
                                                  ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside])
                sensor_data.uy_max_all = np.maximum(sensor_data.uy_max_all,
                                                  uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside])

        elif (dim == 3):
            if file_index == 0:
                sensor_data.ux_max_all = ux_sgx[record.x1_inside:record.x2_inside,
                                                record.y1_inside:record.y2_inside,
                                                record.z1_inside:record.z2_inside]
                sensor_data.uy_max_all = uy_sgy[record.x1_inside:record.x2_inside,
                                                record.y1_inside:record.y2_inside,
                                                record.z1_inside:record.z2_inside]
                sensor_data.uz_max_all = uz_sgz[record.x1_inside:record.x2_inside,
                                                record.y1_inside:record.y2_inside,
                                                record.z1_inside:record.z2_inside]
            else:
                sensor_data.ux_max_all = np.maximum(sensor_data.ux_max_all,
                                                  ux_sgx[record.x1_inside:record.x2_inside,
                                                         record.y1_inside:record.y2_inside,
                                                         record.z1_inside:record.z2_inside])
                sensor_data.uy_max_all = np.maximum(sensor_data.uy_max_all,
                                                  uy_sgy[record.x1_inside:record.x2_inside,
                                                         record.y1_inside:record.y2_inside,
                                                         record.z1_inside:record.z2_inside])
                sensor_data.uz_max_all = np.maximum(sensor_data.uz_max_all,
                                                  uz_sgz[record.x1_inside:record.x2_inside,
                                                         record.y1_inside:record.y2_inside,
                                                         record.z1_inside:record.z2_inside])

        else:
            raise RuntimeError("Wrong dimensions")

    # store the minimum particle velocity over all the grid elements
    if flags.record_u_min_all and (not flags.use_cuboid_corners):
        if (dim == 1):
            if file_index == 0:
                sensor_data.ux_min_all = ux_sgx[record.x1_inside:record.x2_inside]
            else:
                sensor_data.ux_min_all = np.minimum(sensor_data.ux_min_all, ux_sgx[record.x1_inside:record.x2_inside])

        elif (dim == 2):
            if file_index == 0:
                sensor_data.ux_min_all = ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
                sensor_data.uy_min_all = uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
            else:
                sensor_data.ux_min_all = np.minimum(sensor_data.ux_min_all,
                                                  ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside])
                sensor_data.uy_min_all = np.minimum(sensor_data.uy_min_all,
                                                  uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside])

        elif (dim == 3):
            if file_index == 0:
                sensor_data.ux_min_all = ux_sgx[record.x1_inside:record.x2_inside,
                                                record.y1_inside:record.y2_inside,
                                                record.z1_inside:record.z2_inside]
                sensor_data.uy_min_all = uy_sgy[record.x1_inside:record.x2_inside,
                                                record.y1_inside:record.y2_inside,
                                                record.z1_inside:record.z2_inside]
                sensor_data.uz_min_all = uz_sgz[record.x1_inside:record.x2_inside,
                                                record.y1_inside:record.y2_inside,
                                                record.z1_inside:record.z2_inside]
            else:
                sensor_data.ux_min_all = np.minimum(sensor_data.ux_min_all,
                                                  ux_sgx[record.x1_inside:record.x2_inside,
                                                         record.y1_inside:record.y2_inside,
                                                         record.z1_inside:record.z2_inside])
                sensor_data.uy_min_all = np.minimum(sensor_data.uy_min_all,
                                                  uy_sgy[record.x1_inside:record.x2_inside,
                                                         record.y1_inside:record.y2_inside,
                                                         record.z1_inside:record.z2_inside])
                sensor_data.uz_min_all = np.minimum(sensor_data.uz_min_all,
                                                  uz_sgz[record.x1_inside:record.x2_inside,
                                                         record.y1_inside:record.y2_inside,
                                                         record.z1_inside:record.z2_inside])
        else:
            raise RuntimeError("Wrong dimensions")

    return sensor_data