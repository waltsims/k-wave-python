import numpy as np
from kwave.utils.math import fourier_shift

def save_intensity(kgrid, sensor_data, record_I_avg: bool, use_cuboid_corners: bool):
    """
    save_intensity is a method to calculate the acoustic intensity from the time
    varying acoustic pressure and particle velocity recorded during the simulation.
    The particle velocity is first temporally shifted forwards by dt/2 using a
    Fourier interpolant so both variables are on the regular (non-staggered) grid.

    This function is called before the binary sensor data is reordered
    for cuboid corners, so it works for both types of sensor mask.

    If using cuboid corners the sensor data may be given as a structure
    array, i.e., sensor_data(n).ux_non_staggered. In this case, the
    calculation of intensity is applied to each cuboid sensor mask separately.

    ABOUT:
        author      - Bradley Treeby
        date        - 4th September 2013
        last update - 15th May 2018

    This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    Copyright (C) 2013-2018 Bradley Treeby

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

    shift = 0.5


    # loop through the number of sensor masks (this can be > 1 if using cuboid
    # corners)
    if use_cuboid_corners:
        for sensor_mask_index in len(sensor_data):
            # shift the recorded particle velocity to the regular (non-staggered)
            # temporal grid and then compute the time varying intensity
            ux_sgt = fourier_shift(sensor_data[sensor_mask_index].ux_non_staggered, shift=shift)
            sensor_data[sensor_mask_index].Ix = np.mutiply(sensor_data[sensor_mask_index].p, ux_sgt, order='F')
            if kgrid.dim > 1:
                uy_sgt = fourier_shift(sensor_data[sensor_mask_index].uy_non_staggered, shift=shift)
                sensor_data[sensor_mask_index].Iy = np.mutiply(sensor_data[sensor_mask_index].p, uy_sgt, order='F')
            if kgrid.dim > 2:
                uz_sgt = fourier_shift(sensor_data[sensor_mask_index].uz_non_staggered, shift=shift)
                sensor_data[sensor_mask_index].Iz = np.mutiply(sensor_data[sensor_mask_index].p, uz_sgt, order='F')

            # calculate the time average of the intensity if required using the last
            # dimension (this works for both linear and cuboid sensor masks)
            if record_I_avg:
                sensor_data[sensor_mask_index].Ix_avg = np.mean(sensor_data[sensor_mask_index].Ix,
                                                                axis=np.ndim(sensor_data[sensor_mask_index].Ix))
                if kgrid.dim > 1:
                    sensor_data[sensor_mask_index].Iy_avg = np.mean(sensor_data[sensor_mask_index].Iy,
                                                                    axis=np.ndim(sensor_data[sensor_mask_index].Iy))
                if kgrid.dim > 2:
                    sensor_data[sensor_mask_index].Iz_avg = np.mean(sensor_data[sensor_mask_index].Iz,
                                                                    axis=np.ndim(sensor_data[sensor_mask_index].Iz))
    else:
        # shift the recorded particle velocity to the regular (non-staggered)
        # temporal grid and then compute the time varying intensity
        ux_sgt = fourier_shift(sensor_data.ux_non_staggered, shift=shift)
        sensor_data.Ix = np.mutiply(sensor_data.p, ux_sgt, order='F')
        if kgrid.dim > 1:
            uy_sgt = fourier_shift(sensor_data.uy_non_staggered, shift=shift)
            sensor_data.Iy = np.mutiply(sensor_data.p, uy_sgt, order='F')
        if kgrid.dim > 2:
            uz_sgt = fourier_shift(sensor_data.uz_non_staggered, shift=shift)
            sensor_data.Iz = np.mutiply(sensor_data.p, uz_sgt, order='F')

        # calculate the time average of the intensity if required using the last
        # dimension (this works for both linear and cuboid sensor masks)
        if record_I_avg:
            sensor_data.Ix_avg = np.mean(sensor_data.Ix, axis=np.ndim(sensor_data.Ix))
            if kgrid.dim > 1:
                sensor_data.Iy_avg = np.mean(sensor_data.Iy, axis=np.ndim(sensor_data.Iy))
            if kgrid.dim > 2:
                sensor_data.Iz_avg = np.mean(sensor_data.Iz, axis=np.ndim(sensor_data.Iz))

    # # remove the non staggered particle velocity variables if not required
    # if not flags.record_u_non_staggered:
    #     if kgrid.dim == 1:
    #         sensor_data = rmfield(sensor_data, {'ux_non_staggered'})
    #     elif kgrid.dim == 2:
    #         sensor_data = rmfield(sensor_data, {'ux_non_staggered', 'uy_non_staggered'})
    #     elif kgrid.dim == 3:
    #         sensor_data = rmfield(sensor_data, {'ux_non_staggered', 'uy_non_staggered', 'uz_non_staggered'})

    # # remove the time varying intensity if not required
    # if not flags.record.I:
    #     if kgrid.dim == 1:
    #         sensor_data = rmfield(sensor_data, {'Ix'})
    #     elif kgrid.dim == 2:
    #         sensor_data = rmfield(sensor_data, {'Ix', 'Iy'})
    #     elif kgrid.dim == 3:
    #         sensor_data = rmfield(sensor_data, {'Ix', 'Iy', 'Iz'})

    # # remove the time varying pressure if not required
    # if not flags.record.p:
    #     sensor_data = rmfield(sensor_data, {'p'})

    return sensor_data
