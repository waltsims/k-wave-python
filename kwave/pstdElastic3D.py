# import numpy as np

# from typing import Union

# from kwave.data import Vector
# from kwave.kWaveSimulation_helper import display_simulation_params, set_sound_speed_ref, \
#   expand_grid_matrices, create_storage_variables, create_absorption_variables, \
#     scale_source_terms_func
# from kwave.kgrid import kWaveGrid
# from kwave.kmedium import kWaveMedium
# from kwave.ksensor import kSensor
# from kwave.ksource import kSource
# from kwave.ktransducer import NotATransducer
# from kwave.options.simulation_options import SimulationOptions, SimulationType
# from kwave.recorder import Recorder
# from kwave.utils.checks import check_stability
# from kwave.utils.colormap import get_color_map
# from kwave.utils.conversion import cast_to_type, cart2grid, db2neper
# from kwave.utils.data import get_smallest_possible_type, get_date_string, scale_time, scale_SI
# from kwave.utils.dotdictionary import dotdict
# from kwave.utils.filters import smooth, gaussian_filter
# from kwave.utils.matlab import matlab_find, matlab_mask
# from kwave.utils.matrix import num_dim2
# from kwave.utils.pml import get_pml
# from kwave.utils.tictoc import TicToc

import numpy as np
from scipy.interpolate import interpn
import scipy.io as sio
# from tqdm import tqdm
from typing import Union
# from termcolor import colored
# import cupy as cp

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kWaveSimulation import kWaveSimulation

from kwave.ktransducer import NotATransducer

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_time
# from kwave.utils.data import scale_SI
from kwave.utils.filters import gaussian_filter
# from kwave.utils.matlab import rem
from kwave.utils.pml import get_pml
from kwave.utils.signals import reorder_sensor_data
from kwave.utils.tictoc import TicToc
from kwave.utils.dotdictionary import dotdict

from kwave.options.simulation_options import SimulationOptions

from kwave.kWaveSimulation_helper import extract_sensor_data, reorder_cuboid_corners


# @jit(nopython=True)
# def add3(sxx_split_x, sxx_split_y, sxx_split_z):
#     return sxx_split_x + sxx_split_y + sxx_split_z


# @jit(nopython=True)
# def add2(sxx_split_x, sxx_split_y):
#     return sxx_split_x + sxx_split_y


def compute_stress_grad_3(sxx_split_x, sxx_split_y, sxx_split_z, ddx_k_shift_pos, axis: int = 0):
    temp = sxx_split_x + sxx_split_y + sxx_split_z
    return np.real(np.fft.ifft(ddx_k_shift_pos * np.fft.fft(temp, axis=axis), axis=axis))


def compute_stress_grad_2(sxx_split_x, sxx_split_y, ddx_k_shift_pos, axis=0):
    temp = sxx_split_x + sxx_split_y
    return np.real(np.fft.ifft(ddx_k_shift_pos * np.fft.fft(temp, axis=axis), axis=axis))


# @jit(nopython=True)
# def compute_syz_split_z(mpml_x, mpml_y_sgy, pml_z_sgz, syz_split_z, dt, mu_sgyz, duydz):
#     """
#     Not on trace
#     """
#     return mpml_x * mpml_y_sgy * pml_z_sgz * (mpml_x * mpml_y_sgy * pml_z_sgz * syz_split_z + dt * mu_sgyz * duydz)


# @jit(nopython=True)
# def compute_sxx_split_x(mpml_z, mpml_y, pml_x, sxx_split_x, dt, mu, lame_lambda, duxdx, eta, chi, dduxdxdt):
#     """
#     On trace
#     """
#     return mpml_z * mpml_y * pml_x * (mpml_z * mpml_y * pml_x * sxx_split_x + dt * (2.0 * mu + lame_lambda) * duxdx + dt * (2.0 * eta + chi) * dduxdxdt)


# @jit
# def jit_accelerated_fft_operation(ddx_k_shift_pos, temp):
#     """
#     Perform FFT, manipulate and inverse FFT
#     """
#     temp_fft = np.fft.fft(temp, axis=0)
#     result_fft = ddx_k_shift_pos * temp_fft
#     result_ifft = np.fft.ifft(result_fft, axis=0)
#     result_real = np.real(result_ifft)
#     return result_real

# # Assuming ddx_k_shift_pos and temp are numpy arrays, convert them to cupy arrays
# ddx_k_shift_pos_gpu = cp.asarray(ddx_k_shift_pos)
# temp_gpu = cp.asarray(temp)

# # Perform FFT, element-wise multiplication, and inverse FFT using CuPy on the GPU
# temp_fft_gpu = cp.fft.fft(temp_gpu, axis=0)
# result_fft_gpu = cp.multiply(ddx_k_shift_pos_gpu, temp_fft_gpu)
# result_ifft_gpu = cp.fft.ifft(result_fft_gpu, axis=0)
# result_real_gpu = cp.real(result_ifft_gpu)

# # Convert the result back to a NumPy array if necessary
# result_real = cp.asnumpy(result_real_gpu)

# def xp_accelerated_fft_operation(ddx_k_shift_pos, x):
#     xp = cp.get_array_module(x)
#     x_fft = xp.fft.fft(x, axis=0)
#     result_fft = xp.multiply(ddx_k_shift_pos, x_fft)
#     result_ifft = xp.fft.ifft(result_fft, axis=0)
#     result_real =  xp.real(result_ifft)
#     return result_real


# # update the normal components and shear components of stress tensor
# # using a split field pml
# @jit(nopython=True)
# def compute_syz_split_z(mpml_x, mpml_y_sgy, pml_z_sgz, syz_split_z, dt, mu_sgyz, duydz):
#     return mpml_x * mpml_y_sgy * pml_z_sgz * (mpml_x * mpml_y_sgy * pml_z_sgz * syz_split_z + dt * mu_sgyz * duydz)

# @jit(nopython=True)
# def compute_sxx_split_x(mpml_z, mpml_y, pml_x, sxx_split_x, dt, mu, lame_lambda, duxdx, eta, chi, dduxdxdt):
#     return mpml_z * mpml_y * pml_x * (mpml_z * mpml_y * pml_x * sxx_split_x + dt * (2.0 * mu + lame_lambda) * duxdx + dt * (2.0 * eta + chi) * dduxdxdt)


def pstd_elastic_3d(kgrid: kWaveGrid,
        source: kSource,
        sensor: Union[NotATransducer, kSensor],
        medium: kWaveMedium,
        simulation_options: SimulationOptions):
    """
    pstd_elastic_3d 3D time-domain simulation of elastic wave propagation.

    DESCRIPTION:

        pstd_elastic_3d simulates the time-domain propagation of elastic waves
        through a three-dimensional homogeneous or heterogeneous medium given
        four input structures: kgrid, medium, source, and sensor. The
        computation is based on a pseudospectral time domain model which
        accounts for viscoelastic absorption and heterogeneous material
        parameters. At each time-step (defined by kgrid.dt and kgrid.Nt or
        kgrid.t_array), the wavefield parameters at the positions defined by
        sensor.mask are recorded and stored. If kgrid.t_array is set to
        'auto', this array is automatically generated using the makeTime
        method of the kWaveGrid class. An anisotropic absorbing boundary
        layer called a perfectly matched layer (PML) is implemented to
        prevent waves that leave one side of the domain being reintroduced
        from the opposite side (a consequence of using the FFT to compute the
        spatial derivatives in the wave equation). This allows infinite
        domain simulations to be computed using small computational grids.

        An initial pressure distribution can be specified by assigning a
        matrix of pressure values the same size as the computational grid to
        source.p0. This is then assigned to the normal components of the
        stress within the simulation function. A time varying stress source
        can similarly be specified by assigning a binary matrix (i.e., a
        matrix of 1's and 0's with the same dimensions as the computational
        grid) to source.s_mask where the 1's represent the grid points that
        form part of the source. The time varying input signals are then
        assigned to source.sxx, source.syy, source.szz, source.sxy,
        source.sxz, and source.syz. These can be a single time series (in
        which case it is applied to all source elements), or a matrix of time
        series following the source elements using MATLAB's standard
        column-wise linear matrix index ordering. A time varying velocity
        source can be specified in an analogous fashion, where the source
        location is specified by source.u_mask, and the time varying input
        velocity is assigned to source.ux, source.uy, and source.uz.

        The field values are returned as arrays of time series at the sensor
        locations defined by sensor.mask. This can be defined in three
        different ways. (1) As a binary matrix (i.e., a matrix of 1's and 0's
        with the same dimensions as the computational grid) representing the
        grid points within the computational grid that will collect the data.
        (2) As the grid coordinates of two opposing corners of a cuboid in
        the form [x1 y1 z1 x2 y2 z2]. This is equivalent to using a
        binary sensor mask covering the same region, however, the output is
        indexed differently as discussed below. (3) As a series of Cartesian
        coordinates within the grid which specify the location of the
        pressure values stored at each time step. If the Cartesian
        coordinates don't exactly match the coordinates of a grid point, the
        output values are calculated via interpolation. The Cartesian points
        must be given as a 3 by N matrix corresponding to the x, y, and z
        positions, respectively, where the Cartesian origin is assumed to be
        in the center of the grid. If no output is required, the sensor input
        can be replaced with `None`.

        If sensor.mask is given as a set of Cartesian coordinates, the
        computed sensor_data is returned in the same order. If sensor.mask is
        given as a binary matrix, sensor_data is returned using MATLAB's
        standard column-wise linear matrix index ordering. In both cases, the
        recorded data is indexed as sensor_data(sensor_point_index,
        time_index). For a binary sensor mask, the field values at a
        particular time can be restored to the sensor positions within the
        computation grid using unmaskSensorData. If sensor.mask is given as a
        list of cuboid corners, the recorded data is indexed as
        sensor_data(cuboid_index).p(x_index, y_index, z_index, time_index),
        where x_index, y_index, and z_index correspond to the grid index
        within the cuboid, and cuboid_index corresponds to the number of the
        cuboid if more than one is specified.

        By default, the recorded acoustic pressure field is passed directly
        to the output sensor_data. However, other acoustic parameters can
        also be recorded by setting sensor.record to a cell array of the form
        {'p', 'u', 'p_max', \}. For example, both the particle velocity and
        the acoustic pressure can be returned by setting sensor.record =
        {'p', 'u'}. If sensor.record is given, the output sensor_data is
        returned as a structure with the different outputs appended as
        structure fields. For example, if sensor.record = {'p', 'p_final',
        'p_max', 'u'}, the output would contain fields sensor_data.p,
        sensor_data.p_final, sensor_data.p_max, sensor_data.ux,
        sensor_data.uy, and sensor_data.uz. Most of the output parameters are
        recorded at the given sensor positions and are indexed as
        sensor_data.field(sensor_point_index, time_index) or
        sensor_data(cuboid_index).field(x_index, y_index, z_index,
        time_index) if using a sensor mask defined as cuboid corners. The
        exceptions are the averaged quantities ('p_max', 'p_rms', 'u_max',
        'p_rms', 'I_avg'), the 'all' quantities ('p_max_all', 'p_min_all',
        'u_max_all', 'u_min_all'), and the final quantities ('p_final',
        'u_final'). The averaged quantities are indexed as
        sensor_data.p_max(sensor_point_index) or
        sensor_data(cuboid_index).p_max(x_index, y_index, z_index) if using
        cuboid corners, while the final and 'all' quantities are returned
        over the entire grid and are always indexed as
        sensor_data.p_final(nx, ny, nz), regardless of the type of sensor
        mask.

        pstd_elastic_3d may also be used for time reversal image reconstruction
        by assigning the time varying pressure recorded over an arbitrary
        sensor surface to the input field sensor.time_reversal_boundary_data.
        This data is then enforced in time reversed order as a time varying
        Dirichlet boundary condition over the sensor surface given by
        sensor.mask. The boundary data must be indexed as
        sensor.time_reversal_boundary_data(sensor_point_index, time_index).
        If sensor.mask is given as a set of Cartesian coordinates, the
        boundary data must be given in the same order. An equivalent binary
        sensor mask (computed using nearest neighbour interpolation) is then
        used to place the pressure values into the computational grid at each
        time step. If sensor.mask is given as a binary matrix of sensor
        points, the boundary data must be ordered using MATLAB's standard
        column-wise linear matrix indexing. If no additional inputs are
        required, the source input can be replaced with an empty array [].

    USAGE:
        sensor_data = pstd_elastic_3d(kgrid, medium, source, sensor)
        sensor_data = pstd_elastic_3d(kgrid, medium, source, sensor, \)

    INPUTS:
    The minimum fields that must be assigned to run an initial value problem
    (for example, a photoacoustic forward simulation) are marked with a *.

        kgrid*                 - k-Wave grid object returned by kWaveGrid
                                containing Cartesian and k-space grid fields
        kgrid.t_array*         - evenly spaced array of time values [s] (set
                                to 'auto' by kWaveGrid)

        medium.sound_speed_compression*
                                - compressional sound speed distribution
                                within the acoustic medium [m/s]
        medium.sound_speed_shear*
                                - shear sound speed distribution within the
                                acoustic medium [m/s]
        medium.density*        - density distribution within the acoustic
                                medium [kg/m^3]
        medium.alpha_coeff_compression
                                - absorption coefficient for compressional
                                waves [dB/(MHz^2 cm)]
        medium.alpha_coeff_shear
                                - absorption coefficient for shear waves
                                [dB/(MHz^2 cm)]

        source.p0*             - initial pressure within the acoustic medium
        source.sxx             - time varying stress at each of the source
                                positions given by source.s_mask
        source.syy             - time varying stress at each of the source
                                positions given by source.s_mask
        source.szz             - time varying stress at each of the source
                                positions given by source.s_mask
        source.sxy             - time varying stress at each of the source
                                positions given by source.s_mask
        source.sxz             - time varying stress at each of the source
                                positions given by source.s_mask
        source.syz             - time varying stress at each of the source
                                positions given by source.s_mask
        source.s_mask          - binary matrix specifying the positions of
                                the time varying stress source distributions
        source.s_mode          - optional input to control whether the input
                                stress is injected as a mass source or
                                enforced as a dirichlet boundary condition
                                valid inputs are 'additive' (the default) or
                                'dirichlet'
        source.ux              - time varying particle velocity in the
                                x-direction at each of the source positions
                                given by source.u_mask
        source.uy              - time varying particle velocity in the
                                y-direction at each of the source positions
                                given by source.u_mask
        source.uz              - time varying particle velocity in the
                                z-direction at each of the source positions
                                given by source.u_mask
        source.u_mask          - binary matrix specifying the positions of
                                the time varying particle velocity
                                distribution
        source.u_mode          - optional input to control whether the input
                                velocity is applied as a force source or
                                enforced as a dirichlet boundary condition
                                valid inputs are 'additive' (the default) or
                                'dirichlet'

        sensor.mask*           - binary matrix or a set of Cartesian points
                                where the pressure is recorded at each
                                time-step
        sensor.record          - cell array of the acoustic parameters to
                                record in the form sensor.record = {'p',
                                'u', \} valid inputs are:

            'p' (acoustic pressure)
            'p_max' (maximum pressure)
            'p_min' (minimum pressure)
            'p_rms' (RMS pressure)
            'p_final' (final pressure field at all grid points)
            'p_max_all' (maximum pressure at all grid points)
            'p_min_all' (minimum pressure at all grid points)
            'u' (particle velocity)
            'u_max' (maximum particle velocity)
            'u_min' (minimum particle velocity)
            'u_rms' (RMS particle21st January 2014 velocity)
            'u_final' (final particle velocity field at all grid points)
            'u_max_all' (maximum particle velocity at all grid points)
            'u_min_all' (minimum particle velocity at all grid points)
            'u_non_staggered' (particle velocity on non-staggered grid)
            'u_split_field' (particle velocity on non-staggered grid split
                            into compressional and shear components)
            'I' (time varying acoustic intensity)
            'I_avg' (average acoustic intensity)

            NOTE: the acoustic pressure outputs are calculated from the
            normal stress via: p = -(sxx + syy)/2

        sensor.record_start_index
                                - time index at which the sensor should start
                                recording the data specified by
                                sensor.record (default = 1)
        sensor.time_reversal_boundary_data
                                - time varying pressure enforced as a
                                Dirichlet boundary condition over
                                sensor.mask

    Note: For a heterogeneous medium, medium.sound_speed_compression,
    medium.sound_speed_shear, and medium.density must be given in matrix form
    with the same dimensions as kgrid. For a homogeneous medium, these can be
    given as scalar values.

    OPTIONAL INPUTS:
        Optional 'string', value pairs that may be used to modify the default
        computational settings.

        See .html help file for details.

    OUTPUTS:
    If sensor.record is not defined by the user:
        sensor_data            - time varying pressure recorded at the sensor
                                positions given by sensor.mask

    If sensor.record is defined by the user:
        sensor_data.p          - time varying pressure recorded at the
                                sensor positions given by sensor.mask
                                (returned if 'p' is set)
        sensor_data.p_max      - maximum pressure recorded at the sensor
                                positions given by sensor.mask (returned if
                                'p_max' is set)
        sensor_data.p_min      - minimum pressure recorded at the sensor
                                positions given by sensor.mask (returned if
                                'p_min' is set)
        sensor_data.p_rms      - rms of the time varying pressure recorded
                                at the sensor positions given by
                                sensor.mask (returned if 'p_rms' is set)
        sensor_data.p_final    - final pressure field at all grid points
                                within the domain (returned if 'p_final' is
                                set)
        sensor_data.p_max_all  - maximum pressure recorded at all grid points
                                within the domain (returned if 'p_max_all'
                                is set)
        sensor_data.p_min_all  - minimum pressure recorded at all grid points
                                within the domain (returned if 'p_min_all'
                                is set)
        sensor_data.ux         - time varying particle velocity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'u' is
                                set)
        sensor_data.uy         - time varying particle velocity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'u' is
                                set)
        sensor_data.uz         - time varying particle velocity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'u' is
                                set)
        sensor_data.ux_max     - maximum particle velocity in the x-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)
        sensor_data.uy_max     - maximum particle velocity in the y-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)
        sensor_data.uz_max     - maximum particle velocity in the z-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)
        sensor_data.ux_min     - minimum particle velocity in the x-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)
        sensor_data.uy_min     - minimum particle velocity in the y-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)
        sensor_data.uz_min     - minimum particle velocity in the z-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)
        sensor_data.ux_rms     - rms of the time varying particle velocity in
                                the x-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)
        sensor_data.uy_rms     - rms of the time varying particle velocity in
                                the y-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)
        sensor_data.uz_rms     - rms of the time varying particle velocity
                                in the z-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)
        sensor_data.ux_final   - final particle velocity field in the
                                x-direction at all grid points within the
                                domain (returned if 'u_final' is set)
        sensor_data.uy_final   - final particle velocity field in the
                                y-direction at all grid points within the
                                domain (returned if 'u_final' is set)
        sensor_data.uz_final   - final particle velocity field in the
                                z-direction at all grid points within the
                                domain (returned if 'u_final' is set)
        sensor_data.ux_max_all - maximum particle velocity in the x-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.uy_max_all - maximum particle velocity in the y-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.uz_max_all - maximum particle velocity in the z-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.ux_min_all - minimum particle velocity in the x-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)
        sensor_data.uy_min_all - minimum particle velocity in the y-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)
        sensor_data.uz_min_all - minimum particle velocity in the z-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)
        sensor_data.ux_non_staggered
                                - time varying particle velocity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.uy_non_staggered
                                - time varying particle velocity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.uz_non_staggered
                                - time varying particle velocity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.ux_split_p - compressional component of the time varying
                                particle velocity in the x-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.ux_split_s - shear component of the time varying particle
                                velocity in the x-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uy_split_p - compressional component of the time varying
                                particle velocity in the y-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uy_split_s - shear component of the time varying particle
                                velocity in the y-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uz_split_p - compressional component of the time varying
                                particle velocity in the z-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.uz_split_s - shear component of the time varying particle
                                velocity in the z-direction on the
                                non-staggered grid recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_split_field' is set)
        sensor_data.Ix         - time varying acoustic intensity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Iy         - time varying acoustic intensity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Iz         - time varying acoustic intensity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Ix_avg     - average acoustic intensity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)
        sensor_data.Iy_avg     - average acoustic intensity in the
                                y-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)
        sensor_data.Iz_avg     - average acoustic intensity in the
                                z-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)

    ABOUT:
        author                 - Bradley Treeby & Ben Cox
        date                   - 11th March 2013
        last update            - 13th January 2019

    This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    Copyright (C) 2013-2019 Bradley Treeby and Ben Cox

    See also kspaceFirstOrder3D, kWaveGrid, pstdElastic2D

    This file is part of k-Wave. k-Wave is free software: you can
    redistribute it and/or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
    more details.

    You should have received a copy of the GNU Lesser General Public License
    along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
    """

# =========================================================================
# CHECK INPUT STRUCTURES AND OPTIONAL INPUTS
# =========================================================================

    # start the timer and store the start time
    timer = TicToc()
    timer.tic()

    k_sim = kWaveSimulation(
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        medium=medium,
        simulation_options=simulation_options
    )

    # run helper script to check inputs
    k_sim.input_checking('pstd_elastic_3d')

    sensor_data = k_sim.sensor_data

    options = k_sim.options

    rho0 = np.atleast_1d(k_sim.rho0) # maybe at least 3d?

    m_rho0 : int = np.squeeze(rho0).ndim

    # assign the lame parameters
    mu = medium.sound_speed_shear**2 * medium.density
    lame_lambda = medium.sound_speed_compression**2 * medium.density - 2.0 * mu
    m_mu : int = np.squeeze(mu).ndim

    # assign the viscosity coefficients
    if (options.kelvin_voigt_model):
        eta = 2.0 * rho0 * medium.sound_speed_shear**3 * db2neper(medium.alpha_coeff_shear, 2)
        chi = 2.0 * rho0 * medium.sound_speed_compression**3 * db2neper(medium.alpha_coeff_compression, 2) - 2.0 * eta
        m_eta : int = np.squeeze(eta).ndim


    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================

    # calculate the values of the density at the staggered grid points
    # using the arithmetic average [1, 2], where sgx  = (x + dx/2, y),
    # sgy  = (x, y + dy/2) and sgz = (x, y, z + dz/2)
    if (m_rho0 == 3 and (options.use_sg)):
        # rho0 is heterogeneous and staggered grids are used

        points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec), np.squeeze(k_sim.kgrid.z_vec))

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
                         np.squeeze(k_sim.kgrid.y_vec),
                         np.squeeze(k_sim.kgrid.z_vec), indexing='ij',)
        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgx_a = interpn(points, k_sim.rho0, interp_points,  method='linear', bounds_error=False)

        # mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
        #                  np.squeeze(k_sim.kgrid.y_vec),
        #                  np.squeeze(k_sim.kgrid.z_vec), indexing='xy',)
        # interp_points = np.moveaxis(mg, 0, -1)
        # rho0_sgx_b = interpn(points, k_sim.rho0, interp_points,  method='linear', bounds_error=False)

        # interp_mesh = np.array(np.meshgrid(np.squeeze(k_sim.kgrid.x_vec)+ k_sim.kgrid.dy / 2,
        #                                    np.squeeze(k_sim.kgrid.y_vec),
        #                                    np.squeeze(k_sim.kgrid.z_vec), indexing='ij'))
        # interp_points = np.moveaxis(interp_mesh, 0, -1).reshape(k_sim.kgrid.Nx * k_sim.kgrid.Ny * k_sim.kgrid.Nz, 3)
        # rho0_sgx_c = interpn(points, k_sim.rho0, interp_points,  method='linear', bounds_error=False)
        # rho0_sgx_c = np.reshape(rho0_sgx_c, (k_sim.kgrid.Nx, k_sim.kgrid.Ny, k_sim.kgrid.Nz))

        # interp_mesh = np.array(np.meshgrid(np.squeeze(k_sim.kgrid.x_vec)+ k_sim.kgrid.dy / 2,
        #                                    np.squeeze(k_sim.kgrid.y_vec),
        #                                    np.squeeze(k_sim.kgrid.z_vec), indexing='xy'))
        # interp_points = np.moveaxis(interp_mesh, 0, -1).reshape(k_sim.kgrid.Nx * k_sim.kgrid.Ny * k_sim.kgrid.Nz, 3)
        # rho0_sgx_d = interpn(points, k_sim.rho0, interp_points,  method='linear', bounds_error=False)
        # rho0_sgx_d = np.reshape(rho0_sgx_d, (k_sim.kgrid.Nx, k_sim.kgrid.Ny, k_sim.kgrid.Nz))

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2,
                         np.squeeze(k_sim.kgrid.z_vec), indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgy = interpn(points, k_sim.rho0, interp_points, method='linear', bounds_error=False)
        #rho0_sgy = np.transpose(rho0_sgy)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec),
                         np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2, indexing='ij')
        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgz = interpn(points, k_sim.rho0, interp_points, method='linear', bounds_error=False)
        #rho0_sgz = np.transpose(rho0_sgz)

        # set values outside of the interpolation range to original values
        rho0_sgx_a[np.isnan(rho0_sgx_a)] = rho0[np.isnan(rho0_sgx_a)]
        # rho0_sgx_a[np.isnan(rho0_sgx_a)] = rho0[np.isnan(rho0_sgx_a)]
        # rho0_sgx_a[np.isnan(rho0_sgx_a)] = rho0[np.isnan(rho0_sgx_a)]
        # rho0_sgx_a[np.isnan(rho0_sgx_a)] = rho0[np.isnan(rho0_sgx_a)]
        rho0_sgy[np.isnan(rho0_sgy)] = rho0[np.isnan(rho0_sgy)]
        rho0_sgz[np.isnan(rho0_sgz)] = rho0[np.isnan(rho0_sgz)]
    else:
        # rho0 is homogeneous or staggered grids are not used
        rho0_sgx = rho0
        rho0_sgy = rho0
        rho0_sgz = rho0

    rho0_sgx = rho0_sgx_a

    # elementwise reciprocal of rho0 so it doesn't have to be done each time step
    rho0_sgx_inv = 1.0 / rho0_sgx
    rho0_sgy_inv = 1.0 / rho0_sgy
    rho0_sgz_inv = 1.0 / rho0_sgz

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        # del rho0_sgx_a
        # del rho0_sgx_b
        # del rho0_sgx_c
        # del rho0_sgx_d
        del rho0_sgx
        del rho0_sgy
        del rho0_sgz

    # calculate the values of mu at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2, z), etc
    if (m_mu == 3 and options.use_sg):
        # interpolation points
        points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec), np.squeeze(k_sim.kgrid.z_vec))

        # mu is heterogeneous and staggered grids are used
        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2,
                         np.squeeze(k_sim.kgrid.z_vec), indexing='ij', )
        interp_points = np.moveaxis(mg, 0, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgxy = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)
        #mu_sgxy = np.transpose(mu_sgxy)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
                         np.squeeze(k_sim.kgrid.y_vec),
                         np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2, indexing='ij',)
        interp_points = np.moveaxis(mg, 0, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgxz = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)
        #mu_sgxz = np.transpose(mu_sgxz)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2,
                         np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2, indexing='ij',)
        interp_points = np.moveaxis(mg, 0, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgyz = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)
        #mu_sgyz = np.transpose(mu_sgyz)

        # set values outside of the interpolation range to original values
        mu_sgxy[np.isnan(mu_sgxy)] = mu[np.isnan(mu_sgxy)]
        mu_sgxz[np.isnan(mu_sgxz)] = mu[np.isnan(mu_sgxz)]
        mu_sgyz[np.isnan(mu_sgyz)] = mu[np.isnan(mu_sgyz)]

    else:
        # mu is homogeneous or staggered grids are not used
        mu_sgxy  = mu
        mu_sgxz  = mu
        mu_sgyz  = mu


    # calculate the values of eta at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2, z) etc
    if options.kelvin_voigt_model:
        if m_eta == 3 and options.use_sg:

            points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec), np.squeeze(k_sim.kgrid.z_vec))

            # eta is heterogeneous and staggered grids are used
            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                             np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                             np.squeeze(k_sim.kgrid.z_vec), indexing='ij',)
            interp_points = np.moveaxis(mg, 0, -1)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgxy = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2.0,
                             np.squeeze(k_sim.kgrid.y_vec),
                             np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0, indexing='ij',)
            interp_points = np.moveaxis(mg, 0, -1)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgxz = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                             np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2.0,
                             np.squeeze(k_sim.kgrid.z_vec) + k_sim.kgrid.dz / 2.0, indexing='ij',)
            interp_points = np.moveaxis(mg, 0, -1)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgyz = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            # set values outside of the interpolation range to original values
            eta_sgxy[np.isnan(eta_sgxy)] = eta[np.isnan(eta_sgxy)]
            eta_sgxz[np.isnan(eta_sgxz)] = eta[np.isnan(eta_sgxz)]
            eta_sgyz[np.isnan(eta_sgyz)] = eta[np.isnan(eta_sgyz)]

        else:

            # eta is homogeneous or staggered grids are not used
            eta_sgxy = eta
            eta_sgxz = eta
            eta_sgyz = eta



    # [1] Moczo, P., Kristek, J., Vavry?uk, V., Archuleta, R. J., & Halada, L.
    # (2002). 3D heterogeneous staggered-grid finite-difference modeling of
    # seismic motion with volume harmonic and arithmetic averaging of elastic
    # moduli and densities. Bulletin of the Seismological Society of America,
    # 92(8), 3042-3066.

    # [2] Toyoda, M., Takahashi, D., & Kawai, Y. (2012). Averaged material
    # parameters and boundary conditions for the vibroacoustic
    # finite-difference time-domain method with a nonuniform mesh. Acoustical
    # Science and Technology, 33(4), 273-276.

    # =========================================================================
    # RECORDER
    # =========================================================================

    record = k_sim.record


    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the regular PML operators based on the reference sound speed and PML settings
    Nx, Ny, Nz = k_sim.kgrid.Nx, k_sim.kgrid.Ny, k_sim.kgrid.Nz
    dx, dy, dz = k_sim.kgrid.dx, k_sim.kgrid.dy, k_sim.kgrid.dz
    dt = k_sim.kgrid.dt

    pml_x_alpha, pml_y_alpha, pml_z_alpha  = options.pml_x_alpha, options.pml_y_alpha, options.pml_z_alpha
    pml_x_size, pml_y_size, pml_z_size = options.pml_x_size, options.pml_y_size, options.pml_z_size

    multi_axial_PML_ratio = options.multi_axial_PML_ratio

    c_ref = k_sim.c_ref

    # get the regular PML operators based on the reference sound speed and PML settings
    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0)
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, (True and options.use_sg), 0)
    pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 1)
    pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, (True and options.use_sg), 1)
    pml_z     = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, False, 2)
    pml_z_sgz = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, (True and options.use_sg), 2)

    # get the multi-axial PML operators
    mpml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, False, 0)
    mpml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, (True and options.use_sg), 0)
    mpml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, False, 1)
    mpml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, (True and options.use_sg), 1)
    mpml_z     = get_pml(Nz, dz, dt, c_ref, pml_z_size, multi_axial_PML_ratio * pml_z_alpha, False, 2)
    mpml_z_sgz = get_pml(Nz, dz, dt, c_ref, pml_z_size, multi_axial_PML_ratio * pml_z_alpha, (True and options.use_sg), 2)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using  np.fft.ifftshift (the option
    # options.use_sg exists for debugging)
    if options.use_sg:

        kx_vec = np.squeeze(k_sim.kgrid.k_vec[0])
        ky_vec = np.squeeze(k_sim.kgrid.k_vec[1])
        kz_vec = np.squeeze(k_sim.kgrid.k_vec[2])

        ddx_k_shift_pos =  np.fft.ifftshift(1j * kx_vec * np.exp(1j * kx_vec * kgrid.dx / 2.0))
        ddx_k_shift_neg =  np.fft.ifftshift(1j * kx_vec * np.exp(-1j * kx_vec * kgrid.dx / 2.0))
        ddy_k_shift_pos =  np.fft.ifftshift(1j * ky_vec * np.exp(1j * ky_vec * kgrid.dy / 2.0))
        ddy_k_shift_neg =  np.fft.ifftshift(1j * ky_vec * np.exp(-1j * ky_vec * kgrid.dy / 2.0))
        ddz_k_shift_pos =  np.fft.ifftshift(1j * kz_vec * np.exp(1j * kz_vec * kgrid.dz / 2.0))
        ddz_k_shift_neg =  np.fft.ifftshift(1j * kz_vec * np.exp(-1j * kz_vec * kgrid.dz / 2.0))
    else:
        ddx_k_shift_pos =  np.fft.ifftshift(1j * kx_vec)
        ddx_k_shift_neg =  np.fft.ifftshift(1j * kx_vec)
        ddy_k_shift_pos =  np.fft.ifftshift(1j * ky_vec)
        ddy_k_shift_neg =  np.fft.ifftshift(1j * ky_vec)
        ddz_k_shift_pos =  np.fft.ifftshift(1j * kz_vec)
        ddz_k_shift_neg =  np.fft.ifftshift(1j * kz_vec)

    # force the derivative and shift operators to be in the correct direction
    # for use with broadcasting
    ddx_k_shift_pos = np.expand_dims(np.expand_dims(ddx_k_shift_pos, axis=-1), axis=-1)
    ddx_k_shift_neg = np.expand_dims(np.expand_dims(ddx_k_shift_neg, axis=-1), axis=-1)
    ddy_k_shift_pos = np.expand_dims(np.expand_dims(np.squeeze(ddy_k_shift_pos), axis=0), axis=-1)
    ddy_k_shift_neg = np.expand_dims(np.expand_dims(np.squeeze(ddy_k_shift_neg), axis=0), axis=-1)
    ddz_k_shift_pos = np.expand_dims(np.expand_dims(np.squeeze(ddz_k_shift_pos), axis=0), axis=0)
    ddz_k_shift_neg = np.expand_dims(np.expand_dims(np.squeeze(ddz_k_shift_neg), axis=0), axis=0)

    ddx_k_shift_pos = np.reshape(ddx_k_shift_pos, ddx_k_shift_pos.shape, order='F')
    ddx_k_shift_neg = np.reshape(ddx_k_shift_neg, ddx_k_shift_neg.shape, order='F')
    ddy_k_shift_pos = np.reshape(ddy_k_shift_pos, ddy_k_shift_pos.shape, order='F')
    ddy_k_shift_neg = np.reshape(ddy_k_shift_neg, ddy_k_shift_neg.shape, order='F')
    ddz_k_shift_pos = np.reshape(ddz_k_shift_pos, ddz_k_shift_pos.shape, order='F')
    ddz_k_shift_neg = np.reshape(ddz_k_shift_neg, ddz_k_shift_neg.shape, order='F')
    # ddz_k_shift_neg = np.transpose(ddz_k_shift_neg, axes=[1, 2, 0])

    pml_x = np.transpose(pml_x)
    pml_x_sgx = np.transpose(pml_x_sgx)
    mpml_x = np.transpose(mpml_x)
    mpml_x_sgx = np.transpose(mpml_x_sgx)

    pml_x = np.expand_dims(pml_x, axis=-1)
    pml_x_sgx = np.expand_dims(pml_x_sgx, axis=-1)
    mpml_x = np.expand_dims(mpml_x, axis=-1)
    mpml_x_sgx = np.expand_dims(mpml_x_sgx, axis=-1)

    pml_y = np.expand_dims(pml_y, axis=0)
    pml_y_sgy = np.expand_dims(pml_y_sgy, axis=0)
    mpml_y = np.expand_dims(mpml_y, axis=0)
    mpml_y_sgy = np.expand_dims(mpml_y_sgy, axis=0)

    pml_z = np.expand_dims(pml_z, axis=0)
    pml_z_sgz = np.expand_dims(pml_z_sgz, axis=0)
    mpml_z = np.expand_dims(mpml_z, axis=0)
    mpml_z_sgz = np.expand_dims(mpml_z_sgz, axis=0)

    # print('pml_x', np.shape(pml_x), np.shape(pml_x_sgx), np.shape(mpml_x), np.shape(mpml_x_sgx) )
    # print('pml_y', np.shape(pml_y), np.shape(pml_y_sgy), np.shape(mpml_y), np.shape(mpml_y_sgy) )
    # print('pml_z', np.shape(pml_z), np.shape(pml_z_sgz), np.shape(mpml_z), np.shape(mpml_z_sgz) )

    # print("ddx_k_shift", np.shape(ddx_k_shift_pos), np.shape(ddx_k_shift_neg))
    # print("ddy_k_shift", np.shape(ddy_k_shift_pos), np.shape(ddy_k_shift_neg))
    # print("ddz_k_shift", np.shape(ddz_k_shift_pos), np.shape(ddz_k_shift_neg))




    # =========================================================================
    # DATA CASTING
    # =========================================================================

    # run subscript to cast the remaining loop variables to the data type
    # specified by data_cast
    if not (options.data_cast == 'off'):
        myType = np.single
    else:
        myType = np.double

    grid_shape = (Nx, Ny, Nz)
    myOrder = 'F'

    # preallocate the loop variables using the castZeros anonymous function
    # (this creates a matrix of zeros in the data type specified by data_cast)
    ux_split_x  = np.zeros(grid_shape, myType, order=myOrder)
    ux_split_y  = np.zeros(grid_shape, myType, order=myOrder)
    ux_split_z  = np.zeros(grid_shape, myType, order=myOrder)
    ux_sgx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    uy_split_x  = np.zeros(grid_shape, myType, order=myOrder)
    uy_split_y  = np.zeros(grid_shape, myType, order=myOrder)
    uy_split_z  = np.zeros(grid_shape, myType, order=myOrder)
    uy_sgy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    uz_split_x  = np.zeros(grid_shape, myType, order=myOrder)
    uz_split_y  = np.zeros(grid_shape, myType, order=myOrder)
    uz_split_z  = np.zeros(grid_shape, myType, order=myOrder)
    uz_sgz      = np.zeros(grid_shape, myType, order=myOrder)  # **

    sxx_split_x = np.zeros(grid_shape, myType, order=myOrder)
    sxx_split_y = np.zeros(grid_shape, myType, order=myOrder)
    sxx_split_z = np.zeros(grid_shape, myType, order=myOrder)
    syy_split_x = np.zeros(grid_shape, myType, order=myOrder)
    syy_split_y = np.zeros(grid_shape, myType, order=myOrder)
    syy_split_z = np.zeros(grid_shape, myType, order=myOrder)
    szz_split_x = np.zeros(grid_shape, myType, order=myOrder)
    szz_split_y = np.zeros(grid_shape, myType, order=myOrder)
    szz_split_z = np.zeros(grid_shape, myType, order=myOrder)
    sxy_split_x = np.zeros(grid_shape, myType, order=myOrder)
    sxy_split_y = np.zeros(grid_shape, myType, order=myOrder)
    sxz_split_x = np.zeros(grid_shape, myType, order=myOrder)
    sxz_split_z = np.zeros(grid_shape, myType, order=myOrder)
    syz_split_y = np.zeros(grid_shape, myType, order=myOrder)
    syz_split_z = np.zeros(grid_shape, myType, order=myOrder)

    duxdx       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duxdy       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duxdz       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duydx       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duydy       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duydz       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duzdx       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duzdy       = np.zeros(grid_shape, myType, order=myOrder)  # **
    duzdz       = np.zeros(grid_shape, myType, order=myOrder)  # **

    dsxxdx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsyydy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dszzdz      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxydx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxydy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxzdx      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsxzdz      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsyzdy      = np.zeros(grid_shape, myType, order=myOrder)  # **
    dsyzdz      = np.zeros(grid_shape, myType, order=myOrder)  # **

    p           = np.zeros(grid_shape, myType, order=myOrder)  # **

    if options.kelvin_voigt_model:
        dduxdxdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduxdydt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduxdzdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduydxdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduydydt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduydzdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduzdxdt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduzdydt       = np.zeros(grid_shape, myType, order=myOrder)  # **
        dduzdzdt       = np.zeros(grid_shape, myType, order=myOrder)  # **

    # to save memory, the variables noted with a ** do not neccesarily need to
    # be np.explicitly stored (they are not needed for update steps). Instead they
    # could be replaced with a small number of temporary variables that are
    # reused several times during the time loop.



    checking: bool = True
    verbose: bool = False
    mat_contents = sio.loadmat('data/3DtwoStep.mat')
    load_index: int = 1
    error_number: int = 0

    tol: float = 10E-12
    if verbose:
        print(sorted(mat_contents.keys()))

    if error_number > 0:
        line = ''
    else:
        line = "\n"

    mat_dsxxdx = mat_contents['dsxxdx']
    mat_dsyydy = mat_contents['dsyydy']
    mat_dszzdz = mat_contents['dszzdz']

    mat_dsxydx = mat_contents['dsxydx']
    mat_dsxydy = mat_contents['dsxydy']

    mat_dsxzdx = mat_contents['dsxzdx']
    mat_dsxzdz = mat_contents['dsxzdz']

    mat_dsyzdy = mat_contents['dsyzdy']
    mat_dsyzdz = mat_contents['dsyzdz']

    mat_dduxdxdt = mat_contents['dduxdxdt']
    mat_dduxdydt = mat_contents['dduxdydt']
    mat_dduxdzdt = mat_contents['dduxdzdt']
    mat_dduydxdt = mat_contents['dduydxdt']
    mat_dduydydt = mat_contents['dduydydt']
    mat_dduydzdt = mat_contents['dduydzdt']
    mat_dduzdxdt = mat_contents['dduzdxdt']
    mat_dduzdydt = mat_contents['dduzdydt']
    mat_dduzdzdt = mat_contents['dduzdzdt']

    mat_duxdx = mat_contents['duxdx']
    mat_duxdy = mat_contents['duxdy']
    mat_duxdz = mat_contents['duxdz']
    mat_duydx = mat_contents['duydx']
    mat_duydy = mat_contents['duydy']
    mat_duydz = mat_contents['duydz']
    mat_duzdx = mat_contents['duzdx']
    mat_duzdy = mat_contents['duzdy']
    mat_duzdz = mat_contents['duzdz']

    mat_ux_sgx = mat_contents['ux_sgx']
    mat_ux_split_x = mat_contents['ux_split_x']
    mat_ux_split_y = mat_contents['ux_split_y']
    mat_ux_split_z = mat_contents['ux_split_z']
    mat_uy_sgy = mat_contents['uy_sgy']
    mat_uy_split_x = mat_contents['uy_split_x']
    mat_uy_split_y = mat_contents['uy_split_y']
    mat_uy_split_z = mat_contents['uy_split_z']
    mat_uz_sgz = mat_contents['uz_sgz']
    mat_uz_split_x = mat_contents['uz_split_x']
    mat_uz_split_y = mat_contents['uz_split_y']
    mat_uz_split_z = mat_contents['uz_split_z']

    mat_sxx_split_x = mat_contents['sxx_split_x']
    mat_sxx_split_y = mat_contents['sxx_split_y']
    mat_sxx_split_z = mat_contents['sxx_split_z']
    mat_syy_split_x = mat_contents['syy_split_x']
    mat_syy_split_y = mat_contents['syy_split_y']
    mat_syy_split_z = mat_contents['syy_split_z']
    mat_szz_split_x = mat_contents['szz_split_x']
    mat_szz_split_y = mat_contents['szz_split_y']
    mat_szz_split_z = mat_contents['szz_split_z']
    mat_sxy_split_x = mat_contents['sxy_split_x']
    mat_sxy_split_y = mat_contents['sxy_split_y']
    mat_sxz_split_x = mat_contents['sxz_split_x']
    mat_sxz_split_z = mat_contents['sxz_split_z']
    mat_syz_split_y = mat_contents['syz_split_y']
    mat_syz_split_z = mat_contents['syz_split_z']

    mat_p = mat_contents['p']

    mat_mu_sgxy = mat_contents['mu_sgxy']
    mat_mu_sgxz = mat_contents['mu_sgxz']
    mat_mu_sgyz = mat_contents['mu_sgyz']
    mat_eta_sgxy = mat_contents['eta_sgxy']
    mat_eta_sgxz = mat_contents['eta_sgxz']
    mat_eta_sgyz = mat_contents['eta_sgyz']
    mat_rho0_sgx_inv = mat_contents['rho0_sgx_inv']
    mat_rho0_sgy_inv = mat_contents['rho0_sgy_inv']
    mat_rho0_sgz_inv = mat_contents['rho0_sgz_inv']

    mat_sensor_data = mat_contents['sensor_data']

    mat_source_ux = mat_contents['sux']

    mat_ddx_k_shift_pos = mat_contents['ddx_k_shift_pos']
    mat_ddy_k_shift_pos = mat_contents['ddy_k_shift_pos']
    mat_ddz_k_shift_pos = mat_contents['ddz_k_shift_pos']
    mat_ddx_k_shift_neg = mat_contents['ddx_k_shift_neg']
    mat_ddy_k_shift_neg = mat_contents['ddy_k_shift_neg']
    mat_ddz_k_shift_neg = mat_contents['ddz_k_shift_neg']

    mat_a_x = mat_contents['a_x']
    mat_b_x = mat_contents['b_x']
    mat_c_x = mat_contents['c_x']

    mat_a_y = mat_contents['a_y']
    mat_b_y = mat_contents['b_y']
    mat_c_y = mat_contents['c_y']

    # =========================================================================
    # CREATE INDEX VARIABLES
    # =========================================================================

    # # setup the time index variable
    # if not options.time_rev:
    #     index_start: int = 0
    #     index_step: int = 1
    #     index_end: int = k_sim.kgrid.Nt
    # else:
    #     # throw error for unsupported feature
    #     raise TypeError('Time reversal using sensor.time_reversal_boundary_data is not currently supported.')

    if checking:
        mu_tol = 5e-3
        if (np.abs(mat_mu_sgxy - mu_sgxy).sum() > mu_tol):
            error_number =+ 1
            print(line + "mu_sgxy is not correct!", np.abs(mat_mu_sgxy - mu_sgxy).sum())
            print(mu_sgxy.shape, mat_mu_sgxy.shape)
            print("mat:", mat_mu_sgxy[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", mu_sgxy[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_mu_sgxy - mu_sgxy)),
                  "\tmat:", mat_mu_sgxy[np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mat_mu_sgxy.shape, order='F')],
                  "\tpy:",      mu_sgxy[np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mu_sgxy.shape, order='F')],
                   "\t",  mat_mu_sgxy[33,33,0], mu_sgxy[33,33,0])
            print("max diff:", np.max(np.abs(mat_mu_sgxy - mu_sgxy)),
                  "\tmat:", mat_mu_sgxy[np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mat_mu_sgxy.shape, order='C')],
                  "\tpy:",      mu_sgxy[np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mu_sgxy.shape, order='C')],
                   "\t",  mat_mu_sgxy[0, 33, 33], mu_sgxy[0, 33, 33])
            print("arg max:", np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mu_sgxy.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mat_mu_sgxy.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mu_sgxy.shape, order='C'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxy - mu_sgxy)), mat_mu_sgxy.shape, order='C'))
        else:
            pass
        if (np.abs(mat_mu_sgxz - mu_sgxz).sum() > mu_tol):
            error_number =+ 1
            print(line + "mu_sgxz is not correct!")
            print(mu_sgxz.shape, mat_mu_sgxz.shape)
            print("mat:", mat_mu_sgxz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", mu_sgxz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_mu_sgxz - mu_sgxz)),
                  "\tmat:", mat_mu_sgxz[np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mat_mu_sgxz.shape, order='F')],
                  "\tpy:",      mu_sgxz[np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mu_sgxz.shape, order='F')],
                   "\t",  mat_mu_sgxz[33,33,0], mu_sgxz[33,33,0])
            print("max diff:", np.max(np.abs(mat_mu_sgxz - mu_sgxz)),
                  "\tmat:", mat_mu_sgxz[np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mat_mu_sgxz.shape, order='C')],
                  "\tpy:",      mu_sgxz[np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mu_sgxz.shape, order='C')],
                   "\t",  mat_mu_sgxz[0, 33, 33], mu_sgxz[0, 33, 33])
            print("arg max:", np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mu_sgxz.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mat_mu_sgxz.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mu_sgxz.shape, order='C'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgxz - mu_sgxz)), mat_mu_sgxz.shape, order='C'))
        else:
            pass
        if (np.abs(mat_mu_sgyz - mu_sgyz).sum() > mu_tol):
            error_number =+ 1
            print(line + "mu_sgyz is not correct!")
            print(mu_sgyz.shape, mat_mu_sgyz.shape)
            print("mat:", mat_mu_sgyz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", mu_sgyz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_mu_sgyz - mu_sgyz)),
                  "\tmat:", mat_mu_sgyz[np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mat_mu_sgyz.shape, order='F')],
                  "\tpy:",      mu_sgyz[np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mu_sgyz.shape, order='F')],
                   "\t",  mat_mu_sgyz[33,33,0], mu_sgyz[33,33,0])
            print("max diff:", np.max(np.abs(mat_mu_sgyz - mu_sgyz)),
                  "\tmat:", mat_mu_sgyz[np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mat_mu_sgyz.shape, order='C')],
                  "\tpy:",      mu_sgyz[np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mu_sgyz.shape, order='C')],
                   "\t",  mat_mu_sgyz[0, 33, 33], mu_sgyz[0, 33, 33])
            print("arg max:", np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mu_sgyz.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mat_mu_sgyz.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mu_sgyz.shape, order='C'),
                  np.unravel_index(np.argmax(np.abs(mat_mu_sgyz - mu_sgyz)), mat_mu_sgyz.shape, order='C'))
        else:
            pass


        if (np.abs(mat_eta_sgxy - eta_sgxy).sum() > tol):
            error_number =+ 1
            print(line + "eta_sgxy is not correct!")
            print("mat:", mat_eta_sgxy[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", eta_sgxy[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_eta_sgxy - eta_sgxy)))
            print("arg max:", np.argmax(np.abs(mat_eta_sgxy - eta_sgxy)),
                  np.unravel_index(np.argmax(np.abs(mat_eta_sgxy - eta_sgxy)), eta_sgxy.shape, order='F'))
        else:
            pass

        if (np.abs(mat_eta_sgxz - eta_sgxz).sum() > tol):
            error_number =+ 1
            print(line + "eta_sgxz is not correct!")
            print("mat:", mat_eta_sgxz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", eta_sgxz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_eta_sgxz - eta_sgxz)))
            print("arg max:", np.argmax(np.abs(mat_eta_sgxz - eta_sgxz)),
                  np.unravel_index(np.argmax(np.abs(mat_eta_sgxz - eta_sgxz)), eta_sgxz.shape, order='F'))
        else:
            pass
        if (np.abs(mat_eta_sgyz - eta_sgyz).sum() > tol):
            error_number =+ 1
            print(line + "eta_sgyz is not correct!")
            print("mat:", mat_eta_sgyz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", eta_sgyz[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_eta_sgyz - eta_sgyz)))
            print("arg max:", np.argmax(np.abs(mat_eta_sgyz - eta_sgyz)),
                  np.unravel_index(np.argmax(np.abs(mat_eta_sgyz - eta_sgyz)), eta_sgxy.shape, order='F'))
        else:
            pass
        if (np.abs(mat_rho0_sgx_inv - rho0_sgx_inv).sum() > tol):
            error_number =+ 1
            print(line + "rho0_sgx_inv is not correct!")
            print("mat:", mat_rho0_sgx_inv[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", rho0_sgx_inv[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_rho0_sgx_inv - rho0_sgx_inv)))
            print("arg max:", np.argmax(np.abs(mat_rho0_sgx_inv - rho0_sgx_inv)),
                  np.unravel_index(np.argmax(np.abs(mat_rho0_sgx_inv - rho0_sgx_inv)), rho0_sgx_inv.shape, order='F'))
        else:
            pass
            # print("rho0_sgx_inv IS CORRECT")
        if (np.abs(mat_rho0_sgy_inv - rho0_sgy_inv).sum() > tol):
            error_number =+ 1
            print(line + "rho0_sgy_inv is not correct!")
            print("mat:", mat_rho0_sgy_inv[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", rho0_sgy_inv[Nx // 2 - 3: Nx //2 +3 , 0, 0])
            print("max diff:", np.max(np.abs(mat_rho0_sgy_inv - rho0_sgy_inv)))
            print("arg max:", np.argmax(np.abs(mat_rho0_sgy_inv - rho0_sgy_inv)),
                  np.unravel_index(np.argmax(np.abs(mat_rho0_sgy_inv - rho0_sgy_inv)), rho0_sgy_inv.shape, order='F'))
        else:
            pass
        if (np.abs(mat_rho0_sgz_inv - rho0_sgz_inv).sum() > tol):
            error_number =+ 1
            print(line + "rho0_sgz_inv is not correct!")
            print("mat:", mat_rho0_sgz_inv[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("py:", rho0_sgz_inv[Nx // 2 - 3: Nx //2 + 3, 0, 0])
            print("max diff:", np.max(np.abs(mat_rho0_sgz_inv - rho0_sgz_inv)))
            print("arg max:", np.argmax(np.abs(mat_rho0_sgz_inv - rho0_sgz_inv)),
                  np.unravel_index(np.argmax(np.abs(mat_rho0_sgz_inv - rho0_sgz_inv)), rho0_sgz_inv.shape, order='F'))
        else:
            pass


    # print("shape ddx_k_shift_pos:", np.shape(mat_ddx_k_shift_pos), np.shape(ddx_k_shift_pos))
    # print("shape ddy_k_shift_pos:", np.shape(mat_ddy_k_shift_pos), np.shape(ddy_k_shift_pos))
    # print("shape ddz_k_shift_pos:", np.shape(mat_ddz_k_shift_pos), np.shape(ddz_k_shift_pos))
    # print("shape ddx_k_shift_neg:", np.shape(mat_ddx_k_shift_neg), np.shape(ddx_k_shift_neg))
    # print("shape ddy_k_shift_neg:", np.shape(mat_ddy_k_shift_neg), np.shape(ddy_k_shift_neg))
    # print("shape ddz_k_shift_neg:", np.shape(mat_ddz_k_shift_neg), np.shape(ddz_k_shift_neg))

    if checking:
        if (np.abs(mat_source_ux - k_sim.source.ux).sum() > tol):
            error_number =+ 1
            print(line + "k_sim.source.ux is not correct!")
        else:
            pass

        if (np.abs(np.squeeze(mat_ddx_k_shift_pos) - np.squeeze(ddx_k_shift_pos)).sum() > tol):
            error_number =+ 1
            print(line + "ddx_k_shift_pos is not correct!")
        else:
            pass
            # print("squeeze works!")

        # if (np.abs(mat_ddx_k_shift_pos - ddx_k_shift_pos).sum() > tol):
        #     error_number =+ 1
        #     print(line + "ddx_k_shift_pos is not correct! \n diff max:", np.abs(mat_ddx_k_shift_pos - ddx_k_shift_pos).sum(),
        #           "\nmax:", np.max(mat_ddx_k_shift_pos), "argmax:", np.argmax(mat_ddx_k_shift_pos),
        #           "min:", np.min(mat_ddx_k_shift_pos), "argmin:", np.argmin(mat_ddx_k_shift_pos),
        #           "\nmax:", np.max(ddx_k_shift_pos), "argmax:", np.argmax(ddx_k_shift_pos),
        #           "min:",np.min(ddx_k_shift_pos), "argmin:", np.argmin(ddx_k_shift_pos), "\n"
        #           "\nmax:", np.max(mat_ddx_k_shift_pos - ddx_k_shift_pos), "argmax:", np.argmax(mat_ddx_k_shift_pos - ddx_k_shift_pos),
        #           np.unravel_index(np.argmax(mat_ddx_k_shift_pos - ddx_k_shift_pos), np.shape(ddx_k_shift_pos)),
        #           "min:", np.min(mat_ddx_k_shift_pos - ddx_k_shift_pos), "argmin:", np.argmin(mat_ddx_k_shift_pos - ddx_k_shift_pos),
        #           np.unravel_index(np.argmin(mat_ddx_k_shift_pos - ddx_k_shift_pos), np.shape(ddx_k_shift_pos)), )
        # else:
        #     pass



        # if (np.abs(mat_ddx_k_shift_neg - ddx_k_shift_neg).sum() > tol):
        #     error_number =+ 1
        #     print(line + "ddx_k_shift_neg is not correct!",
        #           "\n\t", np.max(mat_ddx_k_shift_neg), np.argmax(mat_ddx_k_shift_neg),
        #           np.max(ddx_k_shift_neg), np.argmax(ddx_k_shift_neg),
        #           "\n\t", np.shape(np.abs(mat_ddx_k_shift_neg - ddx_k_shift_neg)))
        #     print(np.squeeze(np.abs(mat_ddx_k_shift_neg - ddx_k_shift_neg)) )
        # else:
        #     pass
        # if (np.abs(mat_ddy_k_shift_pos - ddy_k_shift_pos).sum() > tol):
        #     error_number =+ 1
        #     print(line + "ddy_k_shift_pos is not correct!")
        # else:
        #     pass
        # if (np.abs(mat_ddy_k_shift_neg - ddy_k_shift_neg).sum() > tol):
        #     error_number =+ 1
        #     print(line + "ddy_k_shift_neg is not correct!")
        # else:
        #     pass
        # if (np.abs(mat_ddz_k_shift_pos - ddz_k_shift_pos).sum() > tol):
        #     error_number =+ 1
        #     print(line + "ddz_k_shift_pos is not correct!")
        # else:
        #     pass
        # if (np.abs(mat_ddz_k_shift_neg - ddz_k_shift_neg).sum() > tol):
        #     error_number =+ 1
        #     print(line + "ddz_k_shift_pos is not correct!")
        # else:
        #     pass

    # ddx_k_shift_pos = mat_ddx_k_shift_pos
    # ddy_k_shift_pos = mat_ddy_k_shift_pos
    # ddz_k_shift_pos = mat_ddz_k_shift_pos

    # ddx_k_shift_neg = mat_ddx_k_shift_neg
    # ddy_k_shift_neg = mat_ddy_k_shift_neg
    # ddz_k_shift_neg = mat_ddz_k_shift_neg

    mat_eta = mat_contents['eta']
    mat_mu = mat_contents['mu']
    mat_lambda = mat_contents['lambda']
    mat_chi = mat_contents['chi']

    if checking:
        if (np.abs(mat_eta - eta).sum() > tol):
            error_number =+ 1
            print(line + "eta is not correct!", np.abs(mat_eta - eta).sum(), "\n",
                  np.max(mat_eta), np.argmax(mat_eta),
                  np.min(mat_eta), np.argmin(mat_eta),
                  np.max(eta), np.argmax(eta),
                  np.min(eta), np.argmin(eta))
        else:
            pass
        if (np.abs(mat_chi - chi).sum() > tol):
            error_number =+ 1
            print(line + "chi is not correct!", np.abs(mat_chi - chi).sum(), "\n",
                  np.max(mat_chi), np.argmax(mat_chi),
                  np.min(mat_chi), np.argmin(mat_chi),
                  np.max(chi), np.argmax(chi),
                  np.min(chi), np.argmin(chi))
        else:
            pass
        if (np.abs(mat_mu - mu).sum() > tol):
            error_number =+ 1
            print(line + "mu is not correct!", np.abs(mat_mu - mu).sum(), "\n",
                  np.max(mat_mu), np.argmax(mat_mu),
                  np.min(mat_mu), np.argmin(mat_mu),
                  np.max(mu), np.argmax(mu),
                  np.min(mu), np.argmin(mu))
        else:
            pass
        if (np.abs(mat_lambda - lame_lambda).sum() > tol):
            error_number =+ 1
            print(line + "mu is not correct!", np.abs(mat_lambda - lame_lambda).sum(), "\n",
                  np.max(mat_lambda), np.argmax(mat_lambda),
                  np.min(mat_lambda), np.argmin(mat_lambda),
                  np.max(lame_lambda), np.argmax(lame_lambda),
                  np.min(lame_lambda), np.argmin(lame_lambda))
        else:
            pass

    # print(mat_eta.flags.f_contiguous, eta.flags.f_contiguous)

    # =========================================================================
    # PREPARE VISUALISATIONS
    # =========================================================================

    # # pre-compute suitable axes scaling factor
    # if (options.plot_layout or options.plot_sim):
    #     x_sc, scale, prefix = scaleSI(np.max([kgrid.x_vec kgrid.y_vec kgrid.z_vec])) ##ok<ASGLU>


    # # throw error for currently unsupported plot layout feature
    # if options.plot_layout:
    #     raise TypeError('"PlotLayout" input is not currently supported.')


    # # initialise the figure used for animation if 'PlotSim' is set to 'True'
    # if options.plot_sim:
    #     kspaceFirstOrder_initialiseFigureWindow

    # # initialise movie parameters if 'RecordMovie' is set to 'True'
    # if options.record_movie:
    #     kspaceFirstOrder_initialiseMovieParameters


    # =========================================================================
    # LOOP THROUGH TIME STEPS
    # =========================================================================

    # this squeezes the arrays but also, if they don't exist, returns a np.array with no entries
    k_sim.s_source_pos_index = np.squeeze(k_sim.s_source_pos_index)
    k_sim.u_source_pos_index = np.squeeze(k_sim.u_source_pos_index)
    k_sim.p_source_pos_index = np.squeeze(k_sim.p_source_pos_index)

    # These should be zero indexed
    if hasattr(k_sim, 's_source_pos_index') and k_sim.s_source_pos_index.ndim != 0:
        k_sim.s_source_pos_index = np.squeeze(np.asarray(k_sim.s_source_pos_index)) - int(1)

    if hasattr(k_sim, 'u_source_pos_index') and k_sim.u_source_pos_index.ndim != 0:
        k_sim.u_source_pos_index = np.squeeze(k_sim.u_source_pos_index) - int(1)

    if hasattr(k_sim, 'p_source_pos_index') and k_sim.p_source_pos_index.ndim != 0:
        k_sim.p_source_pos_index = np.squeeze(k_sim.p_source_pos_index) - int(1)

    if hasattr(k_sim, 's_source_sig_index') and k_sim.s_source_sig_index is not None:
        k_sim.s_source_sig_index = np.squeeze(k_sim.s_source_sig_index) - int(1)

    if hasattr(k_sim, 'u_source_sig_index') and k_sim.u_source_sig_index is not None:
        k_sim.u_source_sig_index = np.squeeze(k_sim.u_source_sig_index)

    if hasattr(k_sim, 'p_source_sig_index') and k_sim.p_source_sig_index is not None:
        k_sim.p_source_sig_index = np.squeeze(k_sim.p_source_sig_index) - int(1)

    # These should be zero indexed. Note the x2, y2 and z2 indices do not need to be shifted
    record.x1_inside = int(record.x1_inside - 1)
    record.y1_inside = int(record.y1_inside - 1)
    record.z1_inside = int(record.z1_inside - 1)

    # update command line status
    print('\tprecomputation completed in', scale_time(TicToc.toc()))
    print('\tstarting time loop ...')

    # start time loop
    #for t_index in tqdm(np.arange(index_start, index_end, index_step, dtype=int)):
    for t_index in np.arange(0,12):

        # compute the gradients of the stress tensor (these variables do not
        # necessaily need to be stored, they could be computed as needed)
        dsxxdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(sxx_split_x + sxx_split_y + sxx_split_z, axis=0), order='F'), axis=0))
        dsyydy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(syy_split_x + syy_split_y + syy_split_z, axis=1), order='F'), axis=1))
        dszzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(szz_split_x + szz_split_y + szz_split_z, axis=2), order='F'), axis=2))

        dsxydx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(sxy_split_x + sxy_split_y, axis=0), order='F'), axis=0))
        dsxydy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(sxy_split_x + sxy_split_y, axis=1), order='F'), axis=1))

        dsxzdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(sxz_split_x + sxz_split_z, axis=0), order='F'), axis=0))
        dsxzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(sxz_split_x + sxz_split_z, axis=2), order='F'), axis=2))

        dsyzdy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(syz_split_y + syz_split_z, axis=1), order='F'), axis=1))
        dsyzdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(syz_split_y + syz_split_z, axis=2), order='F'), axis=2))

        if checking:
            if (t_index == load_index):
                if (np.abs(mat_dsxxdx - dsxxdx).sum() > tol):
                    error_number =+ 1
                    print(line + "dsxxdx is not correct!")
                else:
                    pass
                if (np.abs(mat_dsyydy - dsyydy).sum() > tol):
                    error_number =+ 1
                    print(line + "dsyydy is not correct!")
                else:
                    pass
                if (np.abs(mat_dszzdz - dszzdz).sum() > tol):
                    error_number =+ 1
                    print(line + "dszzdz is not correct!")
                else:
                    pass
                if (np.abs(mat_dsxydx - dsxydx).sum() > tol):
                    error_number =+ 1
                    print(line + "dsxydx is not correct!")
                else:
                    pass
                if (np.abs(mat_dsxydy - dsxydy).sum() > tol):
                    error_number =+ 1
                    print(line + "dsxydy is not correct!")
                else:
                    pass
                if (np.abs(mat_dsxzdx - dsxzdx).sum() > tol):
                    error_number =+ 1
                    print(line + "dsxzdx is not correct!")
                else:
                    pass
                if (np.abs(mat_dsxzdz - dsxzdz).sum() > tol):
                    error_number =+ 1
                    print(line + "dsxzdz is not correct!")
                else:
                    pass
                if (np.abs(mat_dsyzdy - dsyzdy).sum() > tol):
                    error_number =+ 1
                    print(line + "dsyzdy is not correct!")
                else:
                    pass
                if (np.abs(mat_dsyzdz - dsyzdz).sum() > tol):
                    error_number =+ 1
                    print(line + "dsyzdz is not correct!")
                else:
                    pass


        # calculate the split-field components of ux_sgx, uy_sgy, and uz_sgz at
        # the next time step using the components of the stress at the current
        # time step

        # ux_split_x = bsxfun(times, mpml_z,
        #                     bsxfun(times, mpml_y,e
        #                            bsxfun(times, pml_x_sgx,d
        #                                   bsxfun(times, mpml_z,c
        #                                          bsxfun(times, mpml_y, b
        #                                                 bsxfun(times, pml_x_sgx, ux_split_x))) + kgrid.dt * rho0_sgx_inv * dsxxdx))) a,c
        a = np.multiply(pml_x_sgx, ux_split_x, order='F')
        b = np.multiply(mpml_y, a, order='F')
        c = np.multiply(mpml_z, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgx_inv, dsxxdx, order='F')
        d = np.multiply(pml_x_sgx, c, order='F')
        e = np.multiply(mpml_y, d, order='F')
        ux_split_x = np.multiply(mpml_z, e, order='F')

        # ux_split_y = bsxfun(@times, mpml_x_sgx, bsxfun(@times, mpml_z,     bsxfun(@times, pml_y, \
        #             bsxfun(@times, mpml_x_sgx, bsxfun(@times, mpml_z,     bsxfun(@times, pml_y, ux_split_y))) \
        #             + kgrid.dt * rho0_sgx_inv * dsxydy)))
        a = np.multiply(pml_y, ux_split_y, order='F')
        b = np.multiply(mpml_z, a, order='F')
        c = np.multiply(mpml_x_sgx, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgx_inv, dsxydy, order='F')
        d = np.multiply(pml_y, c, order='F')
        e = np.multiply(mpml_z, d, order='F')
        ux_split_y = np.multiply(mpml_x_sgx, e, order='F')

        # ux_split_z = bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z, \
        #             bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z, ux_split_z))) \
        #             + kgrid.dt * rho0_sgx_inv * dsxzdz)))
        a = np.multiply(pml_z, ux_split_z, order='F')
        b = np.multiply(mpml_x_sgx, a, order='F')
        c = np.multiply(mpml_y, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgx_inv, dsxzdz, order='F')
        d = np.multiply(pml_z, c, order='F')
        e = np.multiply(mpml_x_sgx, d, order='F')
        ux_split_z = np.multiply(mpml_y, e, order='F')

        # uy_split_x = bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x, \
        #             bsxfun(@times, mpml_z,     bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x, uy_split_x))) \
        #             + kgrid.dt * rho0_sgy_inv * dsxydx)))
        a = np.multiply(pml_x, uy_split_x, order='F')
        b = np.multiply(mpml_y_sgy, a, order='F')
        c = np.multiply(mpml_z, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgy_inv, dsxydx, order='F')
        d = np.multiply(pml_x, c, order='F')
        e = np.multiply(mpml_y_sgy, d, order='F')
        uy_split_x = np.multiply(mpml_z, e, order='F')

        # uy_split_y = bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z, bsxfun(@times, pml_y_sgy, \
        #             bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z, bsxfun(@times, pml_y_sgy, uy_split_y))) \
        #             + kgrid.dt * rho0_sgy_inv * dsyydy)))
        a = np.multiply(pml_y_sgy, uy_split_y, order='F')
        b = np.multiply(mpml_z, a, order='F')
        c = np.multiply(mpml_x, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgy_inv, dsyydy, order='F')
        d = np.multiply(pml_y_sgy, c, order='F')
        e = np.multiply(mpml_z, d, order='F')
        uy_split_y = np.multiply(mpml_x, e, order='F')

        # uy_split_z = bsxfun(@times, mpml_y_sgy,
        #               bsxfun(@times, mpml_x,
        #                 bsxfun(@times, pml_z, \
        #                   bsxfun(@times, mpml_y_sgy,
        #                     bsxfun(@times, mpml_x,
        #                       bsxfun(@times, pml_z, uy_split_z))) \
        #                           + kgrid.dt * rho0_sgy_inv * dsyzdz)))
        a = np.multiply(pml_z, uy_split_z, order='F')
        b = np.multiply(mpml_x, a, order='F')
        c = np.multiply(mpml_y_sgy, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgy_inv, dsyzdz, order='F')
        d = np.multiply(pml_z, c, order='F')
        e = np.multiply(mpml_x, d, order='F')
        uy_split_z = np.multiply(mpml_y_sgy, e, order='F')

        # uz_split_x = bsxfun(@times, mpml_z_sgz, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
        #             bsxfun(@times, mpml_z_sgz, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, uz_split_x))) \
        #             + kgrid.dt * rho0_sgz_inv * dsxzdx)))
        a = np.multiply(pml_x, uz_split_x, order='F')
        b = np.multiply(mpml_y, a, order='F')
        c = np.multiply(mpml_z_sgz, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgz_inv, dsxzdx, order='F')
        d = np.multiply(pml_x, c, order='F')
        e = np.multiply(mpml_y, d, order='F')
        uz_split_x = np.multiply(mpml_z_sgz, e, order='F')

        # uz_split_y = bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y, \
        #             bsxfun(@times, mpml_x,     bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y, uz_split_y))) \
        #             + kgrid.dt * rho0_sgz_inv * dsyzdy)))
        a = np.multiply(pml_y, uz_split_y, order='F')
        b = np.multiply(mpml_z_sgz, a, order='F')
        c = np.multiply(mpml_x, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgz_inv, dsyzdy, order='F')
        d = np.multiply(pml_y, c, order='F')
        e = np.multiply(mpml_z_sgz, d, order='F')
        uz_split_y = np.multiply(mpml_x, e, order='F')

        # uz_split_z = bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x,     bsxfun(@times, pml_z_sgz, ...
        #             bsxfun(@times, mpml_y,     bsxfun(@times, mpml_x,     bsxfun(@times, pml_z_sgz, uz_split_z))) ...
        #             + dt .* rho0_sgz_inv .* dszzdz)));
        # uz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z_sgz, \
        #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z_sgz, uz_split_z))) \
        #             + kgrid.dt * rho0_sgz_inv * dszzdz)))
        a = np.multiply(pml_z_sgz, uz_split_z, order='F')
        b = np.multiply(mpml_x, a, order='F')
        c = np.multiply(mpml_y, b, order='F')
        c = c + kgrid.dt * np.multiply(rho0_sgz_inv, dszzdz, order='F')
        d = np.multiply(pml_z_sgz, c, order='F')
        e = np.multiply(mpml_x, d, order='F')
        uz_split_z = np.multiply(mpml_y, e, order='F')
        # uz_split_z = mpml_y * mpml_x * pml_z_sgz * (mpml_y * mpml_x * pml_z_sgz * uz_split_z + kgrid.dt * rho0_sgz_inv * dszzdz)

        if checking:
            if (t_index == load_index):

                if (np.abs(mat_ux_split_x - ux_split_x).sum() > tol):
                    error_number =+ 1
                    print(line + "ux_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_ux_split_y - ux_split_y).sum() > tol):
                    error_number =+ 1
                    print("ux_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_ux_split_z - ux_split_z).sum() > tol):
                    error_number =+ 1
                    print("ux_split_z is not correct!")
                else:
                    pass

                if (np.abs(mat_uy_split_x - uy_split_x).sum() > tol):
                    error_number =+ 1
                    print("uy_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_uy_split_y - uy_split_y).sum() > tol):
                    error_number =+ 1
                    print("uy_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_uy_split_z - uy_split_z).sum() > tol):
                    error_number =+ 1
                    print("uy_split_z is not correct!")
                else:
                    pass

                if (np.abs(mat_uz_split_x - uz_split_x).sum() > tol):
                    error_number =+ 1
                    print("uz_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_uz_split_y - uz_split_y).sum() > tol):
                    error_number =+ 1
                    print("uz_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_uz_split_z - uz_split_z).sum() > tol):
                    error_number =+ 1
                    print("uz_split_z is not correct!")
                else:
                    pass


        # add in the velocity source terms
        if k_sim.source_ux is not False and k_sim.source_ux >= t_index:

            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                #ux_split_x[k_sim.u_source_pos_index] = source.ux[k_sim.u_source_sig_index, t_index]
                ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] = np.squeeze(k_sim.source.ux[k_sim.u_source_sig_index, t_index])

            else:
                # add the source values to the existing field values
                # if checking:
                #     if (t_index == load_index):
                #         print('Here', np.shape(k_sim.source.ux))
                #         print(k_sim.u_source_pos_index)
                #         print(k_sim.u_source_sig_index)
                ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] = \
                  ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] + \
                  np.squeeze(k_sim.source.ux[k_sim.u_source_sig_index, t_index])


        if k_sim.source_uy is not False and k_sim.source_uy >= t_index:
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                #uy_split_y[k_sim.u_source_pos_index] = source.uy[k_sim.u_source_sig_index, t_index]
                uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] = np.squeeze(k_sim.source.uy[k_sim.u_source_sig_index, t_index])

            else:
                # add the source values to the existing field values
                uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] = \
                  uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] + \
                  np.squeeze(k_sim.source.uy[k_sim.u_source_sig_index, t_index])


        if k_sim.source_uz is not False and k_sim.source_uz >= t_index:
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                #uz_split_z[u_source_pos_index] = source.uz[u_source_sig_index, t_index]
                uz_split_z[np.unravel_index(k_sim.u_source_pos_index, uz_split_z.shape, order='F')] = np.squeeze(k_sim.source.uz[k_sim.u_source_sig_index, t_index])

            else:
                # add the source values to the existing field values
                #uz_split_z[u_source_pos_index] = uz_split_z[u_source_pos_index] + source.uz[u_source_sig_index, t_index]
                uz_split_z[np.unravel_index(k_sim.u_source_pos_index, uz_split_z.shape, order='F')] = \
                  uz_split_z[np.unravel_index(k_sim.u_source_pos_index, uz_split_z.shape, order='F')] + \
                  np.squeeze(k_sim.source.uz[k_sim.u_source_sig_index, t_index])


        # combine split field components (these variables do not necessarily
        # need to be stored, they could be computed when needed)
        ux_sgx = ux_split_x + ux_split_y + ux_split_z
        uy_sgy = uy_split_x + uy_split_y + uy_split_z
        uz_sgz = uz_split_x + uz_split_y + uz_split_z

        if checking:
            if (t_index == load_index):
                if (np.abs(mat_ux_sgx - ux_sgx).sum() > tol):
                    print("ux_sgx is NOT correct!")
                else:
                    print("ux_sgx is correct!", np.abs(mat_ux_sgx - ux_sgx).sum(), np.max(np.abs(mat_ux_sgx - ux_sgx)))
                if (np.abs(mat_uy_sgy - uy_sgy).sum() > tol):
                    print("uy_sgy is NOT correct!")
                else:
                    print("uy_sgy is correct!", np.abs(mat_uy_sgy - uy_sgy).sum(), np.max(np.abs(mat_uy_sgy - uy_sgy)))
                if (np.abs(mat_uz_sgz - uz_sgz).sum() > tol):
                    print("uz_sgz is NOT correct!")
                else:
                    print("uz_sgx is correct!", np.abs(mat_uz_sgz - uz_sgz).sum(), np.max(np.abs(mat_uz_sgz - uz_sgz)))


        # calculate the velocity gradients (these variables do not necessarily
        # need to be stored, they could be computed when needed)
        duxdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(ux_sgx, axis=0), order='F'), axis=0)) #
        duxdy = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(ux_sgx, axis=1), order='F'), axis=1)) #
        duxdz = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(ux_sgx, axis=2), order='F'), axis=2))

        # changed here! ux_sgy -> uy_sgy
        duydx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(uy_sgy, axis=0), order='F'), axis=0))
        duydy = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(uy_sgy, axis=1), order='F'), axis=1))
        duydz = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(uy_sgy, axis=2), order='F'), axis=2))

        # changed here! ux_sgz -> uz_sgz
        duzdx = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(uz_sgz, axis=0), order='F'), axis=0))
        duzdy = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(uz_sgz, axis=1), order='F'), axis=1))
        duzdz = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(uz_sgz, axis=2), order='F'), axis=2))

        if checking:
            if (t_index == load_index):

                a_x = np.fft.fft(ux_sgx, axis=0)
                if (np.abs(mat_a_x - a_x).sum() > tol):
                    error_number =+ 1
                    print(line + "a_x is not correct!", tol, np.abs(mat_a_x - a_x).sum(), np.abs(mat_a_x).sum(), np.abs(a_x).sum(),
                    np.max(np.abs(mat_a_x)), np.max(np.abs(a_x)) )
                else:
                    print(line + "a_x is correct!", np.abs(mat_a_x - a_x).sum(), np.max(np.abs(mat_a_x - a_x)))

                #b_x = np.expand_dims(ddx_k_shift_neg, axis=-1) * np.fft.fft(ux_sgx, axis=0)
                #print(b_x.shape)
                b_x = np.multiply(ddx_k_shift_neg, np.fft.fft(ux_sgx, axis=0), order='F')
                if (np.abs(mat_b_x - b_x).sum() > tol):
                    error_number =+ 1
                    print(line + "b_x is not correct!",
                          "\n\t", np.abs(mat_b_x - b_x).sum(),
                          "\n\t", np.abs(mat_b_x).sum(),
                          "\n\t", np.abs(b_x).sum(),
                          "\n\t", np.max(np.abs(mat_b_x - b_x)),
                          "\n\t", np.max(np.abs(mat_b_x)),
                          "\n\t", np.max(np.abs(b_x)) )
                else:
                    print(line + "b_x is correct!")

                c_x = np.fft.ifft(b_x, axis=0)
                if (np.abs(mat_c_x - c_x).sum() > tol):
                    error_number =+ 1
                    print(line + "c_x is not correct!", np.abs(mat_c_x - c_x).sum(), np.abs(mat_c_x).sum(), np.abs(c_x).sum(),
                    np.max(np.abs(mat_c_x)), np.max(np.abs(c_x)) )
                else:
                    print(line + "c_x is correct!")

                a_y = np.fft.fft(ux_sgx, axis=1)
                if (np.abs(mat_a_y - a_y).sum() > tol):
                    error_number =+ 1
                    print(line + "a_y is not correct!", tol, np.abs(mat_a_y - a_y).sum(), np.abs(mat_a_y).sum(), np.abs(a_y).sum(),
                    np.max(np.abs(mat_a_y)), np.max(np.abs(a_y)) )
                else:
                    print(line + "a_y is correct!")

                b_y = ddy_k_shift_pos * np.fft.fft(ux_sgx, axis=1)
                if (np.abs(mat_b_y - b_y).sum() > tol):
                    error_number =+ 1
                    print(line + "b_y is not correct!", np.abs(mat_b_y - b_y).sum(), np.abs(mat_b_y).sum(), np.abs(b_y).sum(),
                    np.max(np.abs(mat_b_y)), np.max(np.abs(b_y)) )
                else:
                    print(line + "b_y is correct!")

                c_y = np.fft.ifft(ddy_k_shift_pos * np.fft.fft(ux_sgx, axis=1), axis=1)
                if (np.abs(mat_c_y - c_y).sum() > tol):
                    error_number =+ 1
                    print(line + "c_y is not correct!", np.abs(mat_c_y - c_y).sum(), np.abs(mat_c_y).sum(), np.abs(c_y).sum(),
                    np.max(np.abs(mat_c_y)), np.max(np.abs(c_y)) )
                else:
                    print(line + "c_y is correct!")










                if (np.abs(mat_duxdx - duxdx).sum() > tol):
                    error_number =+ 1
                    print(line + "duxdx is not correct!", np.abs(mat_duxdx - duxdx).sum(), np.abs(mat_duxdx).sum(), np.abs(duxdx).sum(),
                    np.max(np.abs(mat_duxdx)), np.max(np.abs(duxdx)) )
                else:
                    print(line + "duxdx is correct!", np.abs(mat_duxdx - duxdx).sum(), np.abs(mat_duxdx).sum(), np.abs(duxdx).sum(),
                    np.max(np.abs(mat_duxdx)), np.max(np.abs(duxdx)) )
                if (np.abs(mat_duxdy - duxdy).sum() > tol):
                    error_number =+ 1
                    print(line + "duxdy is not correct!" + "\tdiff:", np.abs(mat_duxdy - duxdy).sum(), np.abs(mat_duxdy).sum(), np.abs(duxdy).sum(),
                    np.max(np.abs(mat_duxdy)), np.max(np.abs(duxdy)))
                else:
                    pass

                if (np.abs(mat_duxdz - duxdz).sum() > tol):
                    print("duxdz is not correct!")
                else:
                    pass
                if (np.abs(mat_duydx - duydx).sum() > tol):
                    print("duydx is not correct!")
                else:
                    pass
                if (np.abs(mat_duydy - duydy).sum() > tol):
                    print("duydy is not correct!")
                else:
                    pass
                if (np.abs(mat_duydz - duydz).sum() > tol):
                    print("duydz is not correct!")
                else:
                    pass
                if (np.abs(mat_duzdx - duzdx).sum() > tol):
                    print("duzdx is not correct!")
                else:
                    pass
                if (np.abs(mat_duzdy - duzdy).sum() > tol):
                    print("duzdy is not correct!")
                else:
                    pass
                if (np.abs(mat_duzdz - duzdz).sum() > tol):
                    print("duzdz is not correct!")
                else:
                    pass


        if options.kelvin_voigt_model:

            # compute additional gradient terms needed for the Kelvin-Voigt model

            temp = np.multiply((dsxxdx + dsxydy + dsxzdz), rho0_sgx_inv, order='F')
            dduxdxdt = np.real(np.fft.ifft(np.multiply(ddx_k_shift_neg, np.fft.fft(temp, axis=0), order='F'), axis=0))
            dduxdydt = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(temp, axis=1), order='F'), axis=1))
            dduxdzdt = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(temp, axis=2), order='F'), axis=2))

            temp = np.multiply((dsxydx + dsyydy + dsyzdz), rho0_sgy_inv, order='F')
            dduydxdt = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(temp, axis=0), order='F'), axis=0))
            dduydydt = np.real(np.fft.ifft(np.multiply(ddy_k_shift_neg, np.fft.fft(temp, axis=1), order='F'), axis=1))
            dduydzdt = np.real(np.fft.ifft(np.multiply(ddz_k_shift_pos, np.fft.fft(temp, axis=2), order='F'), axis=2))

            temp = np.multiply((dsxzdx + dsyzdy + dszzdz), rho0_sgz_inv, order='F')
            dduzdxdt = np.real(np.fft.ifft(np.multiply(ddx_k_shift_pos, np.fft.fft(temp, axis=0), order='F'), axis=0))
            dduzdydt = np.real(np.fft.ifft(np.multiply(ddy_k_shift_pos, np.fft.fft(temp, axis=1), order='F'), axis=1))
            dduzdzdt = np.real(np.fft.ifft(np.multiply(ddz_k_shift_neg, np.fft.fft(temp, axis=2), order='F'), axis=2))

            if checking:
                if (t_index == load_index):
                    if (np.abs(mat_dduxdxdt - dduxdxdt).sum() > tol):
                        print("dduxdxdt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduxdydt - dduxdydt).sum() > tol):
                        print("dduxdydt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduxdzdt - dduxdzdt).sum() > tol):
                        print("dduxdzdt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduydxdt - dduydxdt).sum() > tol):
                        print("dduydxdt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduydydt - dduydydt).sum() > tol):
                        print("dduydydt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduydzdt - dduydzdt).sum() > tol):
                        print("dduydzdt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduzdxdt - dduzdxdt).sum() > tol):
                        print("dduzdxdt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduzdydt - dduzdydt).sum() > tol):
                        print("dduzdydt is not correct!")
                    else:
                        pass
                    if (np.abs(mat_dduzdzdt - dduzdzdt).sum() > tol):
                        print("dduzdzdt is not correct!")
                    else:
                        pass


            # update the normal shear components of the stress tensor using a
            # Kelvin-Voigt model with a split-field multi-axial pml

            #  sxx_split_x = bsxfun(@times, mpml_z,
            #                  bsxfun(@times, mpml_y,
            #                    bsxfun(@times, pml_x, \
            #                      bsxfun(@times, mpml_z,
            #                        bsxfun(@times, mpml_y,
            #                          bsxfun(@times, pml_x, sxx_split_x) ) ) \
            #                            + kgrid.dt * (2.0 * mu + lame_lambda) * duxdx + kgrid.dt * (2.0 * eta + chi) * dduxdxdt)))
            sxx_split_x = np.multiply(mpml_z,
                            np.multiply(mpml_y,
                              np.multiply(pml_x,
                                np.multiply(mpml_z,
                                  np.multiply(mpml_y,
                                    np.multiply(pml_x, sxx_split_x, order='F'), order='F'), order='F') + \
                                                 kgrid.dt * np.multiply(2.0 * mu + lame_lambda, duxdx, order='F') + \
                                                 kgrid.dt * np.multiply(2.0 * eta + chi, dduxdxdt, order='F'), order='F'), order='F'), order='F')

            # sxx_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * mpml_y * np.multiply(pml_x, sxx_split_x) + \
            #                                           kgrid.dt * np.multiply(2.0 * mu + lame_lambda, duxdx, order='F') + \
            #                                           kgrid.dt * np.multiply(2.0 * eta + chi, dduxdxdt, order='F') )))


            # sxx_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, sxx_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy \
            #             + kgrid.dt * chi * dduydydt)))
            temp0 = np.copy(sxx_split_y)
            temp0 = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * temp0)) +
                                    kgrid.dt * np.multiply(lame_lambda, duydy, order='F') + kgrid.dt * np.multiply(chi, dduydydt, order='F') )))
            temp1 = np.copy(sxx_split_y)
            temp1 = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * temp1)) +
                                    kgrid.dt * lame_lambda * duydy + kgrid.dt * chi * dduydydt)))
            temp2 = np.copy(sxx_split_y)
            a = np.multiply(pml_y, temp2, order='F')
            b = np.multiply(mpml_z, a, order='F')
            c = np.multiply(mpml_x, b, order='F')
            c = c + kgrid.dt * np.multiply(lame_lambda, duydy, order='F') + kgrid.dt * np.multiply(chi, dduydydt, order='F')
            d = np.multiply(pml_y, c, order='F')
            e = np.multiply(mpml_z, d, order='F')
            temp2 = np.multiply(mpml_x, e, order='F')
            # print(np.abs(temp0).sum(), np.abs(temp1).sum(), np.abs(temp2).sum() )
            # print(np.max(np.abs(temp0)), np.max(np.abs(temp0)), np.max(np.abs(temp0)) )
            # print(np.min(np.abs(temp0)), np.min(np.abs(temp0)), np.min(np.abs(temp0)) )


            # sxx_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, sxx_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz \
            #             + kgrid.dt * chi * dduzdzdt)))
            sxx_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * sxx_split_z)) +
                        kgrid.dt * lame_lambda * duzdz + kgrid.dt * chi * dduzdzdt)))

            # syy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, syy_split_x))) \
            #             + kgrid.dt * (lame_lambda * duxdx + kgrid.dt * chi * dduxdxdt) )))
            syy_split_x = mpml_z * mpml_y * pml_x * (
                mpml_z * mpml_y * pml_x * syy_split_x +
                kgrid.dt * (lame_lambda * duxdx + kgrid.dt * chi * dduxdxdt))

            # syy_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #               bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, syy_split_y))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duydy \
            #             + kgrid.dt * (2 * eta + chi) * dduydydt )))
            syy_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * syy_split_y)) + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duydy + \
                                               kgrid.dt * (2.0 * eta + chi) * dduydydt)))

            # syy_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, syy_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz \
            #             + kgrid.dt * chi * dduzdzdt) ))
            syy_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * syy_split_z)) +\
                                               kgrid.dt * lame_lambda * duzdz + \
                                               kgrid.dt * chi * dduzdzdt)))

            # szz_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, szz_split_x))) \
            #             + kgrid.dt * lame_lambda * duxdx\
            #             + kgrid.dt * chi * dduxdxdt)))
            szz_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * (mpml_y * (pml_x * szz_split_x)) + \
                                               kgrid.dt * lame_lambda * duxdx + \
                                               kgrid.dt * chi * dduxdxdt)))

            # szz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, szz_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy \
            #             + kgrid.dt * chi * dduydydt)))
            szz_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * szz_split_y)) + \
                                               kgrid.dt * lame_lambda * duydy + \
                                               kgrid.dt * chi * dduydydt)))

            # szz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, szz_split_z))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duzdz \
            #             + kgrid.dt * (2 * eta + chi) * dduzdzdt)))
            szz_split_z = mpml_y * (mpml_x * (pml_z * ( mpml_y * (mpml_x * (pml_z * szz_split_z)) + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duzdz + \
                                               kgrid.dt * (2.0 * eta + chi) * dduzdzdt)))

            # sxy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_x_sgx, sxy_split_x))) \
            #             + kgrid.dt * mu_sgxy * duydx \
            #             + kgrid.dt * eta_sgxy * dduydxdt)))
            sxy_split_x = mpml_z * (mpml_y_sgy * (pml_x_sgx * (mpml_z * (mpml_y_sgy * (pml_x_sgx * sxy_split_x)) + \
                                                               kgrid.dt * mu_sgxy * duydx + \
                                                               kgrid.dt * eta_sgxy * dduydxdt)))

            # sxy_split_y = bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, sxy_split_y))) \
            #             + kgrid.dt * mu_sgxy * duxdy \
            #             + kgrid.dt * eta_sgxy * dduxdydt)))
            sxy_split_y = mpml_z * (mpml_x_sgx * (pml_y_sgy * (mpml_z * (mpml_x_sgx * (pml_y_sgy * sxy_split_y)) + \
                                               kgrid.dt * mu_sgxy * duxdy + \
                                               kgrid.dt * eta_sgxy * dduxdydt)))

            # sxz_split_x = bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, sxz_split_x))) \
            #             + kgrid.dt * mu_sgxz * duzdx \
            #             + kgrid.dt * eta_sgxz * dduzdxdt)))
            sxz_split_x = mpml_y * (mpml_z_sgz * (pml_x_sgx * (mpml_y * (mpml_z_sgz * (pml_x_sgx * sxz_split_x)) + \
                                                               kgrid.dt * mu_sgxz * duzdx + \
                                                               kgrid.dt * eta_sgxz * dduzdxdt)))

            # sxz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, sxz_split_z))) \
            #             + kgrid.dt * mu_sgxz * duxdz \
            #             + kgrid.dt * eta_sgxz * dduxdzdt)))
            sxz_split_z = mpml_y * (mpml_x_sgx * (pml_z_sgz * (mpml_y * (mpml_x_sgx * (pml_z_sgz * sxz_split_z)) + \
                                                               kgrid.dt * mu_sgxz * duxdz + \
                                                               kgrid.dt * eta_sgxz * dduxdzdt)))

            # syz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, syz_split_y))) \
            #             + kgrid.dt * mu_sgyz * duzdy \
            #             + kgrid.dt * eta_sgyz * dduzdydt)))
            syz_split_y = mpml_x * (mpml_z_sgz * (pml_y_sgy * (mpml_x * (mpml_z_sgz * (pml_y_sgy * syz_split_y)) + \
                                                               kgrid.dt * mu_sgyz * duzdy + \
                                                               kgrid.dt * eta_sgxz * dduzdydt)))

            # syz_split_z = bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, syz_split_z))) \
            #             + kgrid.dt * mu_sgyz * duydz \
            #             + kgrid.dt * eta_sgyz * dduydzdt)))
            syz_split_z = mpml_x * (mpml_y_sgy * (pml_z_sgz * (mpml_x * (mpml_y_sgy * (pml_z_sgz * syz_split_z)) + \
                                                               kgrid.dt * mu_sgyz * duzdz + \
                                                               kgrid.dt * eta_sgxz * dduydzdt)))

        else:

            print('NOT KV')
            temp = 2.0 * mu + lame_lambda

            # update the normal and shear components of the stress tensor using
            # a lossless elastic model with a split-field multi-axial pml
            # sxx_split_x = bsxfun(@times, mpml_z,
            #                      bsxfun(@times, mpml_y,
            #                             bsxfun(@times, pml_x,
            #                                    bsxfun(@times, mpml_z,
            #                                           bsxfun(@times, mpml_y,
            #                                                  bsxfun(@times, pml_x, sxx_split_x))) + kgrid.dt * (2 * mu + lame_lambda) * duxdx ) ))
            sxx_split_x = mpml_z * mpml_y * pml_x * (mpml_z * mpml_y * pml_x * sxx_split_x + kgrid.dt * (2.0 * mu + lame_lambda) * duxdx)

            # sxx_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, sxx_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy )))
            sxx_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * sxx_split_y)) + \
                                               kgrid.dt * lame_lambda * duydy)))
            # sxx_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, sxx_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz )))
            sxx_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * sxx_split_z)) + \
                                               kgrid.dt * lame_lambda * duzdz)))

            # syy_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, syy_split_x))) \
            #             + kgrid.dt * lame_lambda * duxdx)))
            syy_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * (mpml_y * (pml_x * syy_split_x)) + \
                                               kgrid.dt * lame_lambda * duxdx)))
            # syy_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, syy_split_y))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duydy )))
            syy_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * syy_split_y)) + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duydy)))
            # syy_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, syy_split_z))) \
            #             + kgrid.dt * lame_lambda * duzdz )))
            syy_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * syy_split_z)) + \
                                               kgrid.dt * lame_lambda * duzdz)))

            # szz_split_x = bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_y, bsxfun(@times, pml_x, szz_split_x))) \
            #             + kgrid.dt * lame_lambda * duxdx)))
            szz_split_x = mpml_z * (mpml_y * (pml_x * (mpml_z * (mpml_y * (pml_x * szz_split_x)) + \
                                               kgrid.dt * lame_lambda * duxdx)))
            # szz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z, bsxfun(@times, pml_y, szz_split_y))) \
            #             + kgrid.dt * lame_lambda * duydy )))
            szz_split_y = mpml_x * (mpml_z * (pml_y * (mpml_x * (mpml_z * (pml_y * szz_split_y)) + \
                                               kgrid.dt * lame_lambda * duydy)))
            # szz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x, bsxfun(@times, pml_z, szz_split_z))) \
            #             + kgrid.dt * (2 * mu + lame_lambda) * duzdz )))
            szz_split_z = mpml_y * (mpml_x * (pml_z * (mpml_y * (mpml_x * (pml_z * szz_split_z)) + \
                                               kgrid.dt * (2.0 * mu + lame_lambda) * duzdz)))

            # sxy_split_x = bsxfun(@times, mpml_z,
            #                      bsxfun(@times, mpml_y_sgy,
            #                             bsxfun(@times, pml_x_sgx, \
            #                                    bsxfun(@times, mpml_z,
            #                                           bsxfun(@times, mpml_y_sgy,
            #                                                  bsxfun(@times, pml_x_sgx, sxy_split_x))) +\
            #                                                  kgrid.dt * mu_sgxy * duydx)))
            sxy_split_x = mpml_z * (mpml_y_sgy * (pml_x_sgx * (mpml_z * (mpml_y_sgy * (pml_x_sgx * sxy_split_x)) + \
                                                               kgrid.dt * mu_sgxy * duydx)))
            # sxy_split_y = bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_z, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_y_sgy, sxy_split_y))) \
            #             + kgrid.dt * mu_sgxy * duxdy)))
            sxy_split_y = mpml_z * (mpml_x_sgx * (pml_y_sgy * (mpml_z * (mpml_x_sgx * (pml_y_sgy * sxy_split_y)) + \
                                                               kgrid.dt * mu_sgxy * duxdy)))
            # sxz_split_x = bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_x_sgx, sxz_split_x))) \
            #             + kgrid.dt * mu_sgxz * duzdx)))
            sxz_split_x = mpml_y * (mpml_z_sgz * (pml_x_sgx * (mpml_y * (mpml_z_sgz * (pml_x_sgx * sxz_split_x)) + \
                                                               kgrid.dt * mu_sgxz * duzdx)))
            # sxz_split_z = bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_y, bsxfun(@times, mpml_x_sgx, bsxfun(@times, pml_z_sgz, sxz_split_z))) \
            #             + kgrid.dt * mu_sgxz * duxdz)))
            sxz_split_z = mpml_y * (mpml_x_sgx * (pml_z_sgz * (mpml_y * (mpml_x_sgx * (pml_z_sgz * sxz_split_z)) + \
                                                               kgrid.dt * mu_sgxz * duxdz)))

            # syz_split_y = bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_z_sgz, bsxfun(@times, pml_y_sgy, syz_split_y))) \
            #             + kgrid.dt * mu_sgyz * duzdy)))
            syz_split_y = mpml_x * (mpml_z_sgz * (pml_y_sgy * (mpml_x * (mpml_z_sgz * (pml_y_sgy * syz_split_y)) + \
                                                               kgrid.dt * mu_sgyz * duzdy)))

            # syz_split_z = bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, \
            #             bsxfun(@times, mpml_x, bsxfun(@times, mpml_y_sgy, bsxfun(@times, pml_z_sgz, syz_split_z))) \
            #             + kgrid.dt * mu_sgyz * duydz)))
            syz_split_z = mpml_x * (mpml_y_sgy * (pml_z_sgz * (mpml_x * (mpml_y_sgy * (pml_z_sgz * syz_split_z)) + \
                                                               kgrid.dt * mu_sgyz * duydz)))

            # syz_split_z = mpml_x * mpml_y_sgy * pml_z_sgz * (mpml_x * mpml_y_sgy * pml_z_sgz * syz_split_z + kgrid.dt * mu_sgyz * duydz)
            # result = compute_syz_split_z(mpml_x, mpml_y_sgy, pml_z_sgz, syz_split_z, kgrid.dt, mu_sgyz, duydz)

        # add in the pre-scaled stress source terms

        if hasattr(k_sim, 's_source_sig_index'):
            if isinstance(k_sim.s_source_sig_index, str):
                if k_sim.s_source_sig_index == ':':
                    print("converting s_source_sig_index")
                    if (k_sim.source_sxx is not False):
                        s_source_sig_index = np.shape(source.sxx)[0]
                    elif (k_sim.source_syy is not False):
                        s_source_sig_index = np.shape(source.syy)[0]
                    elif (k_sim.source_szz is not False):
                        s_source_sig_index = np.shape(source.szz)[0]
                    elif (k_sim.source_sxy is not False):
                        s_source_sig_index = np.shape(source.sxy)[0]
                    elif (k_sim.source_sxy is not False):
                        s_source_sig_index = np.shape(source.sxz)[0]
                    elif (k_sim.source_syz is not False):
                        s_source_sig_index = np.shape(source.syz)[0]
                    else:
                        raise RuntimeError('Need to set s_source_sig_index')

        # # First find whether source locations are provided and how.
        # if np.ndim(np.squeeze(k_sim.s_source_pos_index)) != 0:
        #     n_pos = np.shape(np.squeeze(k_sim.s_source_pos_index))[0]
        # else:
        #     n_pos = None

        if (k_sim.source_sxx is not False and t_index < np.size(source.sxx)):
            print('sxx')

            if (source.s_mode == 'dirichlet'):

                # enforce the source values as a dirichlet boundary condition
                # sxx_split_x[s_source_pos_index] = source.sxx[s_source_sig_index, t_index]
                # sxx_split_y[s_source_pos_index] = source.sxx[s_source_sig_index, t_index]
                # sxx_split_z[s_source_pos_index] = source.sxx[s_source_sig_index, t_index]
                sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] = k_sim.source.sxx[np.unravel_index(k_sim.s_source_sig_index, sxx_split_x.shape, order='F'), t_index]
                sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] = k_sim.source.sxx[np.unravel_index(k_sim.s_source_sig_index, sxx_split_y.shape, order='F'), t_index]
                sxx_split_z[np.unravel_index(k_sim.s_source_pos_index, sxx_split_z.shape, order='F')] = k_sim.source.sxx[np.unravel_index(k_sim.s_source_sig_index, sxx_split_z.shape, order='F'), t_index]

            else:

                # add the source values to the existing field values
                # sxx_split_x[s_source_pos_index] = sxx_split_x[s_source_pos_index] + source.sxx[s_source_sig_index, t_index]
                # sxx_split_y[s_source_pos_index] = sxx_split_y[s_source_pos_index] + source.sxx[s_source_sig_index, t_index]
                # sxx_split_z[s_source_pos_index] = sxx_split_z[s_source_pos_index] + source.sxx[s_source_sig_index, t_index]
                sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] = sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] + \
                  k_sim.source.sxx[np.unravel_index(k_sim.s_source_sig_index, sxx_split_x.shape, order='F'), t_index]
                sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] = sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] + \
                  k_sim.source.sxx[np.unravel_index(k_sim.s_source_sig_index, sxx_split_y.shape, order='F'), t_index]
                sxx_split_z[np.unravel_index(k_sim.s_source_pos_index, sxx_split_z.shape, order='F')] = sxx_split_z[np.unravel_index(k_sim.s_source_pos_index, sxx_split_z.shape, order='F')] + \
                  k_sim.source.sxx[np.unravel_index(k_sim.s_source_sig_index, sxx_split_z.shape, order='F'), t_index]

        if (k_sim.source_syy is not False and t_index < np.size(source.syy)):
            print('syy')
            if (source.s_mode == 'dirichlet'):

                # enforce the source values as a dirichlet boundary condition
                # syy_split_x[s_source_pos_index] = source.syy[s_source_sig_index, t_index]
                # syy_split_y[s_source_pos_index] = source.syy[s_source_sig_index, t_index]
                # syy_split_z[s_source_pos_index] = source.syy[s_source_sig_index, t_index]
                syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] = k_sim.source.syy[np.unravel_index(k_sim.s_source_sig_index, syy_split_x.shape, order='F'), t_index]
                syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] = k_sim.source.syy[np.unravel_index(k_sim.s_source_sig_index, syy_split_y.shape, order='F'), t_index]
                syy_split_z[np.unravel_index(k_sim.s_source_pos_index, syy_split_z.shape, order='F')] = k_sim.source.syy[np.unravel_index(k_sim.s_source_sig_index, syy_split_z.shape, order='F'), t_index]


            else:

                # add the source values to the existing field values
                # syy_split_x[s_source_pos_index] = syy_split_x[s_source_pos_index] + source.syy[s_source_sig_index, t_index]
                # syy_split_y[s_source_pos_index] = syy_split_y[s_source_pos_index] + source.syy[s_source_sig_index, t_index]
                # syy_split_z[s_source_pos_index] = syy_split_z[s_source_pos_index] + source.syy[s_source_sig_index, t_index]
                syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] = syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] + \
                  k_sim.source.syy[np.unravel_index(k_sim.s_source_sig_index, syy_split_x.shape, order='F'), t_index]
                syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] = syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] + \
                  k_sim.source.syy[np.unravel_index(k_sim.s_source_sig_index, syy_split_y.shape, order='F'), t_index]
                syy_split_z[np.unravel_index(k_sim.s_source_pos_index, syy_split_z.shape, order='F')] = syy_split_z[np.unravel_index(k_sim.s_source_pos_index, syy_split_z.shape, order='F')] + \
                  k_sim.source.syy[np.unravel_index(k_sim.s_source_sig_index, syy_split_z.shape, order='F'), t_index]

        if (k_sim.source_szz is not False and t_index < np.size(source.syy)):
            print('szz')
            if (source.s_mode == 'dirichlet'):

                # enforce the source values as a dirichlet boundary condition
                szz_split_x[s_source_pos_index] = source.szz[s_source_sig_index, t_index]
                szz_split_y[s_source_pos_index] = source.szz[s_source_sig_index, t_index]
                szz_split_z[s_source_pos_index] = source.szz[s_source_sig_index, t_index]

            else:

                # # add the source values to the existing field values
                # szz_split_x[s_source_pos_index] = szz_split_x[s_source_pos_index] + source.szz[s_source_sig_index, t_index]
                # szz_split_y[s_source_pos_index] = szz_split_y[s_source_pos_index] + source.szz[s_source_sig_index, t_index]
                # szz_split_z[s_source_pos_index] = szz_split_z[s_source_pos_index] + source.szz[s_source_sig_index, t_index]

                szz_split_x[np.unravel_index(k_sim.s_source_pos_index, szz_split_x.shape, order='F')] = szz_split_x[np.unravel_index(k_sim.s_source_pos_index, szz_split_x.shape, order='F')] + \
                  k_sim.source.szz[np.unravel_index(k_sim.s_source_sig_index, szz_split_x.shape, order='F'), t_index]
                szz_split_y[np.unravel_index(k_sim.s_source_pos_index, szz_split_y.shape, order='F')] = szz_split_y[np.unravel_index(k_sim.s_source_pos_index, szz_split_y.shape, order='F')] + \
                  k_sim.source.szz[np.unravel_index(k_sim.s_source_sig_index, szz_split_y.shape, order='F'), t_index]
                szz_split_z[np.unravel_index(k_sim.s_source_pos_index, szz_split_z.shape, order='F')] = szz_split_z[np.unravel_index(k_sim.s_source_pos_index, szz_split_z.shape, order='F')] + \
                  k_sim.source.szz[np.unravel_index(k_sim.s_source_sig_index, szz_split_z.shape, order='F'), t_index]

        if (k_sim.source_sxy is not False and t_index < np.size(source.sxy)):
            print('sxy')
            if (source.s_mode == 'dirichlet'):

                # enforce the source values as a dirichlet boundary condition
                sxy_split_x[s_source_pos_index] = source.sxy[s_source_sig_index, t_index]
                sxy_split_y[s_source_pos_index] = source.sxy[s_source_sig_index, t_index]

            else:

                # add the source values to the existing field values
                # sxy_split_x[s_source_pos_index] = sxy_split_x[s_source_pos_index] + source.sxy[s_source_sig_index, t_index]
                # sxy_split_y[s_source_pos_index] = sxy_split_y[s_source_pos_index] + source.sxy[s_source_sig_index, t_index]

                sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] = sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] + \
                  k_sim.source.sxy[np.unravel_index(k_sim.s_source_sig_index, sxy_split_x.shape, order='F'), t_index]
                sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] = sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] + \
                  k_sim.source.sxy[np.unravel_index(k_sim.s_source_sig_index, sxy_split_y.shape, order='F'), t_index]

        if (k_sim.source_sxz is not False and t_index < np.size(source.sxz)):
            print('sxz')
            if (source.s_mode == 'dirichlet'):

                # enforce the source values as a dirichlet boundary condition
                sxz_split_x[s_source_pos_index] = source.sxz[s_source_sig_index, t_index]
                sxz_split_z[s_source_pos_index] = source.sxz[s_source_sig_index, t_index]

            else:

                # add the source values to the existing field values
                # sxz_split_x[s_source_pos_index] = sxz_split_x[s_source_pos_index] + source.sxz[s_source_sig_index, t_index]
                # sxz_split_z[s_source_pos_index] = sxz_split_z[s_source_pos_index] + source.sxz[s_source_sig_index, t_index]
                sxz_split_x[np.unravel_index(k_sim.s_source_pos_index, sxz_split_x.shape, order='F')] = sxz_split_x[np.unravel_index(k_sim.s_source_pos_index, sxz_split_x.shape, order='F')] + \
                  k_sim.source.sxz[np.unravel_index(k_sim.s_source_sig_index, sxz_split_x.shape, order='F'), t_index]
                sxz_split_z[np.unravel_index(k_sim.s_source_pos_index, sxz_split_z.shape, order='F')] = sxz_split_z[np.unravel_index(k_sim.s_source_pos_index, sxz_split_z.shape, order='F')] + \
                  k_sim.source.sxz[np.unravel_index(k_sim.s_source_sig_index, sxz_split_z.shape, order='F'), t_index]

        if (k_sim.source_syz is not False and t_index < np.size(source.syz)):
            print('syz')
            if (source.s_mode == 'dirichlet'):

                # enforce the source values as a dirichlet boundary condition
                syz_split_y[s_source_pos_index] = source.syz[s_source_sig_index, t_index]
                syz_split_z[s_source_pos_index] = source.syz[s_source_sig_index, t_index]

            else:

                # add the source values to the existing field values
                # syz_split_y[s_source_pos_index] = syz_split_y[s_source_pos_index] + source.syz[s_source_sig_index, t_index]
                # syz_split_z[s_source_pos_index] = syz_split_z[s_source_pos_index] + source.syz[s_source_sig_index, t_index]
                syz_split_y[np.unravel_index(k_sim.s_source_pos_index, syz_split_y.shape, order='F')] = syz_split_y[np.unravel_index(k_sim.s_source_pos_index, syz_split_y.shape, order='F')] + \
                  k_sim.source.syz[np.unravel_index(k_sim.s_source_sig_index, syz_split_y.shape, order='F'), t_index]
                syz_split_z[np.unravel_index(k_sim.s_source_pos_index, syz_split_z.shape, order='F')] = syz_split_z[np.unravel_index(k_sim.s_source_pos_index, syz_split_z.shape, order='F')] + \
                  k_sim.source.syz[np.unravel_index(k_sim.s_source_sig_index, syz_split_z.shape, order='F'), t_index]

        if checking:
            if (t_index == load_index):
                if (np.abs(mat_sxx_split_x - sxx_split_x).sum() > tol):
                    print("sxx_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_sxx_split_y - sxx_split_y).sum() > tol):
                    print("sxx_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_sxx_split_z - sxx_split_z).sum() > tol):
                    print("sxx_split_z is not correct!")
                else:
                    pass
                if (np.abs(mat_syy_split_x - syy_split_x).sum() > tol):
                    print("syy_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_syy_split_y - syy_split_y).sum() > tol):
                    print("syy_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_syy_split_z - syy_split_z).sum() > tol):
                    print("syy_split_z is not correct!")
                else:
                    pass
                if (np.abs(mat_szz_split_x - szz_split_x).sum() > tol):
                    print("szz_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_szz_split_y - szz_split_y).sum() > tol):
                    print("szz_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_szz_split_z - szz_split_z).sum() > tol):
                    print("szz_split_z is not correct!")
                else:
                    pass
                if (np.abs(mat_sxy_split_x - sxy_split_x).sum() > tol):
                    print("sxy_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_sxy_split_y - sxy_split_y).sum() > tol):
                    print("sxy_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_sxz_split_x - sxz_split_x).sum() > tol):
                    print("sxz_split_x is not correct!")
                else:
                    pass
                if (np.abs(mat_sxz_split_z - sxz_split_z).sum() > tol):
                    print("sxz_split_z is not correct!")
                else:
                    pass
                if (np.abs(mat_syz_split_y - syz_split_y).sum() > tol):
                    print("syz_split_y is not correct!")
                else:
                    pass
                if (np.abs(mat_syz_split_z - syz_split_z).sum() > tol):
                    print("syz_split_z is not correct!")
                else:
                    pass

        # compute pressure from the normal components of the stress
        p = -(sxx_split_x + sxx_split_y + sxx_split_z +
              syy_split_x + syy_split_y + syy_split_z +
              szz_split_x + szz_split_y + szz_split_z) / 3.0

        if checking:
            if (t_index == load_index):
                if (np.abs(mat_p - p).sum() > tol):
                    print("p is not correct!")
                else:
                    pass


        # extract required sensor data from the pressure and particle velocity
        # fields if the number of time steps elapsed is greater than
        # sensor.record_start_index (defaults to 1) - why not zero?
        if ((options.use_sensor is not False) and (not options.elastic_time_rev) and (t_index >= k_sim.sensor.record_start_index)):

            # update index for data storage
            file_index: int = t_index - sensor.record_start_index + 1

            # store the acoustic pressure if using a transducer object
            if k_sim.transducer_sensor:
                raise TypeError('Using a kWaveTransducer for output is not currently supported.')

            extract_options = dotdict({'record_u_non_staggered': k_sim.record.u_non_staggered,
                               'record_u_split_field': k_sim.record.u_split_field,
                               'record_I': k_sim.record.I,
                               'record_I_avg': k_sim.record.I_avg,
                               'binary_sensor_mask': k_sim.binary_sensor_mask,
                               'record_p': k_sim.record.p,
                               'record_p_max': k_sim.record.p_max,
                               'record_p_min': k_sim.record.p_min,
                               'record_p_rms': k_sim.record.p_rms,
                               'record_p_max_all': k_sim.record.p_max_all,
                               'record_p_min_all': k_sim.record.p_min_all,
                               'record_u': k_sim.record.u,
                               'record_u_max': k_sim.record.u_max,
                               'record_u_min': k_sim.record.u_min,
                               'record_u_rms': k_sim.record.u_rms,
                               'record_u_max_all': k_sim.record.u_max_all,
                               'record_u_min_all': k_sim.record.u_min_all,
                               'compute_directivity': False
                               })

            # run sub-function to extract the required data
            sensor_data = extract_sensor_data(3, sensor_data, file_index, k_sim.sensor_mask_index,
                                              extract_options, k_sim.record, p, ux_sgx, uy_sgy, uz_sgz)

            # check stream to disk option
            if options.stream_to_disk:
                raise TypeError('"StreamToDisk" input is not currently supported.')

            if checking:
                if (t_index == load_index):
                    if hasattr(sensor_data, 'p_max'):
                        temp = np.squeeze(np.array(mat_sensor_data[0][0].tolist()))
                        temp = np.reshape(temp, np.squeeze(np.asarray(sensor_data.p_max)).shape)
                        if (np.abs(temp - np.squeeze(np.asarray(sensor_data.p_max))).sum() > tol):
                            print("p_max is not correct!")
                        else:
                            pass

        # # estimate the time to run the simulation
        # if t_index == ESTIMATE_SIM_TIME_STEPS:

        #     # display estimated simulation time
        #     print('  estimated simulation time ', scale_time(etime(clock, loop_start_time) * index_end / t_index), '\')

        #     # check memory usage
        #     kspaceFirstOrder_checkMemoryUsage


        # # plot data if required
        # if options.plot_sim and (rem(t_index, plot_freq) == 0 or t_index == 1 or t_index == index_end):

        #     # update progress bar
        #     waitbar(t_index/kgrid.Nt, pbar)
        #     drawnow

        #     # ensure p is cast as a CPU variable and remove the PML from the
        #     # plot if required
        #     if (data_cast == 'gpuArray'):
        #         sii_plot = double(gather(p(x1:x2, y1:y2, z1:z2)))
        #         sij_plot = double(gather((\
        #         sxy_split_x(x1:x2, y1:y2, z1:z2) + sxy_split_y(x1:x2, y1:y2, z1:z2) + \
        #         sxz_split_x(x1:x2, y1:y2, z1:z2) + sxz_split_z(x1:x2, y1:y2, z1:z2) + \
        #         syz_split_y(x1:x2, y1:y2, z1:z2) + syz_split_z(x1:x2, y1:y2, z1:z2) )/3))
        #     else:
        #         sii_plot = double(p(x1:x2, y1:y2, z1:z2))
        #         sij_plot = double((\
        #         sxy_split_x(x1:x2, y1:y2, z1:z2) + sxy_split_y(x1:x2, y1:y2, z1:z2) + \
        #         sxz_split_x(x1:x2, y1:y2, z1:z2) + sxz_split_z(x1:x2, y1:y2, z1:z2) + \
        #         syz_split_y(x1:x2, y1:y2, z1:z2) + syz_split_z(x1:x2, y1:y2, z1:z2) )/3)


        #     # update plot scale if set to automatic or log
        #     if options.plot_scale_auto or options.plot_scale_log:
        #         kspaceFirstOrder_adjustPlotScale

        #     # add display mask onto plot
        #     if (display_mask == 'default'):
        #         sii_plot(sensor.mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(2)
        #         sij_plot(sensor.mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(end)
        #     elif not (display_mask == 'off'):
        #         sii_plot(display_mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(2)
        #         sij_plot(display_mask(x1:x2, y1:y2, z1:z2) != 0) = plot_scale(end)


        #     # update plot
        #     planeplot(scale * kgrid.x_vec(x1:x2),
        #               scale * kgrid.y_vec(y1:y2),
        #               scale * kgrid.z_vec(z1:z2),
        #               sii_plot,
        #               sij_plot, '',
        #               plot_scale,
        #               prefix,
        #               COLOR_MAP)

        #     # save movie frames if required
        #     if options.record_movie:

        #         # set background color to white
        #         set(gcf, 'Color', [1 1 1])

        #         # save the movie frame
        #         writeVideo(video_obj, getframe(gcf))



            # # update variable used for timing variable to exclude the first
            # # time step if plotting is enabled
            # if t_index == 0:
            #     loop_start_time = clock


    # update command line status
    print('\tsimulation completed in', scale_time(TicToc.toc()))

    # =========================================================================
    # CLEAN UP
    # =========================================================================


    # save the final acoustic pressure if required
    if k_sim.record.p_final or options.elastic_time_rev:
        sensor_data.p_final = p[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside]


    # save the final particle velocity if required
    if k_sim.record.u_final:
        sensor_data.ux_final = ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside]
        sensor_data.uy_final = uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside]
        sensor_data.uz_final = uz_sgz[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside, record.z1_inside:record.z2_inside]


    # # run subscript to cast variables back to double precision if required
    # if options.data_recast:
    #     kspaceFirstOrder_dataRecast


    # # run subscript to compute and save intensity values
    # if options.use_sensor and not options.elastic_time_rev and (options.record_I or options.record_I_avg):
    #     save_intensity_matlab_code = True
    #     kspaceFirstOrder_saveIntensity

    # reorder the sensor points if a binary sensor mask was used for Cartesian
    # sensor mask nearest neighbour interpolation (this is performed after
    # recasting as the GPU toolboxes do not all support this subscript)
    if options.use_sensor and k_sim.reorder_data:
        sensor_data = reorder_sensor_data(kgrid, sensor, sensor_data)


    # filter the recorded time domain pressure signals if transducer filter
    # parameters are given
    if options.use_sensor and (not options.elastic_time_rev) and k_sim.sensor.frequency_response is not None:
        sensor_data.p = gaussian_filter(sensor_data.p, 1.0 / kgrid.dt,
                                        k_sim.sensor.frequency_response[0], k_sim.sensor.frequency_response[1])


    # reorder the sensor points if cuboid corners is used (outputs are indexed
    # as [X, Y, Z, T] or [X, Y, Z] rather than [sensor_index, time_index]
    num_stream_time_points: int = k_sim.kgrid.Nt - k_sim.sensor.record_start_index
    if options.cuboid_corners:
        time_info = dotdict({'num_stream_time_points': num_stream_time_points,
                             'num_recorded_time_points': k_sim.num_recorded_time_points,
                             'stream_to_disk': options.stream_to_disk})
        cuboid_info = dotdict({'record_p': k_sim.record.p,
                               'record_p_rms': k_sim.record.p_rms,
                               'record_p_max': k_sim.record.p_max,
                               'record_p_min': k_sim.record.p_min,
                               'record_p_final': k_sim.record.p_final,
                               'record_p_max_all': k_sim.record.p_max_all,
                               'record_p_min_all': k_sim.record.p_min_all,
                               'record_u': k_sim.record.u,
                               'record_u_non_staggered': k_sim.record.u_non_staggered,
                               'record_u_rms': k_sim.record.u_rms,
                               'record_u_max': k_sim.record.u_max,
                               'record_u_min': k_sim.record.u_min,
                               'record_u_final': k_sim.record.u_final,
                               'record_u_max_all': k_sim.record.u_max_all,
                               'record_u_min_all': k_sim.record.u_min_all,
                               'record_I': k_sim.record.I,
                               'record_I_avg': k_sim.record.I_avg})
        sensor_data = reorder_cuboid_corners(k_sim.kgrid, k_sim.record, sensor_data, time_info, cuboid_info, verbose=True)

    if options.elastic_time_rev:
        # if computing time reversal, reassign sensor_data.p_final to sensor_data
        sensor_data = sensor_data.p_final
        raise NotImplementedError("elastic_time_rev is not implemented")
    elif not options.use_sensor:
        # if sensor is not used, return empty sensor data
        print("not options.use_sensor: returns None ->", options.use_sensor)
        sensor_data = None
    elif (sensor.record is None) and (not options.cuboid_corners):
        # if sensor.record is not given by the user, reassign sensor_data.p to sensor_data
        print("reassigns. Not sure if there is a check for whether this exists though")
        sensor_data = sensor_data.p
    else:
        pass

    # update command line status
    print('\ttotal computation time', scale_time(TicToc.toc()))

    # # switch off log
    # if options.create_log:
    #     diary off

    return sensor_data


# def planeplot(x_vec, y_vec, z_vec, s_normal_plot, s_shear_plot, data_title, plot_scale, prefix, color_map):
#     """
#     Subfunction to produce a plot of the elastic wavefield
#     """

#     # plot normal stress
#     subplot(2, 3, 1)
#     imagesc(y_vec, x_vec, squeeze(s_normal_plot(:, :, round(end/2))), plot_scale(1:2))
#     title('Normal Stress (x-y plane)'), axis image

#     subplot(2, 3, 2)
#     imagesc(z_vec, x_vec, squeeze(s_normal_plot(:, round(end/2), :)), plot_scale(1:2))
#     title('Normal Stress  (x-z plane)'), axis image

#     subplot(2, 3, 3)
#     imagesc(z_vec, y_vec, squeeze(s_normal_plot(round(end/2), :, :)), plot_scale(1:2))
#     title('Normal Stress  (y-z plane)'), axis image

#     # plot shear stress
#     subplot(2, 3, 4)
#     imagesc(y_vec, x_vec, squeeze(s_shear_plot(:, :, round(end/2))), plot_scale(end - 1:end))
#     title('Shear Stress (x-y plane)'), axis image

#     subplot(2, 3, 5)
#     imagesc(z_vec, x_vec, squeeze(s_shear_plot(:, round(end/2), :)), plot_scale(end - 1:end))
#     title('Shear Stress (x-z plane)'), axis image

#     subplot(2, 3, 6)
#     imagesc(z_vec, y_vec, squeeze(s_shear_plot(round(end/2), :, :)), plot_scale(end - 1:end))
#     title('Shear Stress (y-z plane)'), axis image

#     xlabel(['(All axes in ' prefix 'm)'])
#     colormap(color_map)
#     drawnow