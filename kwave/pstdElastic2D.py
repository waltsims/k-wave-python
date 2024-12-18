import numpy as np
from scipy.interpolate import interpn
import scipy.fft
from tqdm import tqdm
from typing import Union
from copy import deepcopy

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

from kwave.kWaveSimulation_helper import extract_sensor_data

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def pstd_elastic_2d(kgrid: kWaveGrid,
                    source: kSource,
                    sensor: Union[NotATransducer, kSensor],
                    medium: kWaveMedium,
                    simulation_options: SimulationOptions,
                    verbose: bool = False):
    """
    2D time-domain simulation of elastic wave propagation.

    DESCRIPTION:
        pstd_elastic_2d simulates the time-domain propagation of elastic waves
        through a two-dimensional homogeneous or heterogeneous medium given
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
        assigned to source.sxx, source.syy, and source.sxy. These can be a
        single time series (in which case it is applied to all source
        elements), or a matrix of time series following the source elements
        using MATLAB's standard column-wise linear matrix index ordering. A
        time varying velocity source can be specified in an analogous
        fashion, where the source location is specified by source.u_mask, and
        the time varying input velocity is assigned to source.ux and
        source.uy.

        The field values are returned as arrays of time series at the sensor
        locations defined by sensor.mask. This can be defined in three
        different ways. (1) As a binary matrix (i.e., a matrix of 1's and 0's
        with the same dimensions as the computational grid) representing the
        grid points within the computational grid that will collect the data.
        (2) As the grid coordinates of two opposing corners of a rectangle in
        the form [x1; y1; x2; y2]. This is equivalent to using a binary
        sensor mask covering the same region, however, the output is indexed
        differently as discussed below. (3) As a series of Cartesian
        coordinates within the grid which specify the location of the
        pressure values stored at each time step. If the Cartesian
        coordinates don't exactly match the coordinates of a grid point, the
        output values are calculated via interpolation. The Cartesian points
        must be given as a 2 by N matrix corresponding to the x and y
        positions, respectively, where the Cartesian origin is assumed to be
        in the center of the grid. If no output is required, the sensor input
        can be replaced with an empty array [].

        If sensor.mask is given as a set of Cartesian coordinates, the
        computed sensor_data is returned in the same order. If sensor.mask is
        given as a binary matrix, sensor_data is returned using MATLAB's
        standard column-wise linear matrix index ordering. In both cases, the
        recorded data is indexed as sensor_data(sensor_point_index,
        time_index). For a binary sensor mask, the field values at a
        particular time can be restored to the sensor positions within the
        computation grid using unmaskSensorData. If sensor.mask is given as a
        list of opposing corners of a rectangle, the recorded data is indexed
        as sensor_data(rect_index).p(x_index, y_index, time_index), where
        x_index and y_index correspond to the grid index within the
        rectangle, and rect_index corresponds to the number of rectangles if
        more than one is specified.

        By default, the recorded acoustic pressure field is passed directly
        to the output sensor_data. However, other acoustic parameters can
        also be recorded by setting sensor.record to a cell array of the form
        {'p', 'u', 'p_max', ...}. For example, both the particle velocity and
        the acoustic pressure can be returned by setting sensor.record =
        {'p', 'u'}. If sensor.record is given, the output sensor_data is
        returned as a structure with the different outputs appended as
        structure fields. For example, if sensor.record = {'p', 'p_final',
        'p_max', 'u'}, the output would contain fields sensor_data.p,
        sensor_data.p_final, sensor_data.p_max, sensor_data.ux, and
        sensor_data.uy. Most of the output parameters are recorded at the
        given sensor positions and are indexed as
        sensor_data.field(sensor_point_index, time_index) or
        sensor_data(rect_index).field(x_index, y_index, time_index) if using
        a sensor mask defined as opposing rectangular corners. The exceptions
        are the averaged quantities ('p_max', 'p_rms', 'u_max', 'p_rms',
        'I_avg'), the 'all' quantities ('p_max_all', 'p_min_all',
        'u_max_all', 'u_min_all'), and the final quantities ('p_final',
        'u_final'). The averaged quantities are indexed as
        sensor_data.p_max(sensor_point_index) or
        sensor_data(rect_index).p_max(x_index, y_index) if using rectangular
        corners, while the final and 'all' quantities are returned over the
        entire grid and are always indexed as sensor_data.p_final(nx, ny),
        regardless of the type of sensor mask.

        pstd_elastic_2d may also be used for time reversal image reconstruction
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
        sensor_data = pstd_elastic_2d(kWaveGrid, kWaveMedium, kSource, kSensor)


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
        source.sxy             - time varying stress at each of the source
                                  positions given by source.s_mask
        source.s_mask          - binary matrix specifying the positions of
                                  the time varying stress source distributions
        source.s_mode          - optional input to control whether the input
                                  stress is injected as a mass source or
                                  enforced as a dirichlet boundary condition;
                                  valid inputs are 'additive' (the default) or
                                  'dirichlet'
        source.ux              - time varying particle velocity in the
                                  x-direction at each of the source positions
                                  given by source.u_mask
        source.uy              - time varying particle velocity in the
                                  y-direction at each of the source positions
                                  given by source.u_mask
        source.u_mask          - binary matrix specifying the positions of
                                  the time varying particle velocity
                                  distribution
        source.u_mode          - optional input to control whether the input
                                  velocity is applied as a force source or
                                  enforced as a dirichlet boundary condition;
                                  valid inputs are 'additive' (the default) or
                                  'dirichlet'

        sensor.mask*           - binary matrix or a set of Cartesian points
                                  where the pressure is recorded at each
                                  time-step
        sensor.record          - cell array of the acoustic parameters to
                                  record in the form sensor.record = {'p',
                                  'u', ...}; valid inputs are:

            - 'p' (acoustic pressure)
            - 'p_max' (maximum pressure)
            - 'p_min' (minimum pressure)
            - 'p_rms' (RMS pressure)
            - 'p_final' (final pressure field at all grid points)
            - 'p_max_all' (maximum pressure at all grid points)
            - 'p_min_all' (minimum pressure at all grid points)
            - 'u' (particle velocity)
            - 'u_max' (maximum particle velocity)
            - 'u_min' (minimum particle velocity)
            - 'u_rms' (RMS particle21st January 2014 velocity)
            - 'u_final' (final particle velocity field at all grid points)
            - 'u_max_all' (maximum particle velocity at all grid points)
            - 'u_min_all' (minimum particle velocity at all grid points)
            - 'u_non_staggered' (particle velocity on non-staggered grid)
            - 'u_split_field' (particle velocity on non-staggered grid split
                              into compressional and shear components)
            - 'I' (time varying acoustic intensity)
            - 'I_avg' (average acoustic intensity)

            NOTE: the acoustic pressure outputs are calculated from the
            normal stress via: p = -(sxx + syy) / 2

        sensor.record_start_index
                                - time index at which the sensor should start
                                  recording the data specified by
                                  sensor.record (default = 1, but shifted to 0)
        sensor.time_reversal_boundary_data
                                - time varying pressure enforced as a
                                  Dirichlet boundary condition over sensor.mask

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
        sensor_data.p          - time varying pressure recorded at the sensor
                                  positions given by sensor.mask (returned if
                                  'p' is set)
        sensor_data.p_max      - maximum pressure recorded at the sensor
                                  positions given by sensor.mask (returned if
                                  'p_max' is set)
        sensor_data.p_min      - minimum pressure recorded at the sensor
                                  positions given by sensor.mask (returned if
                                  'p_min' is set)
        sensor_data.p_rms      - rms of the time varying pressure recorded at
                                  the sensor positions given by sensor.mask
                                  (returned if 'p_rms' is set)
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
        sensor_data.ux_max     - maximum particle velocity in the x-direction
                                  recorded at the sensor positions given by
                                  sensor.mask (returned if 'u_max' is set)
        sensor_data.uy_max     - maximum particle velocity in the y-direction
                                  recorded at the sensor positions given by
                                  sensor.mask (returned if 'u_max' is set)
        sensor_data.ux_min     - minimum particle velocity in the x-direction
                                  recorded at the sensor positions given by
                                  sensor.mask (returned if 'u_min' is set)
        sensor_data.uy_min     - minimum particle velocity in the y-direction
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
        sensor_data.ux_final   - final particle velocity field in the
                                  x-direction at all grid points within the
                                  domain (returned if 'u_final' is set)
        sensor_data.uy_final   - final particle velocity field in the
                                  y-direction at all grid points within the
                                  domain (returned if 'u_final' is set)
        sensor_data.ux_max_all - maximum particle velocity in the x-direction
                                  recorded at all grid points within the
                                  domain (returned if 'u_max_all' is set)
        sensor_data.uy_max_all - maximum particle velocity in the y-direction
                                  recorded at all grid points within the
                                  domain (returned if 'u_max_all' is set)
        sensor_data.ux_min_all - minimum particle velocity in the x-direction
                                  recorded at all grid points within the
                                  domain (returned if 'u_min_all' is set)
        sensor_data.uy_min_all - minimum particle velocity in the y-direction
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
        sensor_data.Ix         - time varying acoustic intensity in the
                                  x-direction recorded at the sensor positions
                                  given by sensor.mask (returned if 'I' is
                                  set)
        sensor_data.Iy         - time varying acoustic intensity in the
                                  y-direction recorded at the sensor positions
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

    ABOUT:
        author                 - Bradley Treeby & Ben Cox
        date                   - 11th March 2013
        last update            - 13th January 2019

    This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    Copyright (C) 2013-2019 Bradley Treeby and Ben Cox

    See also kspaceFirstOrder2D, kWaveGrid, pstdElastic3D

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

    # =========================================================================
    # CHECK INPUT STRUCTURES AND OPTIONAL INPUTS
    # =========================================================================

    # start the timer and store the start time
    timer = TicToc()
    timer.tic()

    # run script to check inputs and create the required arrays
    k_sim = kWaveSimulation(kgrid=kgrid, source=source, sensor=sensor, medium=medium,
                            simulation_options=simulation_options)

    # this will create the sensor_data dotdict
    k_sim.input_checking("pstd_elastic_2d")

    sensor_data = k_sim.sensor_data
    options = k_sim.options

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================

    rho0 = np.atleast_1d(k_sim.rho0)

    m_rho0: int = np.squeeze(k_sim.rho0).ndim

    # assign the lame parameters
    mu = medium.density * np.power(medium.sound_speed_shear, 2)
    lame_lambda = medium.density * medium.sound_speed_compression**2 - 2.0 * mu
    m_mu: int = np.squeeze(mu).ndim

    # assign the viscosity coefficients
    if options.kelvin_voigt_model:
        eta = 2.0 * rho0 * medium.sound_speed_shear**3 * db2neper(deepcopy(medium.alpha_coeff_shear), 2.0)
        chi = 2.0 * rho0 * medium.sound_speed_compression**3 * db2neper(deepcopy(medium.alpha_coeff_compression), 2.0) - 2.0 * eta
        m_eta : int = np.squeeze(eta).ndim

    # calculate the values of the density at the staggered grid points
    # using the arithmetic average [1, 2], where sgx  = (x + dx/2, y) and
    # sgy  = (x, y + dy/2)

    points = (np.squeeze(k_sim.kgrid.x_vec), np.squeeze(k_sim.kgrid.y_vec))

    if (m_rho0 == 2 and options.use_sg):

        # rho0 is heterogeneous and staggered grids are used
        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
                         np.squeeze(k_sim.kgrid.y_vec),
                         indexing='ij',)
        interp_points = np.moveaxis(mg, 0, -1)

        rho0_sgx = interpn(points, k_sim.rho0, interp_points, method='linear', bounds_error=False)

        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec),
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2,
                         indexing='ij',)

        interp_points = np.moveaxis(mg, 0, -1)
        rho0_sgy = interpn(points, k_sim.rho0, interp_points, method='linear', bounds_error=False)

        rho0_sgx[np.isnan(rho0_sgx)] = k_sim.rho0[np.isnan(rho0_sgx)]
        rho0_sgy[np.isnan(rho0_sgy)] = k_sim.rho0[np.isnan(rho0_sgy)]

    else:

        # rho0 is homogeneous or staggered grids are not used
        rho0_sgx = k_sim.rho0
        rho0_sgy = k_sim.rho0


    # invert rho0 so it doesn't have to be done each time step
    rho0_sgx_inv = 1.0 / rho0_sgx
    rho0_sgy_inv = 1.0 / rho0_sgy

    # clear unused variables
    del rho0_sgx
    del rho0_sgy


    # calculate the values of mu at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2)
    if (m_mu == 2 and options.use_sg):

        # mu is heterogeneous and staggered grids are used
        mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
                         np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2,
                         indexing='ij',)

        interp_points = np.moveaxis(mg, 0, -1)

        with np.errstate(divide='ignore', invalid='ignore'):
            mu_sgxy = 1.0 / interpn(points, 1.0 / mu, interp_points, method='linear', bounds_error=False)

        # set values outside of the interpolation range to original values
        mu_sgxy[np.isnan(mu_sgxy)] = mu[np.isnan(mu_sgxy)]

    else:
        # mu is homogeneous or staggered grids are not used
        mu_sgxy = mu


    # calculate the values of eta at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2)
    if options.kelvin_voigt_model:
        if (m_eta == 2 and options.use_sg):

            # eta is heterogeneous and staggered grids are used
            mg = np.meshgrid(np.squeeze(k_sim.kgrid.x_vec) + k_sim.kgrid.dx / 2,
                             np.squeeze(k_sim.kgrid.y_vec) + k_sim.kgrid.dy / 2,
                             indexing ='ij')

            interp_points = np.moveaxis(mg, 0, -1)

            with np.errstate(divide='ignore', invalid='ignore'):
                eta_sgxy = 1.0 / interpn(points, 1.0 / eta, interp_points, method='linear', bounds_error=False)

            # set values outside of the interpolation range to original values
            eta_sgxy[np.isnan(eta_sgxy)] = eta[np.isnan(eta_sgxy)]

        else:

            # eta is homogeneous or staggered grids are not used
            eta_sgxy = eta

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
    Nx, Ny = k_sim.kgrid.Nx, k_sim.kgrid.Ny
    dx, dy = k_sim.kgrid.dx, k_sim.kgrid.dy
    dt = k_sim.kgrid.dt
    Nt = k_sim.kgrid.Nt

    pml_x_alpha, pml_y_alpha = options.pml_x_alpha, options.pml_y_alpha
    pml_x_size, pml_y_size = options.pml_x_size, options.pml_y_size
    c_ref = k_sim.c_ref

    multi_axial_PML_ratio = options.multi_axial_PML_ratio

    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0)
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True,  0)
    pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 1)
    pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True,  1)

    # get the multi-axial PML operators
    mpml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, False, 0)
    mpml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, True,  0)
    mpml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, False, 1)
    mpml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, True,  1)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using ifftshift (the option
    # options.use_sg exists for debugging)
    kx_vec = np.squeeze(k_sim.kgrid.k_vec[0])
    ky_vec = np.squeeze(k_sim.kgrid.k_vec[1])

    if options.use_sg:
        ddx_k_shift_pos = scipy.fft.ifftshift(1j * kx_vec * np.exp(1j * kx_vec * dx / 2.0))
        ddy_k_shift_pos = scipy.fft.ifftshift(1j * ky_vec * np.exp(1j * ky_vec * dy / 2.0))
        ddx_k_shift_neg = scipy.fft.ifftshift(1j * kx_vec * np.exp(-1j * kx_vec * dx / 2.0))
        ddy_k_shift_neg = scipy.fft.ifftshift(1j * ky_vec * np.exp(-1j * ky_vec * dy / 2.0))
    else:
        ddx_k_shift_pos = scipy.fft.ifftshift(1j * kx_vec)
        ddx_k_shift_neg = scipy.fft.ifftshift(1j * kx_vec)
        ddy_k_shift_pos = scipy.fft.ifftshift(1j * ky_vec)
        ddy_k_shift_neg = scipy.fft.ifftshift(1j * ky_vec)

    # shape for broadcasting
    ddx_k_shift_pos = np.expand_dims(ddx_k_shift_pos, axis=1)
    ddx_k_shift_neg = np.expand_dims(ddx_k_shift_neg, axis=1)
    ddy_k_shift_pos = np.expand_dims(np.squeeze(ddy_k_shift_pos), axis=0)
    ddy_k_shift_neg = np.expand_dims(np.squeeze(ddy_k_shift_neg), axis=0)

    # =========================================================================
    # DATA CASTING
    # =========================================================================

    # run subscript to cast the remaining loop variables to the data type
    # specified by data_cast
    if not (options.data_cast == 'off'):
        myType = np.single
    else:
        myType = np.double

    grid_shape = (Nx, Ny)

    # preallocate the loop variables
    ux_split_x = np.zeros((Nx, Ny), dtype=myType)
    ux_split_y = np.zeros((Nx, Ny), dtype=myType)
    uy_split_x = np.zeros((Nx, Ny), dtype=myType)
    uy_split_y = np.zeros((Nx, Ny), dtype=myType)

    ux_sgx = np.zeros((Nx, Ny), dtype=myType)  # **
    uy_sgy = np.zeros((Nx, Ny), dtype=myType)  # **

    sxx_split_x = np.zeros((Nx, Ny), dtype=myType)
    sxx_split_y = np.zeros((Nx, Ny), dtype=myType)
    syy_split_x = np.zeros((Nx, Ny), dtype=myType)
    syy_split_y = np.zeros((Nx, Ny), dtype=myType)
    sxy_split_x = np.zeros((Nx, Ny), dtype=myType)
    sxy_split_y = np.zeros((Nx, Ny), dtype=myType)

    duxdx = np.zeros((Nx, Ny), dtype=myType)  # **
    duxdy = np.zeros((Nx, Ny), dtype=myType)  # **
    duydy = np.zeros((Nx, Ny), dtype=myType)  # **
    duydx = np.zeros((Nx, Ny), dtype=myType)  # **

    dsxxdx = np.zeros((Nx, Ny), dtype=myType)  # **
    dsxydy = np.zeros((Nx, Ny), dtype=myType)  # **
    dsxydx = np.zeros((Nx, Ny), dtype=myType)  # **
    dsyydy = np.zeros((Nx, Ny), dtype=myType)  # **

    p = np.zeros((Nx, Ny), dtype=myType)  # **

    if not (options.data_cast == 'off'):
        two = np.float32(2.0)
        dt = np.float32(dt)
    else:
        two = np.float64(2.0)
        dt = np.float64(dt)

    if m_mu == 2:
        mu = mu.astype(myType)
        lame_lambda = lame_lambda.astype(myType)
    else:
        if not (options.data_cast == 'off'):
            mu = np.float32(mu)
            lame_lambda = np.float32(lame_lambda)
        else:
            mu = np.float64(mu)
            lame_lambda = np.float64(lame_lambda)

    if options.kelvin_voigt_model:
        dduxdxdt = np.zeros(grid_shape, dtype=myType)  # **
        dduydydt = np.zeros(grid_shape, dtype=myType)  # **
        dduxdydt = np.zeros(grid_shape, dtype=myType)  # **
        dduydxdt = np.zeros(grid_shape, dtype=myType)  # **
        if m_eta == 2:
            eta = eta.astype(myType)
            chi = chi.astype(myType)
        else:
            if not (options.data_cast == 'off'):
                eta = np.float32(eta)
                chi = np.float32(chi)
            else:
                eta = np.float64(eta)
                chi = np.float64(chi)



    # to save memory, the variables noted with a ** do not neccesarily need to
    # be explicitly stored (they are not needed for update steps). Instead they
    # could be replaced with a small number of temporary variables that are
    # reused several times during the time loop.


    # =========================================================================
    # CREATE INDEX VARIABLES
    # =========================================================================

    # setup the time index variable
    if (not options.time_rev):
        index_start: int = 0
        index_step: int = 1
        index_end: int = Nt
    else:
        # throw error for unsupported feature
        raise TypeError('Time reversal using sensor.time_reversal_boundary_data is not currently supported.')


    # =========================================================================
    # PREPARE VISUALISATIONS
    # =========================================================================

    # pre-compute suitable axes scaling factor
    # if (options.plot_layout or options.plot_sim):
    #     (x_sc, scale, prefix) = scale_SI(np.max([k_sim.kgrid.x_vec, k_sim.kgrid.y_vec]))

    # throw error for currently unsupported plot layout feature
    # if options.plot_layout:
    #     raise TypeError('PlotLayout input is not currently supported.')

    # initialise the figure used for animation if 'PlotSim' is set to 'True'
    # if options.plot_sim:
    #     kspaceFirstOrder_initialiseFigureWindow;

    # initialise movie parameters if 'RecordMovie' is set to 'True'
    # if options.record_movie:
    #     kspaceFirstOrder_initialiseMovieParameters;


    # =========================================================================
    # LOOP THROUGH TIME STEPS
    # =========================================================================

    # update command line status
    t0 = timer.toc()
    t0_scale = scale_time(t0)
    print('\tprecomputation completed in', t0_scale)
    print('\tstarting time loop...')

    # end at this point - but nothing is saved to disk.
    if options.save_to_disk_exit:
        return

    # consistent sizing for broadcasting
    pml_x_sgx = np.transpose(pml_x_sgx)

    pml_y_sgy = np.squeeze(pml_y_sgy)
    pml_y_sgy = np.expand_dims(pml_y_sgy, axis=0)

    mpml_x = np.transpose(mpml_x)

    mpml_y = np.squeeze(mpml_y)
    mpml_y = np.expand_dims(mpml_y, axis=0)

    mpml_x_sgx = np.transpose(mpml_x_sgx)

    mpml_y_sgy = np.squeeze(mpml_y_sgy)
    mpml_y_sgy = np.expand_dims(mpml_y_sgy, axis=0)

    pml_x = np.transpose(pml_x)

    pml_y = np.squeeze(pml_y)
    pml_y = np.expand_dims(pml_y, axis=0)


    # These should be zero indexed
    if hasattr(k_sim, 's_source_sig_index') and k_sim.s_source_pos_index is not None:
        k_sim.s_source_pos_index = np.squeeze(k_sim.s_source_pos_index) - int(1)

    if hasattr(k_sim, 'u_source_pos_index') and k_sim.u_source_pos_index is not None:
        k_sim.u_source_pos_index = np.squeeze(k_sim.u_source_pos_index) - int(1)

    if hasattr(k_sim, 'p_source_pos_index') and k_sim.p_source_pos_index is not None:
        k_sim.p_source_pos_index = np.squeeze(k_sim.p_source_pos_index) - int(1)

    if hasattr(k_sim, 's_source_sig_index') and k_sim.s_source_sig_index is not None:
        k_sim.s_source_sig_index = np.squeeze(k_sim.s_source_sig_index) - int(1)

    if hasattr(k_sim, 'u_source_sig_index') and k_sim.u_source_sig_index is not None:
        k_sim.u_source_sig_index = np.squeeze(k_sim.u_source_sig_index) - int(1)

    if hasattr(k_sim, 'p_source_sig_index') and k_sim.p_source_sig_index is not None:
        k_sim.p_source_sig_index = np.squeeze(k_sim.p_source_sig_index) - int(1)

    # These should be zero indexed
    record.x1_inside = int(record.x1_inside - 1)
    record.y1_inside = int(record.y1_inside - 1)

    sensor.record_start_index = sensor.record_start_index - int(1)

    # start time loop
    for t_index in tqdm(np.arange(index_start, index_end, index_step, dtype=int)):

        # compute the gradients of the stress tensor
        # these variables do not necessaily need to be stored, they could be computed as needed
        dsxxdx = np.real(scipy.fft.ifftn(ddx_k_shift_pos * scipy.fft.fftn(sxx_split_x + sxx_split_y, axes=(0,) ), axes=(0,) ))
        dsyydy = np.real(scipy.fft.ifftn(ddy_k_shift_pos * scipy.fft.fftn(syy_split_x + syy_split_y, axes=(1,) ), axes=(1,) ))
        temp = sxy_split_x + sxy_split_y
        dsxydx = np.real(scipy.fft.ifftn(ddx_k_shift_neg * scipy.fft.fftn(temp, axes=(0,) ), axes=(0,) ))
        dsxydy = np.real(scipy.fft.ifftn(ddy_k_shift_neg * scipy.fft.fftn(temp, axes=(1,) ), axes=(1,) ))

        # calculate the split-field components of ux_sgx and uy_sgy at the next
        # time step using the components of the stress at the current time step
        temp =  mpml_y * pml_x_sgx
        ux_split_x = temp * (temp * ux_split_x + dt * rho0_sgx_inv * dsxxdx)

        temp =  mpml_x_sgx * pml_y
        ux_split_y = temp * (temp * ux_split_y + dt * rho0_sgx_inv * dsxydy)

        temp =  mpml_y_sgy * pml_x
        uy_split_x = temp * (temp * uy_split_x + dt * rho0_sgy_inv * dsxydx)

        temp =  mpml_x * pml_y_sgy
        uy_split_y = temp * (temp * uy_split_y + dt * rho0_sgy_inv * dsyydy)

        # add in the pre-scaled velocity source terms
        if (k_sim.source_ux > t_index):
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] = k_sim.source.ux[k_sim.u_source_sig_index, t_index]
            else:
                # add the source values to the existing field values
                ux_split_x[np.unravel_index(k_sim.u_source_pos_index, ux_split_x.shape, order='F')] += k_sim.source.ux[k_sim.u_source_sig_index, t_index]

        if (k_sim.source_uy > t_index):
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] = k_sim.source.uy[k_sim.u_source_sig_index, t_index]
            else:
                # add the source values to the existing field values
                uy_split_y[np.unravel_index(k_sim.u_source_pos_index, uy_split_y.shape, order='F')] += k_sim.source.uy[k_sim.u_source_sig_index, t_index]


        # combine split field components
        # these variables do not necessarily need to be stored, they could be computed when needed
        ux_sgx = ux_split_x + ux_split_y
        uy_sgy = uy_split_x + uy_split_y

        # calculate the velocity gradients
        # these variables do not necessarily need to be stored, they could be computed when needed
        duxdx = np.real(scipy.fft.ifftn(ddx_k_shift_neg * scipy.fft.fftn(ux_sgx, axes=(0,) ), axes=(0,) ))
        duxdy = np.real(scipy.fft.ifftn(ddy_k_shift_pos * scipy.fft.fftn(ux_sgx, axes=(1,) ), axes=(1,) ))
        duydx = np.real(scipy.fft.ifftn(ddx_k_shift_pos * scipy.fft.fftn(uy_sgy, axes=(0,) ), axes=(0,) ))
        duydy = np.real(scipy.fft.ifftn(ddy_k_shift_neg * scipy.fft.fftn(uy_sgy, axes=(1,) ), axes=(1,) ))

        # update the normal components and shear components of stress tensor using a split field pml
        if options.kelvin_voigt_model:

            # compute additional gradient terms needed for the Kelvin-Voigt model

            temp = (dsxxdx + dsxydy) * rho0_sgx_inv
            dduxdxdt = np.real(scipy.fft.ifftn(ddx_k_shift_neg * scipy.fft.fftn(temp, axes=(0,) ), axes=(0,) ))
            dduxdydt = np.real(scipy.fft.ifftn(ddy_k_shift_pos * scipy.fft.fftn(temp, axes=(1,) ), axes=(1,) ))

            temp = (dsyydy + dsxydx) * rho0_sgy_inv
            dduydxdt = np.real(scipy.fft.ifftn(ddx_k_shift_pos * scipy.fft.fftn(temp, axes=(0,) ), axes=(0,) ))
            dduydydt = np.real(scipy.fft.ifftn(ddy_k_shift_neg * scipy.fft.fftn(temp, axes=(1,) ), axes=(1,) ))

            temp = mpml_y * pml_x
            temp1 = dt * (lame_lambda * duxdx + chi * dduxdxdt)
            temp2 = two * dt * (mu * duxdx + eta * dduxdxdt)

            sxx_split_x = temp * (temp * sxx_split_x + temp1 + temp2)
            syy_split_x = temp * (temp * syy_split_x + temp1)


            temp = mpml_x * pml_y
            temp1 = dt * (lame_lambda * duydy + chi * dduydydt)
            temp2 = two * dt * (mu * duydy + eta * dduydydt)

            sxx_split_y = temp * (temp * sxx_split_y + temp1)
            syy_split_y = temp * (temp * syy_split_y + temp1 + temp2)


            temp = mpml_y_sgy * pml_x_sgx
            sxy_split_x = temp * (temp * sxy_split_x + dt * (mu_sgxy * duydx + eta_sgxy * dduydxdt))

            temp = mpml_x_sgx * pml_y_sgy
            sxy_split_y = temp * (temp * sxy_split_y + dt * (mu_sgxy * duxdy + eta_sgxy * dduxdydt))

        else:
            # update the normal and shear components of the stress tensor using
            # a lossless elastic model with a split-field multi-axial pml

            temp = mpml_y * pml_x
            temp1 = dt * lame_lambda * duxdx
            temp2 = dt * two * mu * duxdx

            sxx_split_x = temp * (temp * sxx_split_x + temp1 + temp2)
            syy_split_x = temp * (temp * syy_split_x + temp1)


            temp = mpml_x * pml_y
            temp1 = dt * lame_lambda * duydy
            temp2 = dt * two * mu * duydy

            sxx_split_y = temp * (temp * sxx_split_y + temp1)
            syy_split_y = temp * (temp * syy_split_y + temp1 + temp2)


            temp = mpml_y_sgy * pml_x_sgx
            sxy_split_x = temp * (temp * sxy_split_x + dt * mu_sgxy * duydx)


            temp = mpml_x_sgx * pml_y_sgy
            sxy_split_y = temp * (temp * sxy_split_y + dt * mu_sgxy * duxdy)


        # add stress source terms
        if (k_sim.source_sxx is not False and t_index < np.shape(source.sxx)[1]):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] = k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] = k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
            else:
                # spatially and temporally varying source
                sxx_split_x[np.unravel_index(k_sim.s_source_pos_index, sxx_split_x.shape, order='F')] += k_sim.source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_y[np.unravel_index(k_sim.s_source_pos_index, sxx_split_y.shape, order='F')] += k_sim.source.sxx[k_sim.s_source_sig_index, t_index]

        if (k_sim.source_syy is not False and t_index < np.shape(source.syy)[1]):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] = k_sim.source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] = k_sim.source.syy[k_sim.s_source_sig_index, t_index]
            else:
                # spatially and temporally varying source
                syy_split_x[np.unravel_index(k_sim.s_source_pos_index, syy_split_x.shape, order='F')] += k_sim.source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_y[np.unravel_index(k_sim.s_source_pos_index, syy_split_y.shape, order='F')] += k_sim.source.syy[k_sim.s_source_sig_index, t_index]

        if (k_sim.source_sxy is not False and t_index < k_sim.source_sxy):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] = k_sim.source.sxy[k_sim.s_source_sig_index, t_index]
                sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] = k_sim.source.sxy[k_sim.s_source_sig_index, t_index]
            else:
                # spatially and temporally varying source
                sxy_split_x[np.unravel_index(k_sim.s_source_pos_index, sxy_split_x.shape, order='F')] += k_sim.source.sxy[k_sim.s_source_sig_index, t_index]
                sxy_split_y[np.unravel_index(k_sim.s_source_pos_index, sxy_split_y.shape, order='F')] += k_sim.source.sxy[k_sim.s_source_sig_index, t_index]


        # compute pressure from normal components of the stress
        p = -(sxx_split_x + sxx_split_y + syy_split_x + syy_split_y) / two

        # extract required sensor data from the pressure and particle velocity
        # fields if the number of time steps elapsed is greater than
        # sensor.record_start_index (defaults to 1)
        if ((k_sim.use_sensor is not False) and (not k_sim.elastic_time_rev) and (t_index >= sensor.record_start_index)):

            # update index for data storage
            file_index: int = t_index - sensor.record_start_index

            # run sub-function to extract the required data
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
                                       'compute_directivity': False})

            sensor_data = extract_sensor_data(2, sensor_data, file_index, k_sim.sensor_mask_index,
                                              extract_options, k_sim.record, p, ux_sgx, uy_sgy)

            # update variable used for timing variable to exclude the first
            # time step if plotting is enabled
            if t_index == 0:
                clock1 = TicToc()
                clock1.tic()


    # update command line status
    t1 = timer.toc()
    t1_scale = scale_time(t1)
    print('\tsimulation completed in', t1_scale)

    # =========================================================================
    # CLEAN UP
    # =========================================================================

    # # clean up used figures
    # if options.plot_sim:
    #     # close(img);
    #     # close(pbar);
    #     pass

    # # save the movie frames to disk
    # if options.record_movie:
    #     # close(video_obj);
    #     pass

    # save the final acoustic pressure if required
    if (k_sim.record.p_final or k_sim.elastic_time_rev):
        sensor_data.p_final = p[record.x1_inside:record.x2_inside,
                                record.y1_inside:record.y2_inside]

    # save the final particle velocity if required
    if k_sim.record.u_final:
        sensor_data.ux_final = ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
        sensor_data.uy_final = uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]

    # # run subscript to cast variables back to double precision if required
    # if options.data_recast:
    #     #kspaceFirstOrder_dataRecast;
    #     pass

    # run subscript to compute and save intensity values
    if (k_sim.use_sensor is not False and (not k_sim.elastic_time_rev) and (k_sim.record.I or k_sim.record.I_avg)):
        # save_intensity_matlab_code = True
        # kspaceFirstOrder_saveIntensity;
        pass

    # reorder the sensor points if a binary sensor mask was used for Cartesian
    # sensor mask nearest neighbour interpolation (this is performed after
    # recasting as the GPU toolboxes do not all support this subscript)
    if (k_sim.use_sensor is not False and k_sim.reorder_data):
        # kspaceFirstOrder_reorderCartData;
        pass

    # filter the recorded time domain pressure signals if transducer filter
    # parameters are given
    if (k_sim.use_sensor is not False and not k_sim.elastic_time_rev and hasattr(sensor, 'frequency_response') and
        sensor.frequency_response is not None):
        fs = 1.0 / kgrid.dt
        sensor_data.p = gaussian_filter(sensor_data.p, fs, sensor.frequency_response[0], sensor.frequency_response[1])

    # reorder the sensor points if cuboid corners is used (outputs are indexed
    # as [X, Y, T] or [X, Y] rather than [sensor_index, time_index]
    if options.cuboid_corners:
        sensor_data = reorder_sensor_data(kgrid, sensor, sensor_data)

    if k_sim.elastic_time_rev:
        # if computing time reversal, reassign sensor_data.p_final to sensor_data
        sensor_data = sensor_data.p_final
    elif (k_sim.use_sensor is False):
        # if sensor is not used, return empty sensor data
        sensor_data = None
    elif ((not hasattr(sensor, 'record')) and (not options.cuboid_corners)):
        # if sensor.record is not given by the user, reassign sensor_data.p to sensor_data
        sensor_data = sensor_data.p

    # update command line status
    t_total = t0 + t1
    print('\ttotal computation time', scale_time(t_total), '\n')

    return sensor_data
