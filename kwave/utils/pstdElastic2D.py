import numpy as np
from scipy.interpolate import interpn

from kwave.kWaveSimulation import kWaveSimulation

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_time, scale_SI
from kwave.utils.filters import gaussian_filter
from kwave.utils.pml import get_pml
from kwave.utils.signals import reorder_sensor_data
from kwave.utils.tictoc import TicToc

from kwave.options.simulation_options import SimulationOptions

from kwave.kWaveSimulation_helper import extract_sensor_data

def pstd_elastic_2d(kgrid: kWaveGrid,
        source: kSource,
        sensor: Union[NotATransducer, kSensor],
        medium: kWaveMedium,
        simulation_options: SimulationOptions):
    """
    2D time-domain simulation of elastic wave propagation.

    DESCRIPTION:
        pstdElastic2D simulates the time-domain propagation of elastic waves
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

        pstdElastic2D may also be used for time reversal image reconstruction
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
        sensor_data = pstdElastic2D(kWaveGrid, kWaveMedium, kSource, kSensor)


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
                                  sensor.record (default = 0)
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

    # run subscript to check inputs
    k_sim = kWaveSimulation(kgrid=kgrid, source=source, sensor=sensor, medium=medium,
                            simulation_options=simulation_options)

    k_sim.input_checking("pstdElastic2D")

    k_sim.rho0 = np.atleast_1d(k_sim.rho0)

    m_rho0 : int = np.squeeze(k_sim.rho0).ndim
    m_mu : int = np.squeeze(_mu).ndim
    m_eta : int = np.squeeze(eta).ndim

    # assign the lame parameters
    _mu     = medium.sound_speed_shear**2 * medium.density
    _lambda = medium.sound_speed_compression**2 * medium.density - 2.0 * _mu

    # assign the viscosity coefficients
    if options.kelvin_voigt_model:
        eta = 2.0 * k_sim.rho0 * medium.sound_speed_shear**3 * db2neper(medium.alpha_coeff_shear, 2)
        chi = 2.0 * k_sim.rho0 * medium.sound_speed_compression**3 * db2neper(medium.alpha_coeff_compression, 2) - 2.0 * eta


    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================

    options = k_sim.options

    # calculate the values of the density at the staggered grid points
    # using the arithmetic average [1, 2], where sgx  = (x + dx/2, y) and
    # sgy  = (x, y + dy/2)
    if (m_rho0 == 2 and options.use_sg):

        # rho0 is heterogeneous and staggered grids are used
        rho0_sgx = interpn(kgrid.x, kgrid.y, k_sim.rho0, kgrid.x + kgrid.dx/2, kgrid.y, 'linear')
        rho0_sgy = interpn(kgrid.x, kgrid.y, k_sim.rho0, kgrid.x, kgrid.y + kgrid.dy/2, 'linear')

        # set values outside of the interpolation range to original values
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
        mu_sgxy  = 1.0 / interpn(kgrid.x, kgrid.y, 1.0 / _mu, kgrid.x + kgrid.dx/2, kgrid.y + kgrid.dy/2, 'linear')

        # set values outside of the interpolation range to original values
        mu_sgxy[np.isnan(mu_sgxy)] = _mu[np.isnan(mu_sgxy)]

    else:

        # mu is homogeneous or staggered grids are not used
        mu_sgxy = _mu


    # calculate the values of eta at the staggered grid points using the
    # harmonic average [1, 2], where sgxy = (x + dx/2, y + dy/2)
    if options.kelvin_voigt_model:
        if (m_eta == 2 and options.use_sg):

            # eta is heterogeneous and staggered grids are used
            eta_sgxy  = 1.0 / interpn(kgrid.x, kgrid.y, 1./eta, kgrid.x + kgrid.dx/2, kgrid.y + kgrid.dy/2, 'linear')

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
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the regular PML operators based on the reference sound speed and PML settings
    Nx, Ny = k_sim.kgrid.Nx, k_sim.kgrid.Ny
    dx, dy = k_sim.kgrid.dx, k_sim.kgrid.dy
    dt = k_sim.kgrid.dt

    pml_x_alpha, pml_y_alpha = options.pml_x_alpha, options.pml_y_alpha
    pml_x_size, pml_y_size = options.pml_x_size, options.pml_y_size
    c_ref = k_sim.c_ref

    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0)
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True,  0)
    pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 1)
    pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True,  1)

    # get the multi-axial PML operators
    multi_axial_PML_ratio: float = 1.0
    mpml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, False, 0)
    mpml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, multi_axial_PML_ratio * pml_x_alpha, True,  0)
    mpml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, False, 1)
    mpml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, multi_axial_PML_ratio * pml_y_alpha, True,  1)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using ifftshift (the option
    # options.use_sg exists for debugging)
    if options.use_sg:

        ddx_k_shift_pos = np.fft.ifftshift(1j * kgrid.kx_vec * np.exp( 1j * kgrid.kx_vec * dx / 2.0))
        ddx_k_shift_neg = np.fft.ifftshift(1j * kgrid.kx_vec * np.exp(-1j * kgrid.kx_vec * dx / 2.0))
        ddy_k_shift_pos = np.fft.ifftshift(1j * kgrid.ky_vec * np.exp( 1j * kgrid.ky_vec * dy / 2.0))
        ddy_k_shift_neg = np.fft.ifftshift(1j * kgrid.ky_vec * np.exp(-1j * kgrid.ky_vec * dy / 2.0))
    else:
        ddx_k_shift_pos = np.fft.ifftshift(1j * kgrid.kx_vec)
        ddx_k_shift_neg = np.fft.ifftshift(1j * kgrid.kx_vec)
        ddy_k_shift_pos = np.fft.ifftshift(1j * kgrid.ky_vec)
        ddy_k_shift_neg = np.fft.ifftshift(1j * kgrid.ky_vec)


    # force the derivative and shift operators to be in the correct direction
    # for use with BSXFUN
    ddy_k_shift_pos = ddy_k_shift_pos.T
    ddy_k_shift_neg = ddy_k_shift_neg.T

    # =========================================================================
    # DATA CASTING
    # =========================================================================

    # run subscript to cast the remaining loop variables to the data type
    # specified by data_cast
    if not (options.data_cast == 'off'):
        myType = 'np.single'
    else:
        myType = 'np.double'

    # preallocate the loop variables
    ux_split_x = np.zeros((kgrid.Nx, kgrid.Ny))
    ux_split_y = np.zeros((kgrid.Nx, kgrid.Ny))
    ux_sgx = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    uy_split_x = np.zeros((kgrid.Nx, kgrid.Ny))
    uy_split_y = np.zeros((kgrid.Nx, kgrid.Ny))
    uy_sgy = np.zeros((kgrid.Nx, kgrid.Ny))  # **

    sxx_split_x = np.zeros((kgrid.Nx, kgrid.Ny))
    sxx_split_y = np.zeros((kgrid.Nx, kgrid.Ny))
    syy_split_x = np.zeros((kgrid.Nx, kgrid.Ny))
    syy_split_y = np.zeros((kgrid.Nx, kgrid.Ny))
    sxy_split_x = np.zeros((kgrid.Nx, kgrid.Ny))
    sxy_split_y = np.zeros((kgrid.Nx, kgrid.Ny))

    duxdx = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    duxdy = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    duydy = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    duydx = np.zeros((kgrid.Nx, kgrid.Ny))  # **

    dsxxdx = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    dsxydy = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    dsxydx = np.zeros((kgrid.Nx, kgrid.Ny))  # **
    dsyydy = np.zeros((kgrid.Nx, kgrid.Ny))  # **

    p = np.zeros((kgrid.Nx, kgrid.Ny))  # **

    if options.kelvin_voigt_model:
        dduxdxdt = np.zeros(grid_shape, myType)  # **
        dduydydt = np.zeros(grid_shape, myType)  # **
        dduxdydt = np.zeros(grid_shape, myType)  # **
        dduydxdt = np.zeros(grid_shape, myType)  # **


    # to save memory, the variables noted with a ** do not neccesarily need to
    # be explicitly stored (they are not needed for update steps). Instead they
    # could be replaced with a small number of temporary variables that are
    # reused several times during the time loop.


    # =========================================================================
    # CREATE INDEX VARIABLES
    # =========================================================================

    # setup the time index variable
    if (not options.time_rev):
        index_start = 0
        index_step = 1
        index_end = kgrid.Nt
    else:
        # throw error for unsupported feature
        raise TypeError('Time reversal using sensor.time_reversal_boundary_data is not currently supported.')


    # =========================================================================
    # PREPARE VISUALISATIONS
    # =========================================================================

    # pre-compute suitable axes scaling factor
    if (options.plot_layout or options.plot_sim):
        (x_sc, scale, prefix) = scale_SI(np.max([kgrid.x_vec, kgrid.y_vec]))


    # throw error for currently unsupported plot layout feature
    if options.plot_layout:
        raise TypeError('PlotLayout input is not currently supported.')

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
    print('  precomputation completed in ', scale_time(timer.toc()))
    print('  starting time loop...')

    # restart timing variables
    loop_start_time = timer.tic()


    if options.save_to_disk_exit:
        return


    # start time loop
    for t_index in np.arange(index_start, index_end, index_step):

        # compute the gradients of the stress tensor (these variables do not necessaily need to be stored, they could be computed as needed)

        # dsxxdx = np.real(np.fft.ifft( bsxfun(@times, ddx_k_shift_pos, fft(sxx_split_x + sxx_split_y, [], 1)), [], 1) );
        dsxxdx = np.real(np.fft.ifft(ddx_k_shift_pos * np.fft.fft(sxx_split_x + sxx_split_y, axis=0), axis=0))

        #dsyydy = real( ifft( bsxfun(@times, ddy_k_shift_pos, fft(syy_split_x + syy_split_y, [], 2)), [], 2) );
        dsyydy = np.real(np.fft.ifft(ddy_k_shift_pos * np.fft.fft(syy_split_x + syy_split_y, axis=1), axis=1))

        #dsxydx = real( ifft( bsxfun(@times, ddx_k_shift_neg, fft(sxy_split_x + sxy_split_y, [], 1)), [], 1) );
        dsxydx = np.real(np.fft.ifft(ddx_k_shift_neg * np.fft.fft(sxy_split_x + sxy_split_y, axis=0), axis=0))

        #dsxydy = real( ifft( bsxfun(@times, ddy_k_shift_neg, fft(sxy_split_x + sxy_split_y, [], 2)), [], 2) );
        dsxydy = np.real(np.fft.ifft(ddy_k_shift_neg * np.fft.fft(sxy_split_x + sxy_split_y, axis=1), axis=1))

        # calculate the split-field components of ux_sgx and uy_sgy at the next
        # time step using the components of the stress at the current time step

        # ux_split_x = bsxfun(@times, mpml_y,
        #                     bsxfun(@times, pml_x_sgx,
        #                            bsxfun(@times, mpml_y,
        #                                   bsxfun(@times, pml_x_sgx, ux_split_x)) + dt .* rho0_sgx_inv .* dsxxdx));
        a = pml_x_sgx * ux_split_x
        b = mpml_y * a
        c = b + kgrid.dt * rho0_sgx_inv * dsxxdx
        d = pml_x_sgx * c
        ux_split_x = mpml_y * d

        # ux_split_y = bsxfun(@times, mpml_x_sgx,
        #                     bsxfun(@times, pml_y,
        #                            bsxfun(@times, mpml_x_sgx,
        #                                   bsxfun(@times, pml_y, ux_split_y)) + dt .* rho0_sgx_inv .* dsxydy));
        a = pml_y * ux_split_y
        b = mpml_x_sgx * a
        c = b + kgrid.dt * rho0_sgx_inv * dsxydy
        d = pml_y * c
        ux_split_y = mpml_x_sgx * d

        # uy_split_x = bsxfun(@times, mpml_y_sgy,
        #                     bsxfun(@times, pml_x,
        #                            bsxfun(@times, mpml_y_sgy,
        #                                   bsxfun(@times, pml_x, uy_split_x)) + dt .* rho0_sgy_inv .* dsxydx));
        a = pml_x * uy_split_x
        b = mpml_y_sgy * a
        c = b + kgrid.dt * rho0_sgy_inv * dsxydx
        d = pml_x * c
        uy_split_x = mpml_y_sgy * d

        # uy_split_y = bsxfun(@times, mpml_x,
        #                     bsxfun(@times, pml_y_sgy,
        #                            bsxfun(@times, mpml_x,
        #                                   bsxfun(@times, pml_y_sgy, uy_split_y)) + dt .* rho0_sgy_inv .* dsyydy));
        a = pml_y_sgy * uy_split_y
        b = mpml_x * a
        c = b + kgrid.dt * rho0_sgy_inv * dsyydy
        d = pml_y_sgy * c
        uy_split_y = mpml_x * d

        # add in the pre-scaled velocity source terms
        if (options.source_ux >= t_index):
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                ux_split_x[k_sim.u_source_pos_index] = source.ux[k_sim.u_source_sig_index, t_index]

            else:
                # add the source values to the existing field values
                ux_split_x[k_sim.u_source_pos_index] = ux_split_x[k_sim.u_source_pos_index] + source.ux[k_sim.u_source_sig_index, t_index]


        if (options.source_uy >= t_index):
            if (source.u_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                uy_split_y[k_sim.u_source_pos_index] = source.uy[k_sim.u_source_sig_index, t_index]

            else:
                # add the source values to the existing field values
                uy_split_y[k_sim.u_source_pos_index] = uy_split_y[k_sim.u_source_pos_index] + source.uy[k_sim.u_source_sig_index, t_index]


        # Q - should the velocity source terms for the Dirichlet condition be
        # added to the split or combined velocity field?

        # combine split field components (these variables do not necessarily
        # need to be stored, they could be computed when needed)
        ux_sgx = ux_split_x + ux_split_y
        uy_sgy = uy_split_x + uy_split_y

        # calculate the velocity gradients (these variables do not necessarily
        # need to be stored, they could be computed when needed)

        # duxdx = real( ifft( bsxfun(@times, ddx_k_shift_neg, fft(ux_sgx, [], 1)), [], 1));
        duxdx = np.real(np.fft.ifft(ddx_k_shift_neg * np.fft.fft(ux_sgx, axis=0), axis=0))

        # duxdy = real( ifft( bsxfun(@times, ddy_k_shift_pos, fft(ux_sgx, [], 2)), [], 2));
        duxdy = np.real(np.fft.ifft(ddy_k_shift_pos * np.fft.fft(ux_sgx, axis=1), axis=1))

        # duydx = real( ifft( bsxfun(@times, ddx_k_shift_pos, fft(uy_sgy, [], 1)), [], 1));
        duydx = np.real(np.fft.ifft(ddx_k_shift_pos * np.fft.fft(uy_sgy, axis=0), axis=0))

        # duydy = real( ifft( bsxfun(@times, ddy_k_shift_neg, fft(uy_sgy, [], 2)), [], 2));
        duydx = np.real(np.fft.ifft(ddy_k_shift_neg * np.fft.fft(uy_sgy, axis=1), axis=1))

        # update the normal components and shear components of stress tensor
        # using a split field pml
        if options.kelvin_voigt_model:

            # compute additional gradient terms needed for the Kelvin-Voigt
            # model

            #dduxdxdt = real(ifft( bsxfun(@times, ddx_k_shift_neg, fft( (dsxxdx + dsxydy) .* rho0_sgx_inv , [], 1 )), [], 1));
            #dduxdydt = real(ifft( bsxfun(@times, ddy_k_shift_pos, fft( (dsxxdx + dsxydy) .* rho0_sgx_inv , [], 2 )), [], 2));
            temp = (dsxxdx + dsxydy) * rho0_sgx_inv
            dduxdxdt = np.real(np.fft.ifft(ddx_k_shift_neg * np.fft.fft(temp, axis=0), axis=0))
            dduxdydt = np.real(np.fft.ifft(ddx_k_shift_pos * np.fft.fft(temp, axis=1), axis=1))

            #dduydydt = real(ifft( bsxfun(@times, ddy_k_shift_neg, fft( (dsyydy + dsxydx) .* rho0_sgy_inv , [], 2 )), [], 2));
            #dduydxdt = real(ifft( bsxfun(@times, ddx_k_shift_pos, fft( (dsyydy + dsxydx) .* rho0_sgy_inv , [], 1 )), [], 1));
            temp = (dsyydy + dsxydx) * rho0_sgy_inv
            dduydydt = np.real(np.fft.ifft(ddy_k_shift_neg * np.fft.fft(temp, axis=1), axis=1))
            dduydxdt = np.real(np.fft.ifft(ddx_k_shift_pos * np.fft.fft(temp, axis=0), axis=0))

            # update the normal shear components of the stress tensor using a
            # Kelvin-Voigt model with a split-field multi-axial pml
            # sxx_split_x = bsxfun(@times, mpml_y,
            #                      bsxfun(@times, pml_x,
            #                             bsxfun(@times, mpml_y,
            #                                    bsxfun(@times, pml_x, sxx_split_x)) + dt .* (2 .* _mu + _lambda) .* duxdx + dt .* (2 .* eta + chi) .* dduxdxdt));
            a = pml_x * sxx_split_x
            b = a * mpml_y
            c = b + dt * (2.0 * _mu + _lambda) * duxdx + dt * (2.0 * eta + chi) * dduxdxdt
            d = c * pml_x
            sxx_split_x = d * mpml_y

            # sxx_split_y = bsxfun(@times, mpml_x,
            #                      bsxfun(@times, pml_y,
            #                             bsxfun(@times, mpml_x,
            #                                    bsxfun(@times, pml_y, sxx_split_y)) + dt .* _lambda .* duydy + dt .* chi .* dduydydt));
            a = pml_y * sxx_split_y
            b = a * mpml_x
            c = b + dt * (_lambda * duydy + chi * dduydydt)
            d = c * pml_y
            sxx_split_y = d * mpml_x

            # syy_split_x = bsxfun(@times, mpml_y,
            #                      bsxfun(@times, pml_x,
            #                             bsxfun(@times, mpml_y,
            #                                    bsxfun(@times, pml_x, syy_split_x)) + dt .* _lambda .* duxdx + dt .* chi .* dduxdxdt));
            a = pml_x * syy_split_x
            b = a * mpml_y
            c = b + dt * _lambda * duxdx + dt * chi * dduxdxdt
            d = c * pml_x
            sxx_split_y = d * mpml_y

            # syy_split_y = bsxfun(@times, mpml_x,
            #                      bsxfun(@times, pml_y,
            #                             bsxfun(@times, mpml_x,
            #                                    bsxfun(@times, pml_y, syy_split_y)) + dt .* (2 .* _mu + _lambda) .* duydy + dt .* (2 .* eta + chi) .* dduydydt));
            a = pml_y * syy_split_y
            b = a * mpml_x
            c = b + 2.0 * dt * ((_mu + _lambda) * duydy + ( eta + chi) * dduydydt)
            d = c * pml_y
            syy_split_y = d * mpml_x

            # sxy_split_x = bsxfun(@times, mpml_y_sgy,
            #                      bsxfun(@times, pml_x_sgx,
            #                             bsxfun(@times, mpml_y_sgy,
            #                                    bsxfun(@times, pml_x_sgx, sxy_split_x)) + dt .* mu_sgxy .* duydx + dt .* eta_sgxy .* dduydxdt));
            a = pml_x_sgx * sxy_split_x
            b = a * mpml_y_sgy
            c = b + dt * (mu_sgxy * duydx + eta_sgxy * dduydxdt)
            d = c * pml_x_sgx
            sxy_split_x = d * mpml_y_sgy

            # sxy_split_y = bsxfun(@times, mpml_x_sgx,
            #                      bsxfun(@times, pml_y_sgy,
            #                             bsxfun(@times, mpml_x_sgx,
            #                                    bsxfun(@times, pml_y_sgy, sxy_split_y)) + dt .* mu_sgxy .* duxdy + dt .* eta_sgxy .* dduxdydt));
            a = pml_y_sgy * sxy_split_y
            b = a * mpml_x_sgx
            c = b + dt * (mu_sgxy * duxdy + eta_sgxy * dduxdydt)
            d = c * pml_y_sgy
            sxy_split_y = d * mpml_x_sgx

        else:

            # update the normal and shear components of the stress tensor using
            # a lossless elastic model with a split-field multi-axial pml
            # sxx_split_x = bsxfun(@times, mpml_y,
            #                      bsxfun(@times, pml_x,
            #                             bsxfun(@times, mpml_y,
            #                                    bsxfun(@times, pml_x, sxx_split_x)) + dt .* (2 .* _mu + _lambda) .* duxdx));
            a = pml_x * sxx_split_x
            b = a * mpml_y
            c = b + dt * (2.0 * _mu + _lambda) * duxdx
            d = c * pml_x
            sxx_split_x = d * mpml_y

            # sxx_split_y = bsxfun(@times, mpml_x,
            #                      bsxfun(@times, pml_y,
            #                             bsxfun(@times, mpml_x,
            #                                    bsxfun(@times, pml_y, sxx_split_y)) + dt .* _lambda .* duydy));
            a = pml_y * sxx_split_y
            b = a * mpml_x
            c = b + dt * _lambda * duydy
            d = c * pml_y
            sxx_split_y = d * mpml_x

            # syy_split_x = bsxfun(@times, mpml_y,
            #                      bsxfun(@times, pml_x,
            #                             bsxfun(@times, mpml_y,
            #                                    bsxfun(@times, pml_x, syy_split_x)) + dt .* _lambda .* duxdx));
            a = pml_x * syy_split_x
            b = a * mpml_y
            c = b + dt * _lambda * duxdx
            d = c * pml_x
            syy_split_x = d * mpml_y

            # syy_split_y = bsxfun(@times, mpml_x,
            #                      bsxfun(@times, pml_y,
            #                             bsxfun(@times, mpml_x,
            #                                    bsxfun(@times, pml_y, syy_split_y)) + dt .* (2 .* _mu + _lambda) .* duydy));
            a = pml_y * syy_split_y
            b = a * mpml_x
            c = b + dt * (2 * _mu + _lambda) * duydy
            d = c * pml_y
            syy_split_y = d * mpml_x

            # sxy_split_x = bsxfun(@times, mpml_y_sgy,
            #                      bsxfun(@times, pml_x_sgx,
            #                             bsxfun(@times, mpml_y_sgy,
            #                                    bsxfun(@times, pml_x_sgx, sxy_split_x)) + dt .* mu_sgxy .* duydx));
            a = pml_x_sgx * sxy_split_x
            b = a * mpml_y_sgy
            c = b + dt * mu_sgxy * duydx
            d = c * pml_x_sgx
            sxy_split_x = d * mpml_y_sgy

            # sxy_split_y = bsxfun(@times, mpml_x_sgx,
            #                      bsxfun(@times, pml_y_sgy,
            #                             bsxfun(@times, mpml_x_sgx,
            #                                    bsxfun(@times, pml_y_sgy, sxy_split_y)) + dt .* mu_sgxy .* duxdy));
            a = pml_y_sgy * sxy_split_y
            b = a * mpml_x_sgx
            c = b + dt * mu_sgxy * duxdy
            d = c * pml_y_sgy
            sxy_split_y = d * mpml_x_sgx



        # add in the pre-scaled stress source terms
        if (options.source_sxx >= t_index):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxx_split_x[k_sim.s_source_pos_index] = source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_y[k_sim.s_source_pos_index] = source.sxx[k_sim.s_source_sig_index, t_index]

            else:
                # add the source values to the existing field values
                sxx_split_x[k_sim.s_source_pos_index] = sxx_split_x[k_sim.s_source_pos_index] + source.sxx[k_sim.s_source_sig_index, t_index]
                sxx_split_y[k_sim.s_source_pos_index] = sxx_split_y[k_sim.s_source_pos_index] + source.sxx[k_sim.s_source_sig_index, t_index]


        if (options.source_syy >= t_index):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                syy_split_x[k_sim.s_source_pos_index] = source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_y[k_sim.s_source_pos_index] = source.syy[k_sim.s_source_sig_index, t_index]

            else:
                # add the source values to the existing field values
                syy_split_x[k_sim.s_source_pos_index] = syy_split_x[k_sim.s_source_pos_index] + source.syy[k_sim.s_source_sig_index, t_index]
                syy_split_y[k_sim.s_source_pos_index] = syy_split_y[k_sim.s_source_pos_index] + source.syy[k_sim.s_source_sig_index, t_index]


        if (options.source_sxy >= t_index):
            if (source.s_mode == 'dirichlet'):
                # enforce the source values as a dirichlet boundary condition
                sxy_split_x[k_sim.s_source_pos_index] = source.sxy[k_sim.s_source_sig_index, t_index]
                sxy_split_y[k_sim.s_source_pos_index] = source.sxy[k_sim.s_source_sig_index, t_index]

            else:
                # add the source values to the existing field values
                sxy_split_x[k_sim.s_source_pos_index] = sxy_split_x[k_sim.s_source_pos_index] + source.sxy[k_sim.s_source_sig_index, t_index]
                sxy_split_y[k_sim.s_source_pos_index] = sxy_split_y[k_sim.s_source_pos_index] + source.sxy[k_sim.s_source_sig_index, t_index]


        # compute pressure from normal components of the stress
        p = -(sxx_split_x + sxx_split_y + syy_split_x + syy_split_y) / 2.0

        # extract required sensor data from the pressure and particle velocity
        # fields if the number of time steps elapsed is greater than
        # sensor.record_start_index (defaults to 1)
        if (options.use_sensor and (not options.elastic_time_rev) and (t_index >= sensor.record_start_index)):

            # update index for data storage
            file_index = t_index - sensor.record_start_index + 1

            # run sub-function to extract the required data
            sensor_data = extract_sensor_data(2, sensor_data, file_index, sensor_mask_index, options, record, p, ux_sgx, uy_sgy)



        # estimate the time to run the simulation
        ESTIMATE_SIM_TIME_STEPS = kgrid.Nt
        if (t_index == ESTIMATE_SIM_TIME_STEPS):

            # print estimated simulation time
            print('  estimated simulation time ', scale_time(etime(clock, loop_start_time) * index_end / t_index), '...')

            # check memory usage
            # kspaceFirstOrder_checkMemoryUsage;



        # plot data if required
        # if (options.plot_sim and (rem(t_index, plot_freq) == 0 or t_index == 1 or t_index == index_end)):

        #     # update progress bar
        #     waitbar(t_index / kgrid.Nt, pbar);
        #     drawnow;

        #     # ensure p is cast as a CPU variable and remove the PML from the
        #     # plot if required
        #     if (data_cast == 'gpuArray'):
        #         sii_plot = p[x1:x2, y1:y2]
        #         sij_plot = sxy_split_x[x1:x2, y1:y2] + sxy_split_y[x1:x2, y1:y2]
        #     else:
        #         sii_plot = p[x1:x2, y1:y2].astype(np.double)
        #         sij_plot = sxy_split_x[x1:x2, y1:y2].astype(np.double) + sxy_split_y[x1:x2, y1:y2].astype(np.double)


        #     # update plot scale if set to automatic or log
        #     if (options.plot_scale_auto or options.plot_scale_log):
        #         kspaceFirstOrder_adjustPlotScale;


        #     # add mask onto plot
        #     if (display_mask == 'default'):
        #         sii_plot[sensor.mask[x1:x2, y1:y2]] = plot_scale[1]
        #         sij_plot[sensor.mask[x1:x2, y1:y2]] = plot_scale[-1]
        #     elif not (display_mask == 'off'):
        #         sii_plot[display_mask[x1:x2, y1:y2] != 0] = plot_scale[1]
        #         sij_plot[display_mask[x1:x2, y1:y2] != 0] = plot_scale[-1]


        #     # update plot
        #     subplot(1, 2, 1);
        #     imagesc(kgrid.y_vec[y1:y2] * scale, kgrid.x_vec[x1:x2] * scale, sii_plot, plot_scale[0:1]);
        #     colormap(COLOR_MAP);
        #     ylabel(['x-position [' prefix 'm]']);
        #     xlabel(['y-position [' prefix 'm]']);
        #     title('Normal Stress (\sigma_{ii}/2)')
        #     axis image;

        #     subplot(1, 2, 2);
        #     imagesc(kgrid.y_vec(y1:y2) .* scale, kgrid.x_vec(x1:x2) .* scale, sij_plot, plot_scale(end - 1:end));
        #     colormap(COLOR_MAP);
        #     ylabel(['x-position [' prefix 'm]']);
        #     xlabel(['y-position [' prefix 'm]']);
        #     title('Shear Stress (\sigma_{xy})')
        #     axis image;

        #     # force plot update
        #     drawnow;

            # # save movie frame if required
            # if options.record_movie:

            #     # set background color to white
            #     set(gcf, 'Color', [1 1 1]);

            #     # save the movie frame
            #     writeVideo(video_obj, getframe(gcf));



            # update variable used for timing variable to exclude the first
            # time step if plotting is enabled
            if t_index == 0:
                clock1 = TicToc()
                clock1.tic()
                loop_start_time = clock1.start_time



    # update command line status
    print('  simulation completed in ', scale_time(timer.toc()))



    # =========================================================================
    # CLEAN UP
    # =========================================================================

    # clean up used figures
    if options.plot_sim:
        # close(img);
        # close(pbar);
        pass


    # save the movie frames to disk
    if options.record_movie:
        # close(video_obj);
        pass


    # save the final acoustic pressure if required
    if (options.record_p_final or options.elastic_time_rev):
        sensor_data.p_final = p[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]


    # save the final particle velocity if required
    if options.record_u_final:
        sensor_data.ux_final = ux_sgx[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]
        sensor_data.uy_final = uy_sgy[record.x1_inside:record.x2_inside, record.y1_inside:record.y2_inside]


    # run subscript to cast variables back to double precision if required
    if options.data_recast:
        #kspaceFirstOrder_dataRecast;
        pass


    # run subscript to compute and save intensity values
    if (options.use_sensor and (not options.elastic_time_rev) and (options.record_I or options.record_I_avg)):
        # save_intensity_matlab_code = True
        # kspaceFirstOrder_saveIntensity;
        pass

    # reorder the sensor points if a binary sensor mask was used for Cartesian
    # sensor mask nearest neighbour interpolation (this is performed after
    # recasting as the GPU toolboxes do not all support this subscript)
    if (options.use_sensor and options.reorder_data):
        # kspaceFirstOrder_reorderCartData;
        pass


    # filter the recorded time domain pressure signals if transducer filter
    # parameters are given
    if (options.use_sensor and not options.elastic_time_rev and hasattr(sensor, 'frequency_response')):
        fs = 1.0 / kgrid.dt
        sensor_data.p = gaussian_filter(sensor_data.p, fs, sensor.frequency_response[0], sensor.frequency_response[1])


    # reorder the sensor points if cuboid corners is used (outputs are indexed
    # as [X, Y, T] or [X, Y] rather than [sensor_index, time_index]
    if options.cuboid_corners:
        sensor_data = reorder_sensor_data(kgrid, sensor, sensor_data)


    if options.elastic_time_rev:
        # if computing time reversal, reassign sensor_data.p_final to
        # sensor_data
        sensor_data = sensor_data.p_final

    elif (not options.use_sensor):
        # if sensor is not used, return empty sensor data
        sensor_data = None

    elif ((not hasattr(sensor, 'record')) and (not options.cuboid_corners)):
        # if sensor.record is not given by the user, reassign sensor_data.p to sensor_data
        sensor_data = sensor_data.p


    # update command line status
    print('  total computation time ', scale_time(etime(clock, start_time)))

    # switch off log
    # if options.create_log:
    #     diary off;

    return sensor_data
