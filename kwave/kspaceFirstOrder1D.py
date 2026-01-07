import numpy as np
import scipy.fft
from tqdm import tqdm
from typing import Union

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kWaveSimulation import kWaveSimulation
from kwave.kWaveSimulation_helper import extract_sensor_data
from kwave.ktransducer import NotATransducer

from kwave.utils.data import scale_time
from kwave.utils.math import sinc
from kwave.utils.pml import get_pml
from kwave.utils.tictoc import TicToc
from kwave.utils.dotdictionary import dotdict

from kwave.options.simulation_options import SimulationOptions

def kspace_first_order_1D(kgrid: kWaveGrid,
                          source: kSource,
                          sensor: Union[NotATransducer, kSensor],
                          medium: kWaveMedium,
                          simulation_options: SimulationOptions):

    """
    1D time-domain simulation of wave propagation.

    DESCRIPTION:
        kspaceFirstOrder1D simulates the time-domain propagation of
        compressional waves through a one-dimensional homogeneous or
        heterogeneous acoustic medium given four input structures: kgrid,
        medium, source, and sensor. The computation is based on a first-order
        k-space model which accounts for power law absorption and a
        heterogeneous sound speed and density. If medium.BonA is specified,
        cumulative nonlinear effects are also modelled. At each time-step
        (defined by kgrid.dt and kgrid.Nt or kgrid.t_array), the acoustic
        field parameters at the positions defined by sensor.mask are recorded
        and stored. If kgrid.t_array is set to 'auto', this array is
        automatically generated using the makeTime method of the kWaveGrid
        class. An anisotropic absorbing boundary layer called a perfectly
        matched layer (PML) is implemented to prevent waves that leave one
        side of the domain being reintroduced from the opposite side (a
        consequence of using the FFT to compute the spatial derivatives in
        the wave equation). This allows infinite domain simulations to be
        computed using small computational grids.

        For a homogeneous medium the formulation is exact and the time-steps
        are only limited by the effectiveness of the perfectly matched layer.
        For a heterogeneous medium, the solution represents a leap-frog
        pseudospectral method with a k-space correction that improves the
        accuracy of computing the temporal derivatives. This allows larger
        time-steps to be taken for the same level of accuracy compared to
        conventional pseudospectral time-domain methods. The computational
        grids are staggered both spatially and temporally.

        An initial pressure distribution can be specified by assigning a
        matrix (the same size as the computational grid) of arbitrary numeric
        values to source.p0. A time varying pressure source can similarly be
        specified by assigning a binary matrix (i.e., a matrix of 1's and 0's
        with the same dimensions as the computational grid) to source.p_mask
        where the 1's represent the grid points that form part of the source.
        The time varying input signals are then assigned to source.p. This
        can be a single time series (in which case it is applied to all
        source elements), or a matrix of time series following the source
        elements using MATLAB's standard column-wise linear matrix index
        ordering. A time varying velocity source can be specified in an
        analogous fashion, where the source location is specified by
        source.u_mask, and the time varying input velocity is assigned to
        source.ux.

        The field values are returned as arrays of time series at the sensor
        locations defined by sensor.mask. This can be defined in three
        different ways. (1) As a binary matrix (i.e., a matrix of 1's and 0's
        with the same dimensions as the computational grid) representing the
        grid points within the computational grid that will collect the data.
        (2) As the grid coordinates of two opposing ends of a line in the
        form [x1; x2]. This is equivalent to using a binary sensor mask
        covering the same region, however, the output is indexed differently
        as discussed below. (3) As a series of Cartesian coordinates within
        the grid which specify the location of the pressure values stored at
        each time step. If the Cartesian coordinates don't exactly match the
        coordinates of a grid point, the output values are calculated via
        interpolation. The Cartesian points must be given as a 1 by N matrix
        corresponding to the x positions, where the Cartesian origin is
        assumed to be in the center of the grid. If no output is required,
        the sensor input can be replaced with an empty array [].

        If sensor.mask is given as a set of Cartesian coordinates, the
        computed sensor_data is returned in the same order. If sensor.mask is
        given as a binary matrix, sensor_data is returned using MATLAB's
        standard column-wise linear matrix index ordering. In both cases, the
        recorded data is indexed as sensor_data(sensor_point_index,
        time_index). For a binary sensor mask, the field values at a
        particular time can be restored to the sensor positions within the
        computation grid using unmaskSensorData. If sensor.mask is given as a
        list of opposing ends of a line, the recorded data is indexed as
        sensor_data(line_index).p(x_index, time_index), where x_index
        corresponds to the grid index within the line, and line_index
        corresponds to the number of lines if more than one is specified.

        By default, the recorded acoustic pressure field is passed directly
        to the output sensor_data. However, other acoustic parameters can
        also be recorded by setting sensor.record to a cell array of the form
        {'p', 'u', 'p_max', ...}. For example, both the particle velocity and
        the acoustic pressure can be returned by setting sensor.record =
        {'p', 'u'}. If sensor.record is given, the output sensor_data is
        returned as a structure with the different outputs appended as
        structure fields. For example, if sensor.record = {'p', 'p_final',
        'p_max', 'u'}, the output would contain fields sensor_data.p,
        sensor_data.p_final, sensor_data.p_max, and sensor_data.ux. Most of
        the output parameters are recorded at the given sensor positions and
        are indexed as sensor_data.field(sensor_point_index, time_index) or
        sensor_data(line_index).field(x_index, time_index) if using a sensor
        mask defined as opposing ends of a line. The exceptions are the
        averaged quantities ('p_max', 'p_rms', 'u_max', 'p_rms', 'I_avg'),
        the 'all' quantities ('p_max_all', 'p_min_all', 'u_max_all',
        'u_min_all'), and the final quantities ('p_final', 'u_final'). The
        averaged quantities are indexed as
        sensor_data.p_max(sensor_point_index) or
        sensor_data(line_index).p_max(x_index) if using line ends, while the
        final and 'all' quantities are returned over the entire grid and are
        always indexed as sensor_data.p_final(nx), regardless of the type of
        sensor mask.

        kspaceFirstOrder1D may also be used for time reversal image
        reconstruction by assigning the time varying pressure recorded over
        an arbitrary sensor surface to the input field
        sensor.time_reversal_boundary_data. This data is then enforced in
        time reversed order as a time varying Dirichlet boundary condition
        over the sensor surface given by sensor.mask. The boundary data must
        be indexed as sensor.time_reversal_boundary_data(sensor_point_index,
        time_index). If sensor.mask is given as a set of Cartesian
        coordinates, the boundary data must be given in the same order. An
        equivalent binary sensor mask (computed using nearest neighbour
        interpolation) is then used to place the pressure values into the
        computational grid at each time step. If sensor.mask is given as a
        binary matrix of sensor points, the boundary data must be ordered
        using MATLAB's standard column-wise linear matrix indexing. If no
        additional inputs are required, the source input can be replaced with
        an empty array [].

        Acoustic attenuation compensation can also be included during time
        reversal image reconstruction by assigning the absorption parameters
        medium.alpha_coeff and medium.alpha_power and reversing the sign of
        the absorption term by setting medium.alpha_sign = [-1, 1]. This
        forces the propagating waves to grow according to the absorption
        parameters instead of decay. The reconstruction should then be
        regularised by assigning a filter to medium.alpha_filter (this can be
        created using getAlphaFilter).

        Note: To run a simple photoacoustic image reconstruction example
        using time reversal (that commits the 'inverse crime' of using the
        same numerical parameters and model for data simulation and image
        reconstruction), the sensor_data returned from a k-Wave simulation
        can be passed directly to sensor.time_reversal_boundary_data with the
        input fields source.p0 and source.p removed or set to zero.

    USAGE:
        sensor_data = kspaceFirstOrder1D(kgrid, medium, source, sensor)
        sensor_data = kspaceFirstOrder1D(kgrid, medium, source, sensor, ...) 
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
    k_sim.input_checking("kspaceFirstOrder1D")

    # aliases from kWaveSimulation class
    sensor_data = k_sim.sensor_data
    options = k_sim.options
    record = k_sim.record

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2)
    rho0 = np.squeeze(k_sim.rho0)
    m_rho0: int = rho0.ndim

    if (m_rho0 > 0 and options.use_sg):

        points = np.squeeze(k_sim.kgrid.x_vec)

        # rho0 is heterogeneous and staggered grids are used
        rho0_sgx = np.interp(points + k_sim.kgrid.dx / 2.0,  points, np.squeeze(k_sim.rho0))

        # set values outside of the interpolation range to original values 
        rho0_sgx[np.isnan(rho0_sgx)] = np.squeeze(k_sim.rho0)[np.isnan(rho0_sgx)]
        
    else:
        # rho0 is homogeneous or staggered grids are not used
        rho0_sgx = k_sim.rho0

    # invert rho0 so it doesn't have to be done each time step
    rho0_sgx_inv = 1.0 / rho0_sgx

    # rho0_sgx_inv = rho0_sgx_inv[:, np.newaxis]

    # clear unused variables
    del rho0_sgx

    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the regular PML operators based on the reference sound speed and PML settings
    Nx = k_sim.kgrid.Nx
    dx = k_sim.kgrid.dx
    dt = k_sim.kgrid.dt
    Nt = k_sim.kgrid.Nt

    pml_x_alpha = options.pml_x_alpha
    pml_x_size = options.pml_x_size
    c_ref = k_sim.c_ref

    kx_vec = np.squeeze(k_sim.kgrid.k_vec[0])

    c0 = medium.sound_speed

    # get the PML operators based on the reference sound speed and PML settings
    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0)
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True,  0)

    pml_x = np.squeeze(pml_x)
    pml_x_sgx = np.squeeze(pml_x_sgx)

    # define the k-space derivative operator
    ddx_k = scipy.fft.ifftshift(1j * kx_vec)
    
    #    ddx_k = ddx_k[:, np.newaxis]

    # define the staggered grid shift operators (the option options.use_sg exists for debugging)
    if options.use_sg:
        ddx_k_shift_pos = scipy.fft.ifftshift( np.exp( 1j * kx_vec * dx / 2.0))
        ddx_k_shift_neg = scipy.fft.ifftshift( np.exp(-1j * kx_vec * dx / 2.0))
        #ddx_k_shift_pos = ddx_k_shift_pos[:, np.newaxis]
        #ddx_k_shift_neg = ddx_k_shift_neg[:, np.newaxis]
    else:
        ddx_k_shift_pos = 1.0
        ddx_k_shift_neg = 1.0 
    

    # create k-space operator (the option options.use_kspace exists for debugging)
    if options.use_kspace:
        kappa        = scipy.fft.ifftshift(sinc(c_ref * kgrid.k * kgrid.dt / 2.0))
        kappa = np.squeeze(kappa)
        # kappa        = kappa[:, np.newaxis]
        if (hasattr(options, 'source_p') and hasattr(k_sim.source, 'p_mode')) and (k_sim.source.p_mode == 'additive') or \
           (hasattr(options, 'source_ux') and hasattr(k_sim.source, 'u_mode')) and (k_sim.source.u_mode == 'additive'):
            source_kappa = scipy.fft.ifftshift(np.cos (c_ref * kgrid.k * kgrid.dt / 2.0))
            #source_kappa = source_kappa[:, np.newaxis]
    else:
        kappa        = 1.0
        source_kappa = 1.0


    # =========================================================================
    # DATA CASTING
    # =========================================================================

    # preallocate the loop variables using the castZeros anonymous function
    # (this creates a matrix of zeros in the data type specified by data_cast)
    if not (options.data_cast == 'off'):
        myType = np.single
    else:
        myType = np.double

    grid_shape = (Nx, )

    # preallocate the loop variables
    p      = np.zeros(grid_shape, dtype=myType)
    rhox   = np.zeros(grid_shape, dtype=myType)
    ux_sgx = np.zeros(grid_shape, dtype=myType)
    p_k    = np.zeros(grid_shape, dtype=myType)

    c0    = c0.astype(myType)

    # =========================================================================
    # CREATE INDEX VARIABLES
    # =========================================================================

    # setup the time index variable
    index_start: int = 0
    index_step: int = 1
    index_end: int = Nt

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

    # =========================================================================
    # PREPARE VISUALISATIONS
    # =========================================================================

    # # pre-compute suitable axes scaling factor
    # if options.plot_layout or options.plot_sim
    #     [x_sc, scale, prefix] = scaleSI(max(kgrid.x));  ##ok<ASGLU>
    # end

    # # run subscript to plot the simulation layout if 'PlotLayout' is set to true
    # if options.plot_layout
    #     kspaceFirstOrder_plotLayout;
    # end

    # # initialise the figure used for animation if 'PlotSim' is set to 'true'
    # if options.plot_sim
    #     kspaceFirstOrder_initialiseFigureWindow;
    # end  

    # # initialise movie parameters if 'RecordMovie' is set to 'true'
    # if options.record_movie
    #     kspaceFirstOrder_initialiseMovieParameters;
    # end

    # =========================================================================
    # LOOP THROUGH TIME STEPS
    # =========================================================================

    # update command line status
    t0 = timer.toc()
    t0_scale = scale_time(t0)
    print('\tprecomputation completed in', t0_scale)
    print('\tstarting time loop...')

    # start time loop
    for t_index in tqdm(np.arange(index_start, index_end, index_step, dtype=int)):
        
        # calculate ux at the next time step using dp/dx at the current time step
        if not k_sim.nonuniform_grid and not options.use_finite_difference:

            # calculate gradient using the k-space method on a regular grid
            ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - 
                                  dt * rho0_sgx_inv * np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_pos * kappa * p_k, axes=(0,))))
        
        elif options.use_finite_difference:
            match options.use_finite_difference:
                case 2:
                    # calculate gradient using second-order accurate finite
                    # difference scheme (including half step forward)
                    dpdx =  (np.append(p[1:], 0.0) - p) / kgrid.dx 
                    ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - dt * rho0_sgx_inv * dpdx )
                    
                case 4:
                    # calculate gradient using fourth-order accurate finite
                    # difference scheme (including half step forward)
                    dpdx = (np.insert(p[:-1], 0, 0) - 27.0 * p + 27 * np.append(p[1:], 0.0) - np.append(p[2:], [0, 0])) / (24.0 * kgrid.dx)
                    ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - dt * rho0_sgx_inv * dpdx )
                    
        else:           
            # calculate gradient using the k-space method on a non-uniform grid
            # via the mapped pseudospectral method         
            ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - 
                                  dt * rho0_sgx_inv * k_sim.kgrid.dxudxn_sgx * np.real(scipy.fft.ifft(ddx_k * ddx_k_shift_pos * kappa * p_k)) )
        

        # # add in the velocity source term
        # if (k_sim.source_ux is not False and t_index < np.shape(source.ux)[1]):
        # #if options.source_ux >= t_index:
        #     match source.u_mode:
        #         case 'dirichlet':         
        #             # enforce the source values as a dirichlet boundary condition
        #             ux_sgx[k_sim.u_source_pos_index] = source.ux[k_sim.u_source_sig_index, t_index]
        #         case 'additive':
        #             # extract the source values into a matrix
        #             source_mat = np.zeros([kgrid.Nx, 1])
        #             source_mat[k_sim.u_source_pos_index] = source.ux[k_sim.u_source_sig_index, t_index]
        #             # apply the k-space correction
        #             source_mat = np.real(scipy.fft.ifft(source_kappa * scipy.fft.fft(source_mat)))
        #             # add the source values to the existing field values including the k-space correction
        #             ux_sgx = ux_sgx + source_mat
        #         case 'additive-no-correction':
        #             # add the source values to the existing field values        
        #             ux_sgx[k_sim.u_source_pos_index] = ux_sgx[k_sim.u_source_pos_index] + source.ux[k_sim.u_source_sig_index, t_index]
            

        # calculate du/dx at the next time step
        if not k_sim.nonuniform_grid and not options.use_finite_difference:
            # calculate gradient using the k-space method on a regular grid
            duxdx = np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_neg * kappa * scipy.fft.fftn(ux_sgx, axes=(0,)), axes=(0,) ) )

        elif options.use_finite_difference:
            match options.use_finite_difference:
                case 2:
                    # calculate gradient using second-order accurate finite difference scheme (including half step backward)
                    duxdx = (ux_sgx - np.append(ux_sgx[:-1], 0)) / kgrid.dx 
                    
                case 4:
                    # calculate gradient using fourth-order accurate finite difference scheme (including half step backward) 
                    duxdx = (np.append([0, 0], ux_sgx[:-2]) - 27.0 * np.append(0, ux_sgx[:-1]) + 27.0 * ux_sgx  - np.append(ux_sgx[1:], 0)) / (24.0 * kgrid.dx)
                                       
        else:      
            # calculate gradients using a non-uniform grid via the mapped
            # pseudospectral method
            duxdx = kgrid.dxudxn * np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_neg * kappa * scipy.fft.fftn(ux_sgx, axes=(0,)), axes=(0,)))


        # calculate rhox at the next time step
        if not k_sim.is_nonlinear:
            # use linearised mass conservation equation
            rhox = pml_x * (pml_x * rhox - dt * rho0 * duxdx)
        else:
            # use nonlinear mass conservation equation (explicit calculation)
            rhox = pml_x * (pml_x * rhox - dt * (2.0 * rhox + rho0) * duxdx)

        # add in the pre-scaled pressure source term as a mass source
        # if options.source_p >= t_index:
        # if (k_sim.source_p is not False and t_index < np.shape(source.p)[1]):
        #     match source.p_mode:
        #         case 'dirichlet':
        #             # enforce source values as a dirichlet boundary condition
        #             rhox[k_sim.p_source_pos_index] = source.p[k_sim.p_source_sig_index, t_index]
        #         case 'additive':
        #             # extract the source values into a matrix
        #             source_mat = np.zeros((kgrid.Nx, 1), dtype=myType)
        #             source_mat[k_sim.p_source_pos_index] = source.p[k_sim.p_source_sig_index, t_index]
        #             # apply the k-space correction
        #             source_mat = np.real(scipy.fft.ifft(source_kappa * scipy.fft.fft(source_mat)))
        #             # add the source values to the existing field values
        #             # including the k-space correction
        #             rhox = rhox + source_mat
        #         case 'additive-no-correction':
        #             # add the source values to the existing field values
        #             rhox[k_sim.p_source_pos_index] = rhox[k_sim.p_source_pos_index] + source.p[k_sim.p_source_sig_index, t_index]
            

        # equation of state
        if not k_sim.is_nonlinear:
            match k_sim.equation_of_state:
                case 'lossless':
                    # calculate p using a linear adiabatic equation of state
                    p = np.squeeze(c0**2) * np.squeeze(rhox)
                    
                case 'absorbing':
                    # calculate p using a linear absorbing equation of state
                    p = np.squeeze(c0**2 * (rhox
                        + medium.absorb_tau * np.real(scipy.fft.ifftn(medium.absorb_nabla1 * scipy.fft.fftn(rho0 * duxdx, axes=(0,)), axes=(0,) ))  
                        - medium.absorb_eta * np.real(scipy.fft.ifftn(medium.absorb_nabla2 * scipy.fft.fftn(rhox, axes=(0,)), axes=(0,))) ) )
    
                case 'stokes':
                    # calculate p using a linear absorbing equation of state
                    # assuming alpha_power = 2
                    p = c0**2 * (rhox + medium.absorb_tau * rho0 * duxdx)
                    
        else:
            match k_sim.equation_of_state:
                case 'lossless':
                    # calculate p using a nonlinear adiabatic equation of state
                    p = c0**2 * (rhox + medium.BonA * rhox**2 / (2.0 * rho0))
                    
                case 'absorbing':
                    # calculate p using a nonlinear absorbing equation of state
                    p = c0**2 * ( rhox
                        + medium.absorb_tau * np.real(scipy.fft.ifftn(medium.absorb_nabla1 * scipy.fft.fftn(rho0 * duxdx, axes=(0,)), axes=(0,)))
                        - medium.absorb_eta * np.real(scipy.fft.ifftn(medium.absorb_nabla2 * scipy.fft.fftn(rhox, axes=(0,)), axes=(0,)))
                        + medium.BonA * rhox**2 / (2.0 * rho0) )
                    
                case 'stokes':
                    # calculate p using a nonlinear absorbing equation of state
                    # assuming alpha_power = 2
                    p = c0**2 * (rhox
                        + medium.absorb_tau * rho0 * duxdx
                        + medium.BonA * rhox**2 / (2.0 * rho0) )
    
        # enforce initial conditions if source.p0 is defined instead of time varying sources
        if t_index == 0 and k_sim.source.p0 is not None:

            p0 = np.squeeze(k_sim.source.p0)


            # add the initial pressure to rho as a mass source
            p = p0
            rhox = p0 / np.squeeze(c0)**2

            # compute u(t = t1 - dt/2) based on u(dt/2) = -u(-dt/2) which forces u(t = t1) = 0
            if not options.use_finite_difference:
                
                # calculate gradient using the k-space method on a regular grid
                ux_sgx = dt * rho0_sgx_inv * np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_pos * kappa * scipy.fft.fftn(p, axes=(0,)), axes=(0,) )) / 2.0

                p_k = scipy.fft.fftn(p, axes=(0,))  
                p_k = np.squeeze(p_k)

            else:
                match options.use_finite_difference:
                    case 2:
                        # calculate gradient using second-order accurate finite difference scheme (including half step forward)            
                        dpdx =  (np.append(p[1:], 0.0) - p) / kgrid.dx
                        ux_sgx = dt * rho0_sgx_inv * dpdx / 2.0
                        
                    case 4:   
                        # calculate gradient using fourth-order accurate finite difference scheme (including half step backward)
                        dpdx = (np.append(p[2:], [0, 0]) - 27.0 * np.append(p[1:], 0) + 27.0 * p - np.append(0, p[:-1])) / (24.0 * kgrid.dx)
                        ux_sgx = dt * rho0_sgx_inv * dpdx / 2.0
                        
        
        else:
            # precompute fft of p here so p can be modified for visualisation
            p_k = scipy.fft.fftn(p, axes=(0,))  
            #p_k = p_k[:, np.newaxis] 
    
        # extract required sensor data from the pressure and particle velocity
        # fields if the number of time steps elapsed is greater than
        # sensor.record_start_index (defaults to 1) 
        if k_sim.use_sensor and (t_index >= sensor.record_start_index):
        
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
            
            # run sub-function to extract the required data from the acoustic variables
            sensor_data = extract_sensor_data(kgrid.dim, sensor_data, file_index, k_sim.sensor_mask_index, 
                                              extract_options, record, p, ux_sgx)



    return sensor_data




