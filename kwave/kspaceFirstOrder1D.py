import numpy as np
import scipy.fft
from tqdm import tqdm
from typing import Union

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kWaveSimulation import kWaveSimulation

from kwave.ktransducer import NotATransducer

from kwave.utils.data import scale_time
from kwave.utils.math import sinc
from kwave.utils.pml import get_pml
from kwave.utils.tictoc import TicToc
from kwave.utils.dotdictionary import dotdict

from kwave.options.simulation_options import SimulationOptions

from kwave.kWaveSimulation_helper import extract_sensor_data

def kspace_first_order_1D(kgrid: kWaveGrid,
                         source: kSource,
                         sensor: Union[NotATransducer, kSensor],
                         medium: kWaveMedium,
                         simulation_options: SimulationOptions,
                         verbose: bool = False):

    """
    KSPACEFIRSTORDER1D 1D time-domain simulation of wave propagation.

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

    INPUTS:
    The minimum fields that must be assigned to run an initial value problem
    (for example, a photoacoustic forward simulation) are marked with a *. 

        kgrid*                 - k-Wave grid object returned by kWaveGrid
                                containing Cartesian and k-space grid fields 
        kgrid.t_array*         - evenly spaced array of time values [s] (set
                                to 'auto' by kWaveGrid) 


        medium.sound_speed*    - sound speed distribution within the acoustic
                                medium [m/s] 
        medium.sound_speed_ref - reference sound speed used within the
                                k-space operator (phase correction term)
                                [m/s]
        medium.density*        - density distribution within the acoustic
                                medium [kg/m^3] 
        medium.BonA            - parameter of nonlinearity
        medium.alpha_power     - power law absorption exponent
        medium.alpha_coeff     - power law absorption coefficient 
                                [dB/(MHz^y cm)] 
        medium.alpha_mode      - optional input to force either the
                                absorption or dispersion terms in the
                                equation of state to be excluded; valid
                                inputs are 'no_absorption' or
                                'no_dispersion'  
        medium.alpha_filter    - frequency domain filter applied to the
                                absorption and dispersion terms in the
                                equation of state 
        medium.alpha_sign      - two element array used to control the sign
                                of absorption and dispersion terms in the
                                equation of state  


        source.p0*             - initial pressure within the acoustic medium
        source.p               - time varying pressure at each of the source
                                positions given by source.p_mask 
        source.p_mask          - binary matrix specifying the positions of
                                the time varying pressure source
                                distribution 
        source.p_mode          - optional input to control whether the input
                                pressure is injected as a mass source or
                                enforced as a dirichlet boundary condition;
                                valid inputs are 'additive' (the default) or
                                'dirichlet'    
        source.ux              - time varying particle velocity in the
                                x-direction at each of the source positions
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
                                    'u_rms' (RMS particle velocity)
                                    'u_final' (final particle velocity field at all grid points)
                                    'u_max_all' (maximum particle velocity at all grid points)
                                    'u_min_all' (minimum particle velocity at all grid points)
                                    'u_non_staggered' (particle velocity on non-staggered grid)
                                    'I' (time varying acoustic intensity)
                                    'I_avg' (average acoustic intensity) 
        sensor.record_start_index 
                                - time index at which the sensor should start
                                recording the data specified by 
                                sensor.record (default = 1) 
        sensor.time_reversal_boundary_data 
                                - time varying pressure enforced as a
                                Dirichlet boundary condition over
                                sensor.mask
        sensor.frequency_response 
                                - two element array specifying the center
                                frequency and percentage bandwidth of a
                                frequency domain Gaussian filter applied to
                                the sensor_data

    Note: For heterogeneous medium parameters, medium.sound_speed and
    medium.density must be given in matrix form with the same dimensions as
    kgrid. For homogeneous medium parameters, these can be given as single
    numeric values. If the medium is homogeneous and velocity inputs or
    outputs are not required, it is not necessary to specify medium.density.

    OPTIONAL INPUTS:
        Optional 'string', value pairs that may be used to modify the default
        computational settings. 

        'CartInterp'           - Interpolation mode used to extract the
                                pressure when a Cartesian sensor mask is
                                given. If set to 'nearest' and more than one
                                Cartesian point maps to the same grid point,
                                duplicated data points are discarded and
                                sensor_data will be returned with less
                                points than that specified by sensor.mask
                                (default = 'linear').  
        'CreateLog'            - Boolean controlling whether the command line
                                output is saved using the diary function
                                with a date and time stamped filename
                                (default = False).
        'DataCast'             - String input of the data type that variables
                                are cast to before computation. For example,
                                setting to 'single' will speed up the
                                computation time (due to the improved
                                efficiency of fftn and ifftn for this data
                                type) at the expense of a loss in precision. 
                                This variable is also useful for utilising
                                GPU parallelisation through libraries such
                                as the Parallel Computing Toolbox by setting
                                'DataCast' to 'gpuArray-single' (default =
                                'off').
        'DataRecast'           - Boolean controlling whether the output data
                                is cast back to double precision. If set to
                                False, sensor_data will be returned in the
                                data format set using the 'DataCast' option.
        'DisplayMask'          - Binary matrix overlaid onto the animated
                                simulation display. Elements set to 1 within
                                the display mask are set to black within the
                                display (default = sensor.mask).
        'LogScale'             - Boolean controlling whether the pressure
                                field is log compressed before display
                                (default = False). The data is compressed by
                                scaling both the positive and negative
                                values between 0 and 1 (truncating the data
                                to the given plot scale), adding a scalar
                                value (compression factor) and then using
                                the corresponding portion of a log10 plot
                                for the compression (the negative parts are
                                remapped to be negative thus the default
                                color scale will appear unchanged). The
                                amount of compression can be controlled by
                                adjusting the compression factor which can
                                be given in place of the Boolean input. The
                                closer the compression factor is to zero,
                                the steeper the corresponding part of the 
                                log10 plot used, and the greater the
                                compression (the default compression factor
                                is 0.02).
        'MovieArgs'            - Settings for VideoWriter. Parameters must be
                                given as {'param', value, ...} pairs within
                                a cell array (default = {}), where 'param'
                                corresponds to a writable property of a
                                VideoWriter object. 
        'MovieName'            - Name of the movie produced when
                                'RecordMovie' is set to true (default =
                                'date-time-kspaceFirstOrder1D'). 
        'MovieProfile'         - Profile input passed to VideoWriter.
        'PlotFreq'             - The number of iterations which must pass 
                                before the simulation plot is updated
                                (default = 10).
        'PlotLayout'           - Boolean controlling whether a four panel
                                plot of the initial simulation layout is
                                produced (initial pressure, sensor mask,
                                sound speed, density) (default = False).
        'PlotPML'              - Boolean controlling whether the perfectly
                                matched layer is shown in the simulation
                                plots. If set to False, the PML is not
                                displayed (default = true). 
        'PlotScale'            - [min, max] values used to control the
                                scaling for imagesc (visualisation). If set
                                to 'auto', a symmetric plot scale is chosen
                                automatically for each plot frame.
        'PlotSim'              - Boolean controlling whether the simulation
                                iterations are progressively plotted
                                (default = true). 
        'PMLAlpha'             - Absorption within the perfectly matched 
                                layer in Nepers per grid point (default =
                                2).
        'PMLInside'            - Boolean controlling whether the perfectly 
                                matched layer is inside or outside the grid.
                                If set to False, the input grids are
                                enlarged by PMLSize before running the
                                simulation (default = true).
        'PMLSize'              - Size of the perfectly matched layer in grid
                                points. To remove the PML, set the
                                appropriate PMLAlpha to zero rather than
                                forcing the PML to be of zero size (default
                                = 20).
        'RecordMovie'          - Boolean controlling whether the displayed
                                image frames are captured and stored as a
                                movie using VideoWriter (default = False).
        'Smooth'               - Boolean controlling whether source.p0,
                                medium.sound_speed, and medium.density are
                                smoothed using smooth before computation.
                                'Smooth' can either be given as a single
                                Boolean value or as a 3 element array to
                                control the smoothing of source.p0,
                                medium.sound_speed, and medium.density,
                                independently (default = [true, False,
                                False]).

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
        sensor_data.ux_max     - maximum particle velocity in the x-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_max' is set)   
        sensor_data.ux_min     - minimum particle velocity in the x-direction
                                recorded at the sensor positions given by
                                sensor.mask (returned if 'u_min' is set)   
        sensor_data.ux_rms     - rms of the time varying particle velocity in
                                the x-direction recorded at the sensor
                                positions given by sensor.mask (returned if
                                'u_rms' is set)     
        sensor_data.ux_final   - final particle velocity field in the
                                x-direction at all grid points within the
                                domain (returned if 'u_final' is set) 
        sensor_data.ux_max_all - maximum particle velocity in the x-direction
                                recorded at all grid points within the
                                domain (returned if 'u_max_all' is set)
        sensor_data.ux_min_all - minimum particle velocity in the x-direction
                                recorded at all grid points within the
                                domain (returned if 'u_min_all' is set)   
        sensor_data.ux_non_staggered 
                                - time varying particle velocity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask after shifting to the
                                non-staggered grid (returned if
                                'u_non_staggered' is set)
        sensor_data.Ix         - time varying acoustic intensity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I' is
                                set)
        sensor_data.Ix_avg     - average acoustic intensity in the
                                x-direction recorded at the sensor positions
                                given by sensor.mask (returned if 'I_avg' is
                                set)

    ABOUT:
        author                 - Bradley Treeby and Ben Cox
        date                   - 22nd April 2009
        last update            - 25th July 2019
        
    This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    Copyright (C) 2009-2019 Bradley Treeby and Ben Cox

    See also kspaceFirstOrderAS, kspaceFirstOrder2D, kspaceFirstOrder3D,
    kWaveGrid, kspaceSecondOrder

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
    k_sim.input_checking("kspaceFirstOrder1D")

    # aliases from simulation
    sensor_data = k_sim.sensor_data
    options = k_sim.options
    record = k_sim.record

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2)
    rho0 = k_sim.rho0
    m_rho0: int = np.squeeze(rho0).ndim

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

    rho0_sgx_inv = rho0_sgx_inv[:, np.newaxis]

    # clear unused variables
    # del rho0_sgx

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
    pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 0).T
    pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True,  0).T

    # define the k-space derivative operator
    ddx_k = scipy.fft.ifftshift(1j * kx_vec)
    ddx_k = ddx_k[:, np.newaxis]

    # define the staggered grid shift operators (the option options.use_sg exists for debugging)
    if options.use_sg:
        ddx_k_shift_pos = scipy.fft.ifftshift( np.exp( 1j * kx_vec * dx / 2.0))
        ddx_k_shift_neg = scipy.fft.ifftshift( np.exp(-1j * kx_vec * dx / 2.0))
        ddx_k_shift_pos = ddx_k_shift_pos[:, np.newaxis]
        ddx_k_shift_neg = ddx_k_shift_neg[:, np.newaxis]
    else:
        ddx_k_shift_pos = 1.0
        ddx_k_shift_neg = 1.0 
    

    # create k-space operator (the option options.use_kspace exists for debugging)
    if options.use_kspace:
        kappa        = scipy.fft.ifftshift(sinc(c_ref * kgrid.k * kgrid.dt / 2.0))
        kappa        = kappa[:, np.newaxis]
        if (hasattr(options, 'source_p') and hasattr(k_sim.source, 'p_mode')) and (k_sim.source.p_mode == 'additive') or \
           (hasattr(options, 'source_ux') and hasattr(k_sim.source, 'u_mode')) and (k_sim.source.u_mode == 'additive'):
            source_kappa = scipy.fft.ifftshift(np.cos (c_ref * kgrid.k * kgrid.dt / 2.0))
            source_kappa = source_kappa[:, np.newaxis]
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

    grid_shape = (Nx, 1)

    # preallocate the loop variables
    p      = np.zeros(grid_shape, dtype=myType)
    rhox   = np.zeros(grid_shape, dtype=myType)
    ux_sgx = np.zeros(grid_shape, dtype=myType)
    p_k    = np.zeros(grid_shape, dtype=myType)

    c0    = c0.astype(myType)

    verbose: bool = False

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

        # reverse the order of the input data
        sensor.time_reversal_boundary_data = np.fliplr(sensor.time_reversal_boundary_data)  
        index_start = 0
        index_step = 0

        # stop one time point before the end so the last points are not
        # propagated
        index_end = kgrid.Nt - 1  

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

    # # =========================================================================
    # # PREPARE VISUALISATIONS
    # # =========================================================================

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
        
        # print("0.", np.shape(p))

        # enforce time reversal bounday condition
        # if options.time_rev:       
        #     # load pressure value and enforce as a Dirichlet boundary condition
        #     p[k_sim.sensor_mask_index] = sensor.time_reversal_boundary_data[:, t_index]
        #     # update p_k
        #     p_k = scipy.fft.fft(p)
        #     # compute rhox using an adiabatic equation of state
        #     rhox_mod = p / c0**2
        #     rhox[k_sim.sensor_mask_index] = rhox_mod[k_sim.sensor_mask_index]
        
    
        # print("1.", np.shape(p))

        # calculate ux at the next time step using dp/dx at the current time step
        if not options.nonuniform_grid and not options.use_finite_difference:

            if verbose:
                print("Here 1-----.", np.shape(pml_x), np.shape(pml_x_sgx), np.shape(ux_sgx), 
                      '......', np.shape(ddx_k), np.shape(ddx_k_shift_pos), np.shape(kappa), 
                      ',,,,,,', np.shape(p_k), np.shape(rho0_sgx_inv))



            # calculate gradient using the k-space method on a regular grid
            ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - 
                                  dt * rho0_sgx_inv * np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_pos * kappa * p_k, axes=(0,))))
        
        elif options.use_finite_difference:
            print("\t\tEXIT! options.use_finite_difference")
            match options.use_finite_difference:
                case 2:
                    
                    # calculate gradient using second-order accurate finite
                    # difference scheme (including half step forward)
                    dpdx =  (np.append(p[1:], 0.0) - p) / kgrid.dx
                    # dpdx = ([p(2:end); 0] - p) / kgrid.dx;    
                    ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - dt * rho0_sgx_inv * dpdx )
                    
                case 4:
                    
                    # calculate gradient using fourth-order accurate finite
                    # difference scheme (including half step forward)
                    # dpdx = ([0; p(1:(end-1))] - 27*p + 27*[p(2:end); 0] - [p(3:end); 0; 0])/(24*kgrid.dx);
                    dpdx = (np.insert(p[:-1], 0, 0) - 27.0 * p + 27 * np.append(p[1:], 0.0) - np.append(p[2:], [0, 0])) / (24.0 * kgrid.dx)
                    ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - dt * rho0_sgx_inv * dpdx )
                    
        else:
            print("\t\tEXIT! else", )
            
            # calculate gradient using the k-space method on a non-uniform grid
            # via the mapped pseudospectral method         
            ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - 
                                dt * rho0_sgx_inv * k_sim.kgrid.dxudxn_sgx * np.real(scipy.fft.ifft(ddx_k * ddx_k_shift_pos * kappa * p_k)) )
        
 
        # print("2.", np.shape(p))

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
            
   
        # print("3.", np.shape(p))

        # calculate du/dx at the next time step
        if not options.nonuniform_grid and not options.use_finite_difference:

            if verbose:
                print("Here 1.", np.shape(p), np.shape(ux_sgx), np.shape(ddx_k), np.shape(ddx_k_shift_neg), np.shape(kappa))

            # calculate gradient using the k-space method on a regular grid
            duxdx = np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_neg * kappa * scipy.fft.fftn(ux_sgx, axes=(0,)), axes=(0,) ) )

            if verbose:
                print("Here 1(end). duxdx:", np.shape(duxdx))

        elif options.use_finite_difference:
            print("\t\tEXIT! options.use_finite_difference")
            match options.use_finite_difference:
                case 2:
                    
                    # calculate gradient using second-order accurate finite difference scheme (including half step backward)
                    # duxdx = (ux_sgx - [0; ux_sgx(1:end - 1)]) / kgrid.dx; 
                    duxdx = (ux_sgx - np.append(ux_sgx[:-1], 0)) / kgrid.dx 
                    
                case 4:
                    
                    # calculate gradient using fourth-order accurate finite difference scheme (including half step backward) 
                    duxdx = (np.append([0, 0], ux_sgx[:-2]) - 27.0 * np.append(0, ux_sgx[:-1]) + 27.0 * ux_sgx  - np.append(ux_sgx[1:], 0)) / (24.0 * kgrid.dx)
                    # duxdx = ([0; 0; ux_sgx(1:(end - 2))]   - 27 * [0; ux_sgx(1:(end - 1))]     + 27 * ux_sgx    - [ux_sgx(2:end); 0]) / (24 * kgrid.dx);
                    
        else:      
            # calculate gradients using a non-uniform grid via the mapped
            # pseudospectral method
            duxdx = kgrid.dxudxn * np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_neg * kappa * scipy.fft.fftn(ux_sgx, axes=(0,)), axes=(0,)))


        # print("4.", np.shape(p))

        # calculate rhox at the next time step
        if not k_sim.is_nonlinear:
            # use linearised mass conservation equation

            if verbose:
                print("pre:", pml_x.shape, rhox.shape, duxdx.shape, dt, rho0.shape)

            rhox = pml_x * (pml_x * rhox - dt * rho0 * duxdx)

            if verbose:
                print("post:", pml_x.shape, rhox.shape, duxdx.shape, dt, rho0.shape)

        else:
            # use nonlinear mass conservation equation (explicit calculation)
            rhox = pml_x * (pml_x * rhox - dt * (2.0 * rhox + rho0) * duxdx)

        # print("5.", np.shape(p))

        # add in the pre-scaled pressure source term as a mass source
        # if options.source_p >= t_index:
        # if (k_sim.source_p is not False and t_index < np.shape(source.p)[1]):
        #     print("??????????")
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
            
    
        # print("6.", np.shape(p))

        # equation of state
        if not k_sim.is_nonlinear:
            # print("is linear", k_sim.equation_of_state, type(k_sim.equation_of_state))
            match k_sim.equation_of_state:
                case 'lossless':

                    # print("Here 2. lossless / linear",  np.shape(p))
                    
                    # calculate p using a linear adiabatic equation of state
                    p = np.squeeze(c0**2) * np.squeeze(rhox)

                    # print("3.", np.shape(p), np.squeeze(c0**2).shape,  np.squeeze(rhox).shape)
                    
                case 'absorbing':

                    # print("Here 2. absorbing / linear",  np.shape(p))

                    # calculate p using a linear absorbing equation of state
                    p = np.squeeze(c0**2 * (rhox
                        + medium.absorb_tau * np.real(scipy.fft.ifftn(medium.absorb_nabla1 * scipy.fft.fftn(rho0 * duxdx, axes=(0,)), axes=(0,) ))  
                        - medium.absorb_eta * np.real(scipy.fft.ifftn(medium.absorb_nabla2 * scipy.fft.fftn(rhox, axes=(0,)), axes=(0,))) ) )

                    
                case 'stokes':
                    
                    # print("Here 2. stokes / linear")

                    # calculate p using a linear absorbing equation of state
                    # assuming alpha_power = 2
                    p = c0**2 * (rhox + medium.absorb_tau * rho0 * duxdx)
                    

        else:
            match k_sim.equation_of_state:
                case 'lossless':

                    print("Here 2. lossless / nonlinear")
                    
                    # calculate p using a nonlinear adiabatic equation of state
                    p = c0**2 * (rhox + medium.BonA * rhox**2 / (2.0 * rho0))
                    
                case 'absorbing':

                    print("Here 2. absorbing / nonlinear")

                    # calculate p using a nonlinear absorbing equation of state
                    p = c0**2 * ( rhox
                        + medium.absorb_tau * np.real(scipy.fft.ifftn(medium.absorb_nabla1 * scipy.fft.fftn(rho0 * duxdx, axes=(0,)), axes=(0,)))
                        - medium.absorb_eta * np.real(scipy.fft.ifftn(medium.absorb_nabla2 * scipy.fft.fftn(rhox, axes=(0,)), axes=(0,)))
                        + medium.BonA * rhox**2 / (2.0 * rho0) )
                    
                case 'stokes':

                    print("Here 2. stokes / nonlinear")
                    
                    # calculate p using a nonlinear absorbing equation of state
                    # assuming alpha_power = 2
                    p = c0**2 * (rhox
                        + medium.absorb_tau * rho0 * duxdx
                        + medium.BonA * rhox**2 / (2.0 * rho0) )
                
  
        # print("7.", np.shape(p), k_sim.source.p0.shape)
    
        # enforce initial conditions if source.p0 is defined instead of time varying sources
        if t_index == 0 and k_sim.source_p0:

            # print(np.shape(rhox))

            if k_sim.source.p0.ndim == 1:
                p0 = k_sim.source.p0[:, np.newaxis]
            else:
                p0 = k_sim.source.p0

            # add the initial pressure to rho as a mass source
            p = p0
            rhox = p0 / c0**2

            # compute u(t = t1 - dt/2) based on u(dt/2) = -u(-dt/2) which forces u(t = t1) = 0
            if not options.use_finite_difference:
                
                # calculate gradient using the k-space method on a regular grid
                ux_sgx = dt * rho0_sgx_inv * np.real(scipy.fft.ifftn(ddx_k * ddx_k_shift_pos * kappa * scipy.fft.fftn(p, axes=(0,)), axes=(0,) )) / 2.0

                p_k = scipy.fft.fftn(p, axes=(0,))  


            else:
                match options.use_finite_difference:
                    case 2:
                        
                        # calculate gradient using second-order accurate finite difference scheme (including half step forward)
                        # dpdx = ([p(2:end); 0] - p) / kgrid.dx;
            
                        dpdx =  (np.append(p[1:], 0.0) - p) / kgrid.dx
                        ux_sgx = dt * rho0_sgx_inv * dpdx / 2.0
                        
                    case 4:
                        
                        # calculate gradient using fourth-order accurate finite difference scheme (including half step backward)
                        # dpdx = ([p(3:end); 0; 0] - 27 * [p(2:end); 0] + 27 * p - [0; p(1:(end-1))]) / (24 * kgrid.dx)
                        dpdx = (np.append(p[2:], [0, 0]) - 27.0 * np.append(p[1:], 0) + 27.0 * p - np.append(0, p[:-1])) / (24.0 * kgrid.dx)
                        ux_sgx = dt * rho0_sgx_inv * dpdx / 2.0
                        
        
        else:
            # precompute fft of p here so p can be modified for visualisation
            p_k = scipy.fft.fftn(p, axes=(0,))  
            p_k = p_k[:, np.newaxis] 
    
        # extract required sensor data from the pressure and particle velocity
        # fields if the number of time steps elapsed is greater than
        # sensor.record_start_index (defaults to 1) 
        if options.use_sensor and not options.time_rev and (t_index >= sensor.record_start_index):
        
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
            sensor_data = extract_sensor_data(kgrid.dim, sensor_data, file_index, k_sim.sensor_mask_index, extract_options, record, p, ux_sgx)



    return sensor_data




