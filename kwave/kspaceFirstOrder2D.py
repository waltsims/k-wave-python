from typing import Union

import numpy as np

from kwave.executor import Executor
from kwave.kWaveSimulation import kWaveSimulation
from kwave.kWaveSimulation_helper import retract_transducer_grid_size, save_to_disk_func
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.dotdictionary import dotdict
from kwave.utils.interp import interpolate2d
from kwave.utils.pml import get_pml
from kwave.utils.tictoc import TicToc


def kspace_first_order_2d_gpu(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: NotATransducer,
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
) -> np.ndarray:
    """
    2D ime-domain simulation of wave propagation on a GPU using C++ CUDA code.

    kspaceFirstOrder2DG provides a blind interface to the C++/CUDA
    version of kspaceFirstOrder2D (called kspaceFirstOrder-CUDA) in the
    same way as kspaceFirstOrder3DC. Note, the C++ code does not support
    all input options, and all display options are ignored (only command
    line outputs are given). See the k-Wave user manual for more
    information.
    The function works by appending the optional input 'save_to_disk' to
    the user inputs and then calling kspaceFirstOrder2D to save the input
    files to disk. The contents of sensor.record (if set) are parsed as
    input flags, and the C++ code is run using the system command. The
    output files are then automatically loaded from disk and returned in
    the same fashion as kspaceFirstOrder2D. The input and output files
    are saved to the temporary directory native to the operating system,
    and are deleted after the function runs.
    This function requires the C++ binary/executable of
    kspaceFirstOrder-CUDA to be downloaded from
    http://www.k-wave.org/download.php and placed in the "binaries"
    directory of the k-Wave toolbox (the 2D and 3D code use the same
    binary). Alternatively, the name and location of the binary can be
    specified using the optional input parameters 'BinaryName' and
    'BinariesPath'.

    This function is essentially a wrapper and directly uses the capabilities
    of kspaceFirstOrder3DC by replacing the binary name with the name of the
    GPU binary.
    """
    execution_options.is_gpu_simulation = True  # force to GPU
    assert isinstance(kgrid, kWaveGrid), "kgrid must be a kWaveGrid object"
    assert isinstance(medium, kWaveMedium), "medium must be a kWaveMedium object"
    assert isinstance(simulation_options, SimulationOptions), "simulation_options must be a SimulationOptions object"
    assert isinstance(execution_options, SimulationExecutionOptions), "execution_options must be a SimulationExecutionOptions object"

    sensor_data = kspaceFirstOrder2D(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options, execution_options=execution_options
    )  # pass inputs to CPU version
    return sensor_data


def kspaceFirstOrder2DC(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    2D time-domain simulation of wave propagation using C++ code.

    kspaceFirstOrder2DC provides a blind interface to the C++ version of
    kspaceFirstOrder2D (called kspaceFirstOrder-OMP) in the same way as
    kspaceFirstOrder3DC. Note, the C++ code does not support all input
    options, and all display options are ignored (only command line
    outputs are given). See the k-Wave user manual for more information.
    The function works by appending the optional input 'save_to_disk' to
    the user inputs and then calling kspaceFirstOrder2D to save the input
    files to disk. The contents of sensor.record (if set) are parsed as
    input flags, and the C++ code is run using the system command. The
    output files are then automatically loaded from disk and returned in
    the same fashion as kspaceFirstOrder2D. The input and output files
    are saved to the temporary directory native to the operating system,
    and are deleted after the function runs.
    For small simulations, running the simulation on a smaller number of
    cores can improve performance as the matrices are often small enough
    to fit within cache. It is recommended to adjust the value of
    'NumThreads' to optimise performance for a given simulation size and
    computer hardware. By default, simulations smaller than 128^2 are
    set to run using a single thread (this behaviour can be over-ridden
    using the 'NumThreads' option). In some circumstances, for very small
    simulations, the C++ code can be slower than the MATLAB code.
    This function requires the C++ binary/executable of
    kspaceFirstOrder-OMP to be downloaded from
    http://www.k-wave.org/download.php and placed in the "binaries"
    directory of the k-Wave toolbox (the same binary is used for
    simulations in 2D, 3D, and axisymmetric coordinates). Alternatively,
    the name and location of the binary can be  specified using the
    optional input parameters 'BinaryName' and 'BinariesPath'.

    This function is essentially a wrapper and directly uses the capabilities
    of kspaceFirstOrder3DC by replacing the binary name with the name of the
    GPU binary.

    Args:
        kgrid: kWaveGrid instance
        source: kWaveSource instance
        sensor: NotATransducer or kSensor instance or None
        medium: kWaveMedium instance
        simulation_options: SimulationOptions instance
        execution_options: SimulationExecutionOptions instance

    Returns:
        Sensor data as a numpy array
    """
    execution_options.is_gpu_simulation = False  # force to CPU
    # generate the input file and save to disk
    sensor_data = kspaceFirstOrder2D(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options, execution_options=execution_options
    )
    return sensor_data


def kspaceFirstOrder2D(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor, None],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    2D time-domain simulation of wave propagation.

    kspaceFirstOrder2D simulates the time-domain propagation of
    compressional waves through a two-dimensional homogeneous or
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
    source.ux and source.uy.

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

    kspaceFirstOrder2D may also be used for time reversal image
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

    Args:
        kgrid: kWaveGrid instance
        medium: kWaveMedium instance
        source: kWaveSource instance
        sensor: kWaveSensor instance or None

    Returns:

    """
    # start the timer and store the start time
    TicToc.tic()

    # Currently we only support binary execution, meaning all simulations must be saved to disk.
    if not simulation_options.save_to_disk:
        if execution_options.is_gpu_simulation:
            raise ValueError("GPU simulation requires saving to disk. Please set SimulationOptions.save_to_disk=True")
        else:
            raise ValueError("CPU simulation requires saving to disk. Please set SimulationOptions.save_to_disk=True")

    k_sim = kWaveSimulation(kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options)

    k_sim.input_checking("kspaceFirstOrder2D")

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================
    options = k_sim.options

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z + dz/2)
    k_sim.rho0 = np.atleast_1d(k_sim.rho0)
    if k_sim.rho0.ndim == 2 and options.use_sg:
        # rho0 is heterogeneous and staggered grids are used
        grid_points = [k_sim.kgrid.x, k_sim.kgrid.y]
        k_sim.rho0_sgx = interpolate2d(grid_points, k_sim.rho0, [k_sim.kgrid.x + k_sim.kgrid.dx / 2, k_sim.kgrid.y])
        k_sim.rho0_sgy = interpolate2d(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y + k_sim.kgrid.dy / 2])
    else:
        # rho0 is homogeneous or staggered grids are not used
        k_sim.rho0_sgx = k_sim.rho0
        k_sim.rho0_sgy = k_sim.rho0
    k_sim.rho0_sgz = None

    # invert rho0 so it doesn't have to be done each time step
    k_sim.rho0_sgx_inv = 1 / k_sim.rho0_sgx
    k_sim.rho0_sgy_inv = 1 / k_sim.rho0_sgy

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del k_sim.rho0_sgx
        del k_sim.rho0_sgy

    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the PML operators based on the reference sound speed and PML settings
    Nx, Ny = k_sim.kgrid.Nx, k_sim.kgrid.Ny
    dx, dy = k_sim.kgrid.dx, k_sim.kgrid.dy
    dt = k_sim.kgrid.dt
    pml_x_alpha, pml_y_alpha = options.pml_x_alpha, options.pml_y_alpha
    pml_x_size, pml_y_size = options.pml_x_size, options.pml_y_size
    c_ref = k_sim.c_ref

    k_sim.pml_x = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 1)
    k_sim.pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True and options.use_sg, 1)
    k_sim.pml_y = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 2)
    k_sim.pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True and options.use_sg, 2)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using ifftshift (the option
    # flgs.use_sg exists for debugging)
    kx_vec, ky_vec = k_sim.kgrid.k_vec
    kx_vec, ky_vec = np.array(kx_vec), np.array(ky_vec)
    if options.use_sg:
        k_sim.ddx_k_shift_pos = np.fft.ifftshift(1j * kx_vec * np.exp(1j * kx_vec * dx / 2))[None, :]
        k_sim.ddx_k_shift_neg = np.fft.ifftshift(1j * kx_vec * np.exp(-1j * kx_vec * dx / 2))[None, :]
        k_sim.ddy_k_shift_pos = np.fft.ifftshift(1j * ky_vec * np.exp(1j * ky_vec * dy / 2))[None, :]
        k_sim.ddy_k_shift_neg = np.fft.ifftshift(1j * ky_vec * np.exp(-1j * ky_vec * dy / 2))[None, :]
    else:
        k_sim.ddx_k_shift_pos = np.fft.ifftshift(1j * kx_vec)[None, :]
        k_sim.ddx_k_shift_neg = np.fft.ifftshift(1j * kx_vec)[None, :]
        k_sim.ddy_k_shift_pos = np.fft.ifftshift(1j * ky_vec)[None, :]
        k_sim.ddy_k_shift_neg = np.fft.ifftshift(1j * ky_vec)[None, :]

    # force the derivative and shift operators to be in the correct direction for use with BSXFUN
    k_sim.ddy_k_shift_pos = k_sim.ddy_k_shift_pos.T
    k_sim.ddy_k_shift_neg = k_sim.ddy_k_shift_neg.T

    # create k-space operators (the option flgs.use_kspace exists for debugging)
    if options.use_kspace:
        k = k_sim.kgrid.k
        k_sim.kappa = np.fft.ifftshift(np.sinc(c_ref * k * dt / 2))
        if (k_sim.source_p and k_sim.source.p_mode == "additive") or (
            (k_sim.source_ux or k_sim.source_uy or k_sim.source_uz) and k_sim.source.u_mode == "additive"
        ):
            k_sim.source_kappa = np.fft.ifftshift(np.cos(c_ref * k * dt / 2))
    else:
        k_sim.kappa = 1
        k_sim.source_kappa = 1

    # =========================================================================
    # SAVE DATA TO DISK FOR RUNNING SIMULATION EXTERNAL TO MATLAB
    # =========================================================================

    # save to disk option for saving the input matrices to disk for running
    # simulations using k-Wave++
    if options.save_to_disk:
        # store the pml size for resizing transducer object below
        retract_size = [[options.pml_x_size, options.pml_y_size, options.pml_z_size]]

        # run subscript to save files to disk
        save_to_disk_func(
            k_sim.kgrid,
            k_sim.medium,
            k_sim.source,
            k_sim.options,
            execution_options.auto_chunking,
            dotdict(
                {
                    "ddx_k_shift_pos": k_sim.ddx_k_shift_pos,
                    "ddx_k_shift_neg": k_sim.ddx_k_shift_neg,
                    "dt": k_sim.dt,
                    "c0": k_sim.c0,
                    "c_ref": k_sim.c_ref,
                    "rho0": k_sim.rho0,
                    "rho0_sgx": k_sim.rho0_sgx,
                    "rho0_sgy": k_sim.rho0_sgy,
                    "rho0_sgz": k_sim.rho0_sgz,
                    "p_source_pos_index": k_sim.p_source_pos_index,
                    "u_source_pos_index": k_sim.u_source_pos_index,
                    "s_source_pos_index": k_sim.s_source_pos_index,
                    "transducer_input_signal": k_sim.transducer_input_signal,
                    "delay_mask": k_sim.delay_mask,
                    "sensor_mask_index": k_sim.sensor_mask_index,
                    "record": k_sim.record,
                }
            ),
            dotdict(
                {
                    "source_p": k_sim.source_p,
                    "source_p0": k_sim.source_p0,
                    "source_ux": k_sim.source_ux,
                    "source_uy": k_sim.source_uy,
                    "source_uz": k_sim.source_uz,
                    "source_sxx": k_sim.source_sxx,
                    "source_syy": k_sim.source_syy,
                    "source_szz": k_sim.source_szz,
                    "source_sxy": k_sim.source_sxy,
                    "source_sxz": k_sim.source_sxz,
                    "source_syz": k_sim.source_syz,
                    "transducer_source": k_sim.transducer_source,
                    "nonuniform_grid": k_sim.nonuniform_grid,
                    "elastic_code": k_sim.options.simulation_type.is_elastic_simulation(),
                    "axisymmetric": k_sim.options.simulation_type.is_axisymmetric(),
                    "cuboid_corners": k_sim.cuboid_corners,
                }
            ),
        )

        # run subscript to resize the transducer object if the grid has been expanded
        retract_transducer_grid_size(k_sim.source, k_sim.sensor, retract_size, k_sim.options.pml_inside)

        # exit matlab computation if required
        if options.save_to_disk_exit:
            return

        executor = Executor(simulation_options=simulation_options, execution_options=execution_options)
        executor_options = execution_options.get_options_string(sensor=k_sim.sensor)
        sensor_data = executor.run_simulation(k_sim.options.input_filename, k_sim.options.output_filename, options=executor_options)
        return sensor_data
