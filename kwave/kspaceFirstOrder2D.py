from typing import Union

import numpy as np
from beartype import beartype as typechecker

from kwave.executor import Executor
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.kWaveSimulation import kWaveSimulation
from kwave.kWaveSimulation_helper import retract_transducer_grid_size, save_to_disk_func
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.dotdictionary import dotdict
from kwave.utils.interp import interpolate2d
from kwave.utils.pml import get_pml
from kwave.utils.tictoc import TicToc


@typechecker
def kspace_first_order_2d_gpu(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor, None],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
) -> Union[np.ndarray, dict]:
    """
    2D time-domain simulation of wave propagation on a GPU using C++ CUDA code.

    This function provides a blind interface to the C++/CUDA version of kspaceFirstOrder2D
    (called kspaceFirstOrder-CUDA). The function works by saving the input files to disk,
    running the C++/CUDA simulation, and loading the output files back into Python.

    Parameters:
    -----------
    kgrid : kWaveGrid
        Grid object containing Cartesian and k-space grid fields
    source : kSource
        Source object containing details of acoustic sources
    sensor : Union[NotATransducer, kSensor, None]
        Sensor object for recording the acoustic field
    medium : kWaveMedium
        Medium properties including sound speed, density, etc.
    simulation_options : SimulationOptions
        Simulation settings and flags
    execution_options : SimulationExecutionOptions
        Options controlling execution environment (CPU/GPU)

    Returns:
    --------
    Union[np.ndarray, dict]
        Either:
        - A numpy array containing the recorded sensor data if sensor.record is set
        - A dictionary containing simulation metadata if sensor.record is not set

    Notes:
    ------
    The GPU version uses the same binary for both 2D and 3D simulations.
    Required binaries are automatically downloaded and managed by k-wave-python.

    See Also:
    ---------
    kspaceFirstOrder2D : CPU version of this simulation function
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
    sensor: Union[NotATransducer, kSensor, None],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    2D time-domain simulation of wave propagation using C++ code.

    This function provides a blind interface to the C++ version of kspaceFirstOrder2D
    (called kspaceFirstOrder-OMP). The function works by saving the input files to disk,
    running the C++ simulation, and loading the output files back into Python.

    For small simulations, running the simulation on a smaller number of cores can improve
    performance as the matrices are often small enough to fit within cache. It is recommended
    to adjust the value of 'NumThreads' to optimize performance for a given simulation size
    and computer hardware. By default, simulations smaller than 128^2 are set to run using
    a single thread (this behavior can be over-ridden using the 'NumThreads' option).

    Parameters:
    -----------
    kgrid : kWaveGrid
        Grid object containing Cartesian and k-space grid fields
    source : kSource
        Source object containing details of acoustic sources
    sensor : Union[NotATransducer, kSensor, None]
        Sensor object for recording the acoustic field
    medium : kWaveMedium
        Medium properties including sound speed, density, etc.
    simulation_options : SimulationOptions
        Simulation settings and flags
    execution_options : SimulationExecutionOptions
        Options controlling execution environment (CPU/GPU)

    Returns:
    --------
    np.ndarray
        Recorded sensor data based on the sensor.record settings

    Notes:
    ------
    1. Required binaries are automatically downloaded and managed by k-wave-python
    2. The same binary is used for 2D, 3D, and axisymmetric simulations
    3. For very small simulations, the C++ code can be slower than the Python code

    See Also:
    ---------
    kspaceFirstOrder2D : Main simulation function
    """
    execution_options.is_gpu_simulation = False  # force to CPU
    # generate the input file and save to disk
    sensor_data = kspaceFirstOrder2D(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options, execution_options=execution_options
    )
    return sensor_data


@typechecker
def kspaceFirstOrder2D(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor, None],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    2D time-domain simulation of wave propagation using k-space pseudospectral method.

    This simulation function performs time-domain acoustic simulations in 2D homogeneous and
    heterogeneous media. The function is based on a k-space pseudospectral method where spatial
    derivatives are calculated using the Fourier collocation spectral method, and temporal
    derivatives are calculated using a k-space corrected finite-difference scheme.

    .. warning::
       The time reversal functionality (using sensor.time_reversal_boundary_data) is deprecated.
       Please use the :class:`TimeReversal` class from kwave.reconstruction instead.

    Key Features:
    ------------
    - Support for both homogeneous and heterogeneous media
    - Perfectly matched layer (PML) boundary conditions
    - Flexible source and sensor configurations
    - Support for absorption and nonlinearity
    - Time-varying source terms
    - Various sensor types (point, line)
    - Binary and Cartesian sensor masks
    - Recording of pressure, velocity, and intensity

    Parameters:
    -----------
    kgrid : kWaveGrid
        Grid object containing Cartesian and k-space grid fields
    source : kSource
        Source object containing details of acoustic sources
    sensor : Union[NotATransducer, kSensor, None]
        Sensor object for recording the acoustic field
    medium : kWaveMedium
        Medium properties including sound speed, density, etc.
    simulation_options : SimulationOptions
        Simulation settings and flags
    execution_options : SimulationExecutionOptions
        Options controlling execution environment (CPU/GPU)

    Returns:
    --------
    np.ndarray
        Recorded sensor data based on the sensor.record settings

    Notes:
    ------
    1. The simulation is based on coupled first-order equations for wave propagation.
    2. The time step is chosen based on the CFL stability criterion.
    3. For time reversal reconstruction, use the TimeReversal class from kwave.reconstruction.

    See Also:
    ---------
    kwave.reconstruction.TimeReversal : Class for time reversal image reconstruction
    kspaceFirstOrder3D : 3D version of this simulation function
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
        executor_options = execution_options.as_list(sensor=k_sim.sensor)
        sensor_data = executor.run_simulation(k_sim.options.input_filename, k_sim.options.output_filename, options=executor_options)
        return sensor_data
