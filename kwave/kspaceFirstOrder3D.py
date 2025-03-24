from typing import Any, Dict, Union

import numpy as np
from deprecated import deprecated

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
from kwave.utils.interp import interpolate3d
from kwave.utils.pml import get_pml
from kwave.utils.tictoc import TicToc


def kspaceFirstOrder3DG(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    3D time-domain simulation of wave propagation on a GPU using C++ CUDA code.

    kspaceFirstOrder3DG provides a blind interface to the C++/CUDA
    version of kspaceFirstOrder3D (called kspaceFirstOrder-CUDA) in the
    same way as kspaceFirstOrder3DC. Note, the C++ code does not support
    all input options, and all display options are ignored (only command
    line outputs are given). See the k-Wave user manual for more
    information.

    The function works by appending the optional input 'save_to_disk' to
    the user inputs and then calling kspaceFirstOrder3D to save the input
    files to disk. The contents of sensor.record (if set) are parsed as
    input flags, and the C++ code is run using the system command. The
    output files are then automatically loaded from disk and returned in
    the same fashion as kspaceFirstOrder3D. The input and output files
    are saved to the temporary directory native to the operating system,
    and are deleted after the function runs.

    This function requires the C++ binary/executable of
    kspaceFirstOrder-CUDA to be downloaded from
    http://www.k-wave.org/download.php and placed in the "binaries"
    directory of the k-Wave toolbox. Alternatively, the name and location
    of the binary can be specified using the optional input parameters
    'BinaryName' and 'BinariesPath'.

    This function is essentially a wrapper and directly uses the capabilities
    of kspaceFirstOrder3DC by replacing the binary name with the name of the
    GPU binary.

    Args:
        **kwargs:

    Returns:

    """
    execution_options.is_gpu_simulation = True
    assert execution_options.is_gpu_simulation, "kspaceFirstOrder2DG can only be used for GPU simulations"
    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options, execution_options=execution_options
    )  # pass inputs to CPU version
    return sensor_data


def kspaceFirstOrder3DC(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    3D time-domain simulation of wave propagation using C++ code.

    kspaceFirstOrder3DC provides a blind interface to the C++ version of
    kspaceFirstOrder3D (called kspaceFirstOrder-OMP). Note, the C++ code
    does not support all input options, and all display options are
    ignored (only command line outputs are given). See the k-Wave user
    manual for more information.

    The function works by appending the optional input 'save_to_disk' to
    the user inputs and then calling kspaceFirstOrder3D to save the input
    files to disk. The contents of sensor.record (if set) are parsed as
    input flags, and the C++ code is run using the system command. The
    output files are then automatically loaded from disk and returned in
    the same fashion as kspaceFirstOrder3D. The input and output files
    are saved to the temporary directory native to the operating system,
    and are deleted after the function runs.

    This function is not recommended for large simulations, as the input
    variables will reside twice in main memory (once in MATLAB, and once
    in C++). For large simulations, the C++ code should be called outside
    of MATLAB. See the k-Wave manual for more information.

    This function requires the C++ binary/executable of
    kspaceFirstOrder-OMP to be downloaded from
    http://www.k-wave.org/download.php and placed in the "binaries"
    directory of the k-Wave toolbox (the same binary is used for
    simulations in 2D, 3D, and axisymmetric coordinates). Alternatively,
    the name and  location of the binary can be specified using the
    optional input parameters 'BinaryName' and 'BinariesPath'.

    Args:
        **kwargs:

    Returns:

    """
    execution_options.is_gpu_simulation = False
    # generate the input file and save to disk
    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options, execution_options=execution_options
    )
    return sensor_data


@deprecated(version="0.4.1", reason="Use TimeReversal class instead. This parameter will be removed in v0.5.", action="once")
def kspaceFirstOrder3D(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
    time_rev: bool = False,  # deprecated parameter
) -> Dict[str, Any]:
    """
    DEPRECATED: Use TimeReversal class instead.

    The time_rev parameter will be removed in v0.5. Please migrate to the new TimeReversal class:

    from kwave.reconstruction.time_reversal import TimeReversal
    tr = TimeReversal(kgrid, medium, sensor)
    p0_recon = tr(kspaceFirstOrder3D, simulation_options, execution_options)
    """
    if time_rev:
        import warnings

        warnings.warn("The time_rev parameter is deprecated. Use the TimeReversal class instead.", DeprecationWarning, stacklevel=2)
    # start the timer and store the start time
    TicToc.tic()

    # Currently we only support binary execution, meaning all simulations must be saved to disk.
    if not simulation_options.save_to_disk:
        if execution_options.is_gpu_simulation:
            raise ValueError("GPU simulation requires saving to disk. Please set SimulationOptions.save_to_disk=True")
        else:
            raise ValueError("CPU simulation requires saving to disk. Please set SimulationOptions.save_to_disk=True")

    k_sim = kWaveSimulation(kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options)
    k_sim.input_checking("kspaceFirstOrder3D")

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================
    options = k_sim.options

    # TODO(walter): this could all be moved inside of ksim

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z + dz/2)
    k_sim.rho0 = np.atleast_1d(k_sim.rho0)
    if k_sim.rho0.ndim == 3 and options.use_sg:
        # rho0 is heterogeneous and staggered grids are used
        grid_points = [k_sim.kgrid.x, k_sim.kgrid.y, k_sim.kgrid.z]
        k_sim.rho0_sgx = interpolate3d(grid_points, k_sim.rho0, [k_sim.kgrid.x + k_sim.kgrid.dx / 2, k_sim.kgrid.y, k_sim.kgrid.z])
        k_sim.rho0_sgy = interpolate3d(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y + k_sim.kgrid.dy / 2, k_sim.kgrid.z])
        k_sim.rho0_sgz = interpolate3d(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y, k_sim.kgrid.z + k_sim.kgrid.dz / 2])
    else:
        # rho0 is homogeneous or staggered grids are not used
        k_sim.rho0_sgx = k_sim.rho0
        k_sim.rho0_sgy = k_sim.rho0
        k_sim.rho0_sgz = k_sim.rho0

    # invert rho0 so it doesn't have to be done each time step
    k_sim.rho0_sgx_inv = 1 / k_sim.rho0_sgx
    k_sim.rho0_sgy_inv = 1 / k_sim.rho0_sgy
    k_sim.rho0_sgz_inv = 1 / k_sim.rho0_sgz

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del k_sim.rho0_sgx
        del k_sim.rho0_sgy
        del k_sim.rho0_sgz

    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the PML operators based on the reference sound speed and PML settings
    Nx, Ny, Nz = k_sim.kgrid.Nx, k_sim.kgrid.Ny, k_sim.kgrid.Nz
    dx, dy, dz = k_sim.kgrid.dx, k_sim.kgrid.dy, k_sim.kgrid.dz
    dt = k_sim.kgrid.dt
    pml_x_alpha, pml_y_alpha, pml_z_alpha = options.pml_x_alpha, options.pml_y_alpha, options.pml_z_alpha
    pml_x_size, pml_y_size, pml_z_size = options.pml_x_size, options.pml_y_size, options.pml_z_size
    c_ref = k_sim.c_ref

    k_sim.pml_x = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 1)
    k_sim.pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True and options.use_sg, 1)
    k_sim.pml_y = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 2)
    k_sim.pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True and options.use_sg, 2)
    k_sim.pml_z = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, False, 3)
    k_sim.pml_z_sgz = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, True and options.use_sg, 3)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using ifftshift (the option
    # flgs.use_sg exists for debugging)
    kx_vec, ky_vec, kz_vec = k_sim.kgrid.k_vec
    kx_vec, ky_vec, kz_vec = np.array(kx_vec), np.array(ky_vec), np.array(kz_vec)
    if options.use_sg:
        k_sim.ddx_k_shift_pos = np.fft.ifftshift(1j * kx_vec * np.exp(1j * kx_vec * dx / 2)).T
        k_sim.ddx_k_shift_neg = np.fft.ifftshift(1j * kx_vec * np.exp(-1j * kx_vec * dx / 2)).T
        k_sim.ddy_k_shift_pos = np.fft.ifftshift(1j * ky_vec * np.exp(1j * ky_vec * dy / 2)).T
        k_sim.ddy_k_shift_neg = np.fft.ifftshift(1j * ky_vec * np.exp(-1j * ky_vec * dy / 2)).T
        k_sim.ddz_k_shift_pos = np.fft.ifftshift(1j * kz_vec * np.exp(1j * kz_vec * dz / 2)).T
        k_sim.ddz_k_shift_neg = np.fft.ifftshift(1j * kz_vec * np.exp(-1j * kz_vec * dz / 2)).T
    else:
        k_sim.ddx_k_shift_pos = np.fft.ifftshift(1j * kx_vec).T
        k_sim.ddx_k_shift_neg = np.fft.ifftshift(1j * kx_vec).T
        k_sim.ddy_k_shift_pos = np.fft.ifftshift(1j * ky_vec).T
        k_sim.ddy_k_shift_neg = np.fft.ifftshift(1j * ky_vec).T
        k_sim.ddz_k_shift_pos = np.fft.ifftshift(1j * kz_vec).T
        k_sim.ddz_k_shift_neg = np.fft.ifftshift(1j * kz_vec).T

    # force the derivative and shift operators to be in the correct direction for use with BSXFUN
    k_sim.ddy_k_shift_pos = k_sim.ddy_k_shift_pos.T
    k_sim.ddy_k_shift_neg = k_sim.ddy_k_shift_neg.T

    ddz_k_shift_pos = k_sim.ddz_k_shift_pos  # N x 1
    ddz_k_shift_pos = np.expand_dims(ddz_k_shift_pos, axis=-1).transpose((1, 2, 0))
    k_sim.ddz_k_shift_pos = ddz_k_shift_pos

    ddz_k_shift_neg = k_sim.ddz_k_shift_neg  # N x 1
    ddz_k_shift_neg = np.expand_dims(ddz_k_shift_neg, axis=-1).transpose((1, 2, 0))
    k_sim.ddz_k_shift_neg = ddz_k_shift_neg

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
