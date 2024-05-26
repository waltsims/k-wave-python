from typing import Union
import logging

import numpy as np
from numpy.fft import ifftshift

from kwave.kgrid import kWaveGrid
from kwave.enums import DiscreteCosine
from kwave.executor import Executor
from kwave.kWaveSimulation import kWaveSimulation
from kwave.kWaveSimulation_helper import retract_transducer_grid_size, save_to_disk_func
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.dotdictionary import dotdict
from kwave.utils.interp import interpolate2d
from kwave.utils.math import sinc
from kwave.utils.matrix import num_dim2
from kwave.utils.pml import get_pml
from kwave.utils.tictoc import TicToc


def kspaceFirstOrderASC(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    Axisymmetric time-domain simulation of wave propagation using C++ code.

    kspaceFirstOrderASC provides a blind interface to the C++ version of
    kspaceFirstOrderAS (called kspaceFirstOrder-OMP) in the same way as
    kspaceFirstOrder3DC. Note, the C++ code does not support all input
    options, and all display options are ignored (only command line
    outputs are given). See the k-Wave user manual for more information.

    The function works by appending the optional input 'save_to_disk' to
    the user inputs and then calling kspaceFirstOrderAS to save the input
    files to disk. The contents of sensor.record (if set) are parsed as
    input flags, and the C++ code is run using the system command. The
    output files are then automatically loaded from disk and returned in
    the same fashion as kspaceFirstOrderAS. The input and output files
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
    the name and location  of the binary can be specified using the
    optional input parameters 'BinaryName' and 'BinariesPath'.

    This function is essentially a wrapper and directly uses the capabilities
    of kspaceFirstOrder3DC by replacing the binary name with the name of the
    GPU binary.

    Args:
        **kwargs:

    Returns:
    """
    # generate the input file and save to disk
    sensor_data = kspaceFirstOrderAS(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options, execution_options=execution_options
    )
    return sensor_data


def kspaceFirstOrderAS(
    kgrid: kWaveGrid,
    source: kSource,
    sensor: Union[NotATransducer, kSensor],
    medium: kWaveMedium,
    simulation_options: SimulationOptions,
    execution_options: SimulationExecutionOptions,
):
    """
    Axisymmetric time-domain simulation of wave propagation.

    kspaceFirstOrderAS simulates the time-domain propagation of
    compressional waves through an axisymmetric homogeneous or
    heterogeneous acoustic medium. The code is functionally very similar
    to kspaceFirstOrder2D. However, a 2D axisymmetric coordinate system
    is used instead of a 2D Cartesian coordinate system. In this case, x
    corresponds to the axial dimension, and y corresponds to the radial
    dimension. In the radial dimension, the first grid point corresponds
    to the grid origin, i.e., y = 0. In comparison, for
    kspaceFirstOrder2D, the Cartesian point y = 0 is in the middle of the
    computational grid.

    The input structures kgrid, medium, source, and sensor are defined in
    exactly the same way as for kspaceFirstOrder2D. However,
    computationally, there are several key differences. First, the
    axisymmetric code solves the coupled first-order equations accounting
    for viscous absorption (not power law), so only medium.alpha_power =
    2 is supported. This value is set by default, and doesn't need to be
    defined. This also means that medium.alpha_mode and
    medium.alpha_filter are not supported. Second, for a homogeneous
    medium, the k-space correction used to counteract the numerical
    dispersion introduced by the finite-difference time step is not exact
    (as it is for the other fluid codes). However, the approximate
    k-space correction still works very effectively, so dispersion errors
    should still be small. See kspaceFirstOrder2D for additional details
    on the function inputs.

    In the x-dimension (axial), the FFT is used to compute spatial
    gradients. In the y-dimension (radial), two choices of symmetry are
    possible. These are whole-sample-symmetric on the interior radial
    boundary (y = 0) and either whole-sample-symmetric or
    whole-sample-asymmetric on the exterior radial boundary. These are
    abbreviated WSWA and WSWS. The WSWA and WSWS symmetries are
    implemented using both discrete trigonometric transforms (DTTs), and
    via the FFT by manually mirroring the domain. The latter options are
    abbreviated as WSWA-FFT and WSWS-FFT. The WSWA/WSWS options and the
    corresponding WSWA-FFT/WSWS-FFT options agree to machine precision.
    When using the PML, the choice of symmetry doesn't matter, and all
    options give very similar results (to several decimal places).
    Computationally, the DTT implementations are more efficient, but
    require additional compiled MATLAB functions (not currently part of
    k-Wave). The symmetry can be set by using the optional input
    'RadialSymmetry'. The WSWA-FFT symmetry is set by default.

    Note: For heterogeneous medium parameters, medium.sound_speed and
    medium.density must be given in matrix form with the same dimensions as
    kgrid. For homogeneous medium parameters, these can be given as single
    numeric values. If the medium is homogeneous and velocity inputs or
    outputs are not required, it is not necessary to specify medium.density.

    Args:
        kgrid: kWaveGrid instance
        medium: kWaveMedium instance
        source: kWaveSource instance
        sensor: kWaveSensor instance
        **kwargs:

    Returns:

    """
    # start the timer and store the start time
    TicToc.tic()

    if simulation_options.simulation_type is not SimulationType.AXISYMMETRIC:
        logging.log(
            logging.WARN,
            "simulation type is not set to axisymmetric while using kSapceFirstOrderAS. " "Setting simulation type to axisymmetric.",
        )
        simulation_options.simulation_type = SimulationType.AXISYMMETRIC

    k_sim = kWaveSimulation(kgrid=kgrid, source=source, sensor=sensor, medium=medium, simulation_options=simulation_options)
    k_sim.input_checking("kspaceFirstOrderAS")

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================
    options = k_sim.options

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z + dz/2)

    k_sim.rho0 = np.atleast_1d(k_sim.rho0)
    if num_dim2(k_sim.rho0) == 2 and options.use_sg:
        # rho0 is heterogeneous and staggered grids are used
        grid_points = [k_sim.kgrid.x, k_sim.kgrid.y]
        k_sim.rho0_sgx = interpolate2d(grid_points, k_sim.rho0, [k_sim.kgrid.x + k_sim.kgrid.dx / 2, k_sim.kgrid.y])
        k_sim.rho0_sgy = interpolate2d(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y + k_sim.kgrid.dy / 2])
    else:
        # rho0 is homogeneous or staggered grids are not used
        k_sim.rho0_sgx = k_sim.rho0
        k_sim.rho0_sgy = k_sim.rho0

    # invert rho0 so it doesn't have to be done each time step
    k_sim.rho0_sgx_inv = 1 / k_sim.rho0_sgx
    k_sim.rho0_sgy_inv = 1 / k_sim.rho0_sgy

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del k_sim.rho0_sgx
        del k_sim.rho0_sgy
    k_sim.rho0_sgz = None

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

    k_sim.pml_x = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, staggered=False, dimension=1, axisymmetric=False)
    k_sim.pml_x_sgx = get_pml(
        Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, staggered=True and options.use_sg, dimension=1, axisymmetric=False
    )
    k_sim.pml_y = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, staggered=False, dimension=2, axisymmetric=True)
    k_sim.pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, staggered=True and options.use_sg, dimension=2, axisymmetric=True)

    # define the k-space, derivative, and shift operators
    # for the x (axial) direction, the operators are the same as normal
    kx_vec = k_sim.kgrid.k_vec.x
    k_sim.ddx_k_shift_pos = ifftshift(1j * kx_vec * np.exp(1j * kx_vec * dx / 2)).T
    k_sim.ddx_k_shift_neg = ifftshift(1j * kx_vec * np.exp(-1j * kx_vec * dx / 2)).T

    # for the y (radial) direction
    # when using DTTs:
    #    - there is no explicit grid shift (this is done by choosing DTTs
    #      with the appropriate symmetry)
    #    - ifftshift isn't required as the wavenumbers start from DC
    # when using FFTs:
    #    - the grid is expanded, and the fields replicated in the radial
    #      dimension to give the required symmetry
    #    - the derivative and shift operators are defined as normal
    if options.radial_symmetry in ["WSWA-FFT", "WSWS-FFT"]:
        # create a new kWave grid object with expanded radial grid
        if options.radial_symmetry == "WSWA-FFT":
            # extend grid by a factor of x4 to account for
            # symmetries in WSWA
            kgrid_exp = kWaveGrid([Nx, Ny * 4], [dx, dy])
        elif options.radial_symmetry == "WSWS-FFT":
            # extend grid by a factor of x2 - 2 to account for
            # symmetries in WSWS
            kgrid_exp = kWaveGrid([Nx, Ny * 2 - 2], [dx, dy])
        # define operators, rotating y-direction for use with bsxfun
        k_sim.ddy_k = ifftshift(1j * k_sim.kgrid.k_vec.y).T
        k_sim.y_shift_pos = ifftshift(np.exp(1j * kgrid_exp.k_vec.y * kgrid_exp.dy / 2)).T
        k_sim.y_shift_neg = ifftshift(np.exp(-1j * kgrid_exp.k_vec.y * kgrid_exp.dy / 2)).T

        # define the k-space operator
        if options.use_kspace:
            k_sim.kappa = ifftshift(sinc(c_ref * kgrid_exp.k * dt / 2))
            if (k_sim.source_p and (k_sim.source.p_mode == "additive")) or (
                (k_sim.source_ux or k_sim.source_uy) and (k_sim.source.u_mode == "additive")
            ):
                k_sim.source_kappa = ifftshift(np.cos(c_ref * kgrid_exp.k * dt / 2))
        else:
            k_sim.kappa = 1
            k_sim.source_kappa = 1
    elif options.radial_symmetry in ["WSWA", "WSWS"]:
        if options.radial_symmetry == "WSWA":
            # get the wavenumbers and implied length for the DTTs
            ky_vec, M = k_sim.kgrid.ky_vec_dtt(DiscreteCosine.TYPE_3)

            # define the derivative operators
            k_sim.ddy_k_wswa = -ky_vec.T
            k_sim.ddy_k_hahs = ky_vec.T
        elif options.radial_symmetry == "WSWS":
            # get the wavenumbers and implied length for the DTTs
            ky_vec, M = k_sim.kgrid.ky_vec_dtt(DiscreteCosine.TYPE_1)

            # define the derivative operators
            k_sim.ddy_k_wsws = -ky_vec[1:].T
            k_sim.ddy_k_haha = ky_vec[1:].T

        # define the k-space operator
        if options.use_kspace:
            # define scalar wavenumber
            k_dtt = np.sqrt(
                np.tile(ifftshift(k_sim.kgrid.k_vec.x) ** 2, [1, k_sim.kgrid.Ny]) + np.tile((ky_vec.T) ** 2, [k_sim.kgrid.Nx, 1])
            )

            # define k-space operators
            k_sim.kappa = sinc(c_ref * k_dtt * k_sim.kgrid.dt / 2)
            if (k_sim.source_p and (k_sim.source.p_mode == "additive")) or (
                (k_sim.source_ux or k_sim.source_uy) and (k_sim.source.u_mode == "additive")
            ):
                k_sim.source_kappa = np.cos(c_ref * k_dtt * k_sim.kgrid.dt / 2)

            # cleanup unused variables
            del k_dtt

        else:
            k_sim.kappa = 1
            k_sim.source_kappa = 1

    # define staggered and non-staggered grid axial distance
    k_sim.y_vec = (k_sim.kgrid.y_vec - k_sim.kgrid.y_vec[0]).T
    k_sim.y_vec_sg = (k_sim.kgrid.y_vec - k_sim.kgrid.y_vec[0] + k_sim.kgrid.dy / 2).T

    # option to run simulations without the spatial staggered grid is not
    # supported for the axisymmetric code
    assert options.use_sg, "Optional input " "UseSG" " is not supported for axisymmetric simulations."

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
