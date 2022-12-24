"""
    Saving Input Files In Parts Example

    This example demonstrates how to save the HDF5 input files required by
    the C++ code in parts. It builds on the Running C++ Simulations Example.
"""
import os
from tempfile import gettempdir

import numpy as np

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import cast_to_type
from kwave.utils.interp import interpolate3d
from kwave.utils.io import get_h5_literals, write_matrix, write_attributes, write_flags, write_grid
from kwave.utils.mapgen import make_ball
from kwave.utils.matlab import matlab_find
from kwave.utils.tictoc import TicToc
from tests.diff_utils import compare_against_ref


def test_cpp_io_in_parts():
    # modify this parameter to run the different examples
    # 1: Save the input data to disk in parts
    # 2: Reload the output data from disk
    example_number = 1

    # input and output filenames (these must have the .h5 extension)
    input_filename = 'out_cpp_io_in_parts.h5'
    output_filename = 'out_cpp_io_in_parts.h5'

    # pathname for the input and output files
    pathname = gettempdir()

    # remove input file if it already exists
    input_file_full_path = os.path.join(pathname, input_filename)
    if example_number == 1 and os.path.exists(input_file_full_path):
        os.remove(input_file_full_path)

    # load HDF5 constants
    h5_literals = get_h5_literals()

    # =========================================================================
    # SIMULATION SETTINGS
    # =========================================================================

    # set the properties of the computational grid
    Nx = 256                   # number of grid points in the x direction
    Ny = 128                   # number of grid points in the y direction
    Nz = 64                    # number of grid points in the z direction
    dx = 0.1e-3                # grid point spacing in the x direction [m]
    dy = 0.1e-3                # grid point spacing in the y direction [m]
    dz = 0.1e-3                # grid point spacing in the z direction [m]
    Nt = 1200                  # number of time steps
    dt = 15e-9                 # time step [s]

    # set the properties of the perfectly matched layer
    pml_x_size  = 10           # [grid points]
    pml_y_size  = 10           # [grid points]
    pml_z_size  = 10           # [grid points]
    pml_x_alpha = 2            # [Nepers/grid point]
    pml_y_alpha = 2            # [Nepers/grid point]
    pml_z_alpha = 2            # [Nepers/grid point]

    # define a scattering ball
    ball_radius = 20           # [grid points]
    ball_x      = Nx/2 + 40    # [grid points]
    ball_y      = Ny/2         # [grid points]
    ball_z      = Nz/2         # [grid points]

    # define the properties of the medium
    c0_background   = 1500     # [kg/m^3]
    c0_ball         = 1800     # [kg/m^3]
    rho0_background = 1000     # [kg/m^3]
    rho0_ball       = 1200     # [kg/m^3]
    alpha_coeff     = 0.75     # [dB/(MHz^y cm)]
    alpha_power     = 1.5

    # define a the properties of a single square source element facing in the
    # x-direction
    source_y_size   = 60       # [grid points]
    source_z_size   = 30       # [grid points]
    source_freq     = 2e6      # [Hz]
    source_strength = 0.5e6    # [Pa]

    if example_number == 1:
        # =========================================================================
        # WRITE THE INPUT FILE
        # =========================================================================

        # ---------------------------------------------------------------------
        # WRITE THE MEDIUM PARAMETERS
        # ---------------------------------------------------------------------

        # update command line status
        TicToc.tic()
        print('Writing medium parameters... ')

        # :::---:::---:::---:::---:::---:::---:::---:::---:::---:::---:::---:::

        # create the scattering ball and density matrix
        ball       = make_ball(float(Nx), float(Ny), float(Nz), float(ball_x), float(ball_y), float(ball_z), float(ball_radius), False, True)
        ball       = np.array(ball)
        rho0       = rho0_background * np.ones((Nx, Ny, Nz), dtype=np.float32)
        rho0[ball] = rho0_ball

        # make sure the input is in the correct data format
        rho0 = cast_to_type(rho0, h5_literals.MATRIX_DATA_TYPE_MATLAB)

        # save the density matrix
        write_matrix(input_file_full_path, rho0, 'rho0')

        # create grid (variables are used for interpolation)
        kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

        # interpolate onto staggered grid in x direction and save
        rho0_sg = interpolate3d([kgrid.x, kgrid.y, kgrid.z], rho0, [kgrid.x + kgrid.dx / 2, kgrid.y, kgrid.z])
        write_matrix(input_file_full_path, rho0_sg.astype(np.float32), 'rho0_sgx')

        # interpolate onto staggered grid in y direction and save
        rho0_sg = interpolate3d([kgrid.x, kgrid.y, kgrid.z], rho0, [kgrid.x, kgrid.y + kgrid.dy / 2, kgrid.z])
        write_matrix(input_file_full_path, rho0_sg.astype(np.float32), 'rho0_sgy')

        # interpolate onto staggered grid in z direction and save
        rho0_sg = interpolate3d([kgrid.x, kgrid.y, kgrid.z], rho0, [kgrid.x, kgrid.y, kgrid.z + kgrid.dz / 2])
        write_matrix(input_file_full_path, rho0_sg.astype(np.float32), 'rho0_sgz')

        # clear variable to free memory
        del kgrid
        del rho0
        del rho0_sg

        # :::---:::---:::---:::---:::---:::---:::---:::---:::---:::---:::---:::

        # create the sound speed matrix
        c0       = c0_background * np.ones((Nx, Ny, Nz), dtype=np.float32)
        c0[ball] = c0_ball

        # set the reference sound speed to the maximum in the medium
        c_ref = np.max(c0)

        # get the sound speed at the location of the source
        c_source = np.min(c0)

        # make sure the input is in the correct data format
        c0 = cast_to_type(c0, h5_literals.MATRIX_DATA_TYPE_MATLAB)

        # save the sound speed matrix
        write_matrix(input_file_full_path, c0, 'c0')

        # clear variable to free memory
        del c0
        del ball

        # :::---:::---:::---:::---:::---:::---:::---:::---:::---:::---:::---:::

        # make sure the inputs are in the correct data format
        alpha_coeff = cast_to_type(alpha_coeff, h5_literals.MATRIX_DATA_TYPE_MATLAB)
        alpha_power = cast_to_type(alpha_power, h5_literals.MATRIX_DATA_TYPE_MATLAB)

        # save the absorption variables
        write_matrix(input_file_full_path, alpha_coeff, 'alpha_coeff')
        write_matrix(input_file_full_path, alpha_power, 'alpha_power')

        # clear variables to free memory
        del alpha_power
        del alpha_coeff

        # ---------------------------------------------------------------------
        # WRITE THE SOURCE PARAMETERS
        # ---------------------------------------------------------------------

        # update command line status
        TicToc.toc(reset=True)
        print('Writing source parameters... ')

        # define a square source mask facing in the x-direction using the
        # normal k-Wave syntax
        p_mask = np.zeros((Nx, Ny, Nz)).astype(bool)
        p_mask[pml_x_size, Ny//2 - source_y_size//2-1:Ny//2 + source_y_size//2, Nz//2 - source_z_size//2-1:Nz//2 + source_z_size//2] = 1

        # find linear source indices
        p_source_index = np.where(p_mask.flatten(order='F') == 1)[0] + 1  # +1 due to Matlab indexing
        p_source_index = p_source_index.reshape((-1, 1))

        # make sure the input is in the correct data format
        p_source_index = cast_to_type(p_source_index, h5_literals.INTEGER_DATA_TYPE_MATLAB)

        # save the source index matrix
        write_matrix(input_file_full_path, p_source_index, 'p_source_index')

        # clear variables to free memory
        del p_mask
        del p_source_index

        # define a time varying sinusoidal source
        # p_source_input = 2 * np.pi * source_freq * np.arange(0, Nt) * dt
        # p_source_input = source_strength * np.sin(p_source_input)
        p_source_input = source_strength * np.sin(2 * np.pi * source_freq * np.arange(0, Nt) * dt)

        # apply an cosine ramp to the beginning to avoid startup transients
        ramp_length = np.round((2 * np.pi / source_freq) / dt).astype(int)
        p_source_input[0:ramp_length] = p_source_input[0:ramp_length] * (-np.cos(np.arange(0, ramp_length) * np.pi / ramp_length) + 1)/2

        # scale the source magnitude to be in the correct units for the code
        p_source_input = p_source_input * (2 * dt / (3 * c_source * dx))

        # cast matrix to single precision
        p_source_input = p_source_input[None, :]
        p_source_input = cast_to_type(p_source_input, h5_literals.MATRIX_DATA_TYPE_MATLAB)

        # save the input signal
        write_matrix(input_file_full_path, p_source_input, 'p_source_input')

        # clear variables to free memory
        del p_source_input

        # ---------------------------------------------------------------------
        # WRITE THE SENSOR PARAMETERS
        # ---------------------------------------------------------------------

        # update command line status
        TicToc.toc(reset=True)
        print('Writing sensor parameters... ')

        # define a sensor mask through the central plane
        sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        sensor_mask[:, :, Nz//2 - 1] = 1

        # extract the indices of the active sensor mask elements
        sensor_mask_index = matlab_find(sensor_mask)
        sensor_mask_index = sensor_mask_index.reshape((-1, 1))

        # make sure the input is in the correct data format
        sensor_mask_index = cast_to_type(sensor_mask_index, h5_literals.INTEGER_DATA_TYPE_MATLAB)

        # save the sensor mask
        write_matrix(input_file_full_path, sensor_mask_index, 'sensor_mask_index')

        # clear variables to free memory
        del sensor_mask
        del sensor_mask_index

        # ---------------------------------------------------------------------
        # WRITE THE GRID PARAMETERS AND FILE ATTRIBUTES
        # ---------------------------------------------------------------------

        # update command line status
        TicToc.toc(reset=True)
        print('Writing grid parameters and attributes... ')

        # write grid parameters
        write_grid(input_file_full_path, [Nx, Ny, Nz], [dx, dy, dz], [pml_x_size, pml_y_size, pml_z_size], [pml_x_alpha, pml_y_alpha, pml_z_alpha], Nt, dt, c_ref)

        # write flags
        write_flags(input_file_full_path)

        # set additional file attributes
        write_attributes(input_file_full_path, legacy=True)

        TicToc.toc()

        # display the required syntax to run the C++ simulation
        print(f'Using a terminal window, navigate to the {os.path.sep}binaries folder of the k-Wave Toolbox')
        print('Then, use the syntax shown below to run the simulation:')
        if os.name == 'posix':
            print(f'./kspaceFirstOrder-OMP -i {input_file_full_path} -o {pathname} {output_filename} --p_final --p_max')
        else:
            print(f'kspaceFirstOrder-OMP.exe -i {input_file_full_path} -o {pathname} {output_filename} --p_final --p_max')

        assert compare_against_ref('out_cpp_io_in_parts', input_file_full_path, precision=6), 'Files do not match!'

    else:
        pass
