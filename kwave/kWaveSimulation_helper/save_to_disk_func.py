import logging
import os

import numpy as np
from scipy.io import savemat

from kwave.kmedium import kWaveMedium
from kwave.kgrid import kWaveGrid
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.data import scale_time
from kwave.utils.dotdictionary import dotdict
from kwave.utils.io import write_attributes, write_matrix
from kwave.utils.matrix import num_dim2
from kwave.utils.tictoc import TicToc


def save_to_disk_func(
    kgrid: kWaveGrid, medium: kWaveMedium, source, opt: SimulationOptions, auto_chunk: bool, values: dotdict, flags: dotdict
):
    # update command line status
    logging.log(logging.INFO, "  precomputation completed in ", scale_time(TicToc.toc()))
    TicToc.tic()
    logging.log(logging.INFO, "  saving input files to disk...")

    # check for a binary sensor mask or cuboid corners
    # modified by Farid | disabled temporarily!
    # assert self.binary_sensor_mask or self.cuboid_corners, \
    #     "Optional input ''save_to_disk'' only supported for sensor masks defined as a binary matrix
    #           or the opposing corners of a rectangle (2D) or cuboid (3D)."

    # =========================================================================
    # VARIABLE LIST
    # =========================================================================
    integer_variables = dotdict()
    float_variables = dotdict()

    grab_integer_variables(integer_variables, kgrid, flags, medium)
    grab_pml_size(integer_variables, opt)
    grab_float_variables(float_variables, kgrid, opt, values, flags.elastic_code, flags.axisymmetric)

    # overwrite z-values for 2D simulations
    if kgrid.dim == 2:
        integer_variables.Nz = 1
        integer_variables.pml_z_size = 0

    grab_medium_props(integer_variables, float_variables, medium, flags.elastic_code)
    grab_source_props(
        integer_variables,
        float_variables,
        source,
        values.u_source_pos_index,
        values.s_source_pos_index,
        values.p_source_pos_index,
        values.transducer_input_signal,
        values.delay_mask,
    )

    grab_sensor_props(integer_variables, kgrid.dim, values.sensor_mask_index, values.record.cuboid_corners_list)
    grab_nonuniform_grid_props(float_variables, kgrid, flags.nonuniform_grid)

    # =========================================================================
    # DATACAST AND SAVING
    # =========================================================================

    remove_z_dimension(float_variables, kgrid.dim)
    save_file(opt.input_filename, integer_variables, float_variables, opt.hdf_compression_level, auto_chunk=auto_chunk)

    # update command line status
    logging.log(logging.INFO, "  completed in ", scale_time(TicToc.toc()))


def grab_integer_variables(integer_variables, kgrid, flags, medium):
    # integer variables used within the time loop for all codes

    variables = dotdict(
        {
            "Nx": kgrid.Nx,
            "Ny": kgrid.Ny,
            "Nz": kgrid.Nz,
            "Nt": kgrid.Nt,
            "p_source_flag": flags.source_p,
            "p0_source_flag": flags.source_p0,
            "ux_source_flag": flags.source_ux,
            "uy_source_flag": flags.source_uy,
            "uz_source_flag": flags.source_uz,
            "sxx_source_flag": flags.source_sxx,
            "syy_source_flag": flags.source_syy,
            "szz_source_flag": flags.source_szz,
            "sxy_source_flag": flags.source_sxy,
            "sxz_source_flag": flags.source_sxz,
            "syz_source_flag": flags.source_syz,
            "transducer_source_flag": flags.transducer_source,
            "nonuniform_grid_flag": flags.nonuniform_grid,
            "nonlinear_flag": medium.is_nonlinear(),
            "absorbing_flag": None,
            "elastic_flag": flags.elastic_code,
            "axisymmetric_flag": flags.axisymmetric,
            # create pseudonyms for the sensor flgs
            #   0: binary mask indices
            #   1: cuboid corners
            "sensor_mask_type": flags.cuboid_corners,
        }
    )
    integer_variables.update(variables)


def grab_pml_size(integer_variables, opt):
    # additional integer variables not used within time loop but stored directly to output file
    integer_variables["pml_x_size"] = opt.pml_x_size
    integer_variables["pml_y_size"] = opt.pml_y_size
    integer_variables["pml_z_size"] = opt.pml_z_size


def grab_float_variables(float_variables: dotdict, kgrid, opt, values, is_elastic_code, is_axisymmetric):
    # single precision variables not used within time loop but stored directly
    # to the output file for all files
    variables = dotdict(
        {
            "dx": kgrid.dx,
            "dy": kgrid.dy,
            "dz": kgrid.dz,
            "pml_x_alpha": opt.pml_x_alpha,
            "pml_y_alpha": opt.pml_y_alpha,
            "pml_z_alpha": opt.pml_z_alpha,
        }
    )
    float_variables.update(variables)

    if is_elastic_code:  # pragma: no cover
        grab_elastic_code_variables(float_variables, kgrid, values)
    elif is_axisymmetric:
        grab_axisymmetric_variables(float_variables, values)
    else:
        # single precision variables used within the time loop
        float_variables["dt"] = values.dt
        float_variables["c0"] = values.c0
        float_variables["c_ref"] = values.c_ref
        float_variables["rho0"] = values.rho0
        float_variables["rho0_sgx"] = values.rho0_sgx
        float_variables["rho0_sgy"] = values.rho0_sgy
        float_variables["rho0_sgz"] = values.rho0_sgz


def grab_elastic_code_variables(float_variables, kgrid, values):  # pragma: no cover
    # single precision variables used within the time loop
    float_variables["dt"] = None
    float_variables["c_ref"] = None
    float_variables["lambda"] = None
    float_variables["mu"] = None

    float_variables["rho0_sgx"] = None
    float_variables["rho0_sgy"] = None
    float_variables["rho0_sgz"] = None

    float_variables["mu_sgxy"] = None
    float_variables["mu_sgxz"] = None
    float_variables["mu_sgyz"] = None

    # create shift variables used for calculating u_non_staggered and I outputs
    x_shift_neg = np.fft.ifftshift(np.exp(-1j * kgrid.k_vec.x * kgrid.dx / 2))
    y_shift_neg = np.fft.ifftshift(np.exp(-1j * kgrid.k_vec.y * kgrid.dy / 2)).T
    z_shift_neg = np.transpose(np.fft.ifftshift(np.exp(-1j * kgrid.k_vec.z * kgrid.dz / 2)), (1, 2, 0))

    # create reduced variables for use with real-to-complex FFT
    Nz = kgrid.Nz if kgrid.dim != 2 else 1
    Nx_r = kgrid.Nx // 2 + 1
    Ny_r = kgrid.Ny // 2 + 1
    Nz_r = Nz // 2 + 1

    ddx_k_shift_pos = values.ddx_k_shift_pos
    ddx_k_shift_neg = values.ddx_k_shift_neg

    float_variables["ddx_k_shift_pos_r"] = ddx_k_shift_pos[:Nx_r]
    float_variables["ddy_k_shift_pos"] = None
    float_variables["ddz_k_shift_pos"] = None

    float_variables["ddx_k_shift_neg_r"] = ddx_k_shift_neg[:Nx_r]
    float_variables["ddy_k_shift_neg"] = None
    float_variables["ddz_k_shift_neg"] = None

    float_variables["x_shift_neg_r"] = x_shift_neg[:Nx_r]
    float_variables["y_shift_neg_r"] = y_shift_neg[:Ny_r]
    float_variables["z_shift_neg_r"] = z_shift_neg[:Nz_r]

    del x_shift_neg

    float_variables["pml_x"] = None
    float_variables["pml_y"] = None
    float_variables["pml_z"] = None

    float_variables["pml_x_sgx"] = None
    float_variables["pml_y_sgy"] = None
    float_variables["pml_z_sgz"] = None

    float_variables["mpml_x_sgx"] = None
    float_variables["mpml_y_sgy"] = None
    float_variables["mpml_z_sgz"] = None

    float_variables["mpml_x"] = None
    float_variables["mpml_y"] = None
    float_variables["mpml_z"] = None


def grab_axisymmetric_variables(float_variables, values):
    # single precision variables used within the time loop
    float_variables["dt"] = values.dt
    float_variables["c0"] = values.c0
    float_variables["c_ref"] = values.c_ref
    float_variables["rho0"] = values.rho0
    float_variables["rho0_sgx"] = values.rho0_sgx
    float_variables["rho0_sgy"] = values.rho0_sgy


def grab_medium_props(integer_variables, float_variables, medium, is_elastic_code):
    # =========================================================================
    # VARIABLES USED IN NONLINEAR SIMULATIONS
    # =========================================================================
    if medium.is_nonlinear():
        float_variables["BonA"] = medium.BonA

    # =========================================================================
    # VARIABLES USED IN ABSORBING SIMULATIONS
    # =========================================================================

    # set absorbing flag
    if medium.absorbing:
        integer_variables.absorbing_flag = 2 if medium.stokes else 1
    else:
        integer_variables.absorbing_flag = 0

    if medium.absorbing:
        if is_elastic_code:  # pragma: no cover
            # add to the variable list
            float_variables["chi"] = None
            float_variables["eta"] = None
            float_variables["eta_sgxy"] = None
            float_variables["eta_sgxz"] = None
            float_variables["eta_sgyz"] = None
        else:
            float_variables["alpha_coeff"] = medium.alpha_coeff
            float_variables["alpha_power"] = medium.alpha_power


def grab_source_props(
    integer_variables,
    float_variables,
    source,
    u_source_pos_index,
    s_source_pos_index,
    p_source_pos_index,
    transducer_input_signal,
    delay_mask,
):
    # =========================================================================
    # SOURCE VARIABLES
    # =========================================================================
    # source modes and indicies
    # - these are only defined if the source flgs are > 0
    # - the source mode describes whether the source will be added or replaced
    # - the source indicies describe which grid points act as the source
    # - the u_source_index is reused for any of the u sources and the transducer source

    grab_velocity_source_props(integer_variables, source, u_source_pos_index)
    grab_stress_source_props(integer_variables, source, s_source_pos_index)
    grab_pressure_source_props(integer_variables, source, p_source_pos_index, u_source_pos_index)
    grab_time_varying_source_props(integer_variables, float_variables, source, transducer_input_signal, delay_mask)


def grab_velocity_source_props(integer_variables, source, u_source_pos_index):
    # velocity source
    if any(integer_variables.get(k) for k in ["ux_source_flag", "uy_source_flag", "uz_source_flag"]):
        integer_variables["u_source_mode"] = {
            "dirichlet": 0,
            "additive-no-correction": 1,
            "additive": 2,
        }[source.u_mode]

        if integer_variables.ux_source_flag:
            u_source_many = num_dim2(source.ux) > 1
        elif integer_variables.uy_source_flag:
            u_source_many = num_dim2(source.uy) > 1
        elif integer_variables.uz_source_flag:
            u_source_many = num_dim2(source.uz) > 1
        integer_variables["u_source_many"] = u_source_many

        integer_variables.u_source_index = u_source_pos_index


def grab_stress_source_props(integer_variables, source, s_source_pos_index):
    # stress source
    if (
        integer_variables.sxx_source_flag
        or integer_variables.syy_source_flag
        or integer_variables.szz_source_flag
        or integer_variables.sxy_source_flag
        or integer_variables.sxz_source_flag
        or integer_variables.syz_source_flag
    ):
        integer_variables.s_source_mode = source.s_mode != "dirichlet"
        if integer_variables.sxx_source_flag:
            s_source_many = num_dim2(source.sxx) > 1
        elif integer_variables.syy_source_flag:
            s_source_many = num_dim2(source.syy) > 1
        elif integer_variables.szz_source_flag:
            s_source_many = num_dim2(source.szz) > 1
        elif integer_variables.sxy_source_flag:
            s_source_many = num_dim2(source.sxy) > 1
        elif integer_variables.sxz_source_flag:
            s_source_many = num_dim2(source.sxz) > 1
        elif integer_variables.syz_source_flag:
            s_source_many = num_dim2(source.syz) > 1
        integer_variables.s_source_many = s_source_many
        integer_variables.s_source_index = s_source_pos_index


def grab_pressure_source_props(integer_variables, source, p_source_pos_index, u_source_pos_index):
    # pressure source
    if integer_variables.p_source_flag:
        integer_variables.p_source_mode = {
            "dirichlet": 0,
            "additive-no-correction": 1,
            "additive": 2,
        }[source.p_mode]
        integer_variables.p_source_many = num_dim2(source.p) > 1
        integer_variables.p_source_index = p_source_pos_index

    # transducer source
    if integer_variables.transducer_source_flag:
        integer_variables.u_source_index = u_source_pos_index


def grab_time_varying_source_props(integer_variables, float_variables, source, transducer_input_signal, delay_mask):
    # time varying source variables
    # - these are only defined if the source flgs are > 0
    # - these are the actual source values
    # - these are indexed as (position_index, time_index)
    if integer_variables.ux_source_flag:
        float_variables.ux_source_input = source.ux

    if integer_variables.uy_source_flag:
        float_variables.uy_source_input = source.uy

    if integer_variables.uz_source_flag:
        float_variables.uz_source_input = source.uz

    if integer_variables.sxx_source_flag:
        float_variables.sxx_source_input = source.sxx

    if integer_variables.syy_source_flag:
        float_variables.syy_source_input = source.syy

    if integer_variables.szz_source_flag:
        float_variables.szz_source_input = source.szz

    if integer_variables.sxy_source_flag:
        float_variables.sxy_source_input = source.sxy

    if integer_variables.sxz_source_flag:
        float_variables.sxz_source_input = source.sxz

    if integer_variables.syz_source_flag:
        float_variables.syz_source_input = source.syz

    if integer_variables.p_source_flag:
        float_variables.p_source_input = source.p

    if integer_variables.transducer_source_flag:
        float_variables.transducer_source_input = transducer_input_signal
        integer_variables.delay_mask = delay_mask

    # initial pressure source variable
    # - this is only defined if the p0 source flag is 1
    # - this defines the initial pressure everywhere (there is no indicies)
    if integer_variables.p0_source_flag:
        float_variables.p0_source_input = source.p0


def grab_sensor_props(integer_variables, kgrid_dim, sensor_mask_index, cuboid_corners_list):
    # =========================================================================
    # SENSOR VARIABLES
    # =========================================================================

    if integer_variables.sensor_mask_type == 0:
        # mask is defined as a list of grid indices
        integer_variables.sensor_mask_index = sensor_mask_index

    elif integer_variables.sensor_mask_type == 1:
        cuboid_corners_list = cuboid_corners_list
        # mask is defined as a list of cuboid corners
        if kgrid_dim == 2:
            sensor_mask_corners = np.ones((6, cuboid_corners_list.shape[1]))
            sensor_mask_corners[0, :] = cuboid_corners_list[0, :]
            sensor_mask_corners[1, :] = cuboid_corners_list[1, :]
            sensor_mask_corners[3, :] = cuboid_corners_list[2, :]
            sensor_mask_corners[4, :] = cuboid_corners_list[3, :]
        else:
            sensor_mask_corners = cuboid_corners_list
        integer_variables.sensor_mask_corners = sensor_mask_corners

    else:
        raise NotImplementedError("unknown option for sensor_mask_type")


def grab_nonuniform_grid_props(float_variables, kgrid, is_nonuniform_grid):
    # =========================================================================
    # VARIABLES USED FOR NONUNIFORM GRIDS
    # =========================================================================

    # set nonuniform flag and variables
    # - these are only defined if nonuniform_grid_flag is 1
    # - these are applied using the bsxfun formulation
    if not is_nonuniform_grid:
        return

    dxudxn = kgrid.dudn.x
    if np.array(dxudxn).size == 1:
        dxudxn = np.ones((kgrid.Nx, 1))
    float_variables["dxudxn"] = dxudxn

    dyudyn = kgrid.dudn.y
    if np.array(dyudyn).size == 1:
        dyudyn = np.ones((1, kgrid.Ny))
    float_variables["dyudyn"] = dyudyn

    dzudzn = kgrid.dudn.z
    if np.array(dzudzn).size == 1:
        dzudzn = np.ones((1, 1, kgrid.Nz))
    float_variables["dzudzn"] = dzudzn

    dxudxn_sgx = kgrid.dudn_sg.x
    if np.array(dxudxn).size == 1:
        dxudxn_sgx = np.ones((kgrid.Nx, 1))
    float_variables["dxudxn_sgx"] = dxudxn_sgx

    dyudyn_sgy = kgrid.dudn_sg.y
    if np.array(dyudyn).size == 1:
        dyudyn_sgy = np.ones((1, kgrid.Ny))
    float_variables["dyudyn_sgy"] = dyudyn_sgy

    dzudzn_sgz = kgrid.dudn_sg.z
    if np.array(dzudzn).size == 1:
        dzudzn_sgz = np.ones((1, 1, kgrid.Nz))
    float_variables["dzudzn_sgz"] = dzudzn_sgz


def remove_z_dimension(float_variables, kgrid_dim):
    # remove z-dimension variables for saving 2D files
    if kgrid_dim == 2:
        for k in list(float_variables.keys()):
            if "z" in k:
                del float_variables[k]


def enforce_filename_standards(filepath):
    # check for HDF5 filename extension
    filename_ext = os.path.splitext(filepath)[1]

    # use .h5 as default if no extension is given
    if len(filename_ext) == 0:
        filename_ext = ".h5"
        filepath = filepath + ".h5"
    return filepath, filename_ext


def save_file(filepath, integer_variables, float_variables, hdf_compression_level, auto_chunk):
    filepath, filename_ext = enforce_filename_standards(filepath)

    # save file
    if filename_ext == ".h5":
        save_h5_file(filepath, integer_variables, float_variables, hdf_compression_level, auto_chunk)

    elif filename_ext == ".mat":
        save_mat_file(filepath, integer_variables, float_variables)
    else:
        # throw error for unknown filetype
        raise NotImplementedError("unknown file extension for " "save_to_disk" " filename")


def save_h5_file(filepath, integer_variables, float_variables, hdf_compression_level, auto_chunk):
    # ----------------
    # SAVE HDF5 FILE
    # ----------------

    # check if file exists, and delete if it does (the hdf5 library will
    # give an error if the file already exists)
    if os.path.exists(filepath):
        os.remove(filepath)

    # change all the variables to be in single precision (float in C++),
    # then add to HDF5 File
    for key, value in float_variables.items():
        # cast matrix to single precision
        value = np.array(value, dtype=np.float32)
        write_matrix(filepath, value, key, hdf_compression_level, auto_chunk)
        del value

    # change all the index variables to be in 64-bit unsigned integers
    # (long in C++), then add to HDF5 file
    for key, value in integer_variables.items():
        # cast matrix to 64-bit unsigned integer
        value = np.array(value, dtype=np.uint64)
        write_matrix(filepath, value, key, hdf_compression_level, auto_chunk)
        del value

    # set additional file attributes
    write_attributes(filepath)


def save_mat_file(filepath, integer_variables, float_variables):
    # ----------------
    # SAVE .MAT FILE
    # ----------------

    # change all the variables to be in single precision (float in C++)
    for key, value in float_variables.items():
        float_variables[key] = np.array(value, dtype=np.float32)

    for key, value in integer_variables.items():
        integer_variables[key] = np.array(value, dtype=np.uint64)

    # save the input variables to disk as a MATLAB binary file
    float_variables = dict(**float_variables, **integer_variables)
    savemat(filepath, float_variables)
