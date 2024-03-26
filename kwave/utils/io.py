import os
import platform
import socket
from datetime import datetime
from typing import Optional

import cv2
import h5py
import numpy as np

import kwave
from .conversion import cast_to_type
from .data import get_date_string
from .dotdictionary import dotdict


def get_h5_literals():
    literals = dotdict(
        {
            # data type
            "DATA_TYPE_ATT_NAME": "data_type",
            "MATRIX_DATA_TYPE_MATLAB": "single",
            "MATRIX_DATA_TYPE_C": "float",
            "INTEGER_DATA_TYPE_MATLAB": "uint64",
            "INTEGER_DATA_TYPE_C": "long",
            # real / complex
            "DOMAIN_TYPE_ATT_NAME": "domain_type",
            "DOMAIN_TYPE_REAL": "real",
            "DOMAIN_TYPE_COMPLEX": "complex",
            # file descriptors
            "FILE_MAJOR_VER_ATT_NAME": "major_version",
            "FILE_MINOR_VER_ATT_NAME": "minor_version",
            "FILE_DESCR_ATT_NAME": "file_description",
            "FILE_CREATION_DATE_ATT_NAME": "creation_date",
            "CREATED_BY_ATT_NAME": "created_by",
            # file type
            "FILE_TYPE_ATT_NAME": "file_type",
            "HDF_INPUT_FILE": "input",
            "HDF_OUTPUT_FILE": "output",
            "HDF_CHECKPOINT_FILE": "checkpoint",
            # file version information
            "HDF_FILE_MAJOR_VERSION": "1",
            "HDF_FILE_MINOR_VERSION": "2",
            # compression level
            "HDF_COMPRESSION_LEVEL": 0,
        }
    )
    return literals


def write_matrix(filename, matrix: np.ndarray, matrix_name: str, compression_level: int = None, auto_chunk: bool = True):
    # get literals
    h5_literals = get_h5_literals()

    assert isinstance(auto_chunk, bool), "auto_chunk must be a boolean."

    if compression_level is None:
        compression_level = h5_literals.HDF_COMPRESSION_LEVEL

    # dims = num_dim(matrix)
    dims = len(matrix.shape)

    if dims == 3:
        matrix = np.transpose(matrix, [2, 1, 0])  # C <=> Fortran ordering
    if dims == 2:
        matrix = np.transpose(matrix)  # C <=> Fortran ordering

    # get the size of the input matrix
    if dims == 3:
        Nx, Ny, Nz = matrix.shape
    elif dims == 2:
        Ny, Nz = matrix.shape
        Nx = 1
    else:
        Nx, Ny, Nz = 1, 1, 1

    # check size of matrix and set chunk size and compression level
    if dims == 3:
        # set chunk size to Nx * Ny
        chunk_size = [Nx, Ny, 1]
    elif dims == 2:
        # set chunk size to Nx
        chunk_size = [Nx, 1, 1]
    elif dims <= 1:
        # check that the matrix size is greater than 1 MB
        one_mb = (1024**2) / 8
        if matrix.size > one_mb:
            # set chunk size to 1 MB
            if Nx > Ny:
                chunk_size = [one_mb, 1, 1]
            elif Ny > Nz:
                chunk_size = [1, one_mb, 1]
            else:
                chunk_size = [1, 1, one_mb]
        else:
            # set no compression
            compression_level = 0

            # set chunk size to grid size
            if matrix.size == 1:
                chunk_size = (1, 1, 1)
            elif Nx > Ny:
                chunk_size = (Nx, 1, 1)
            elif Ny > Nz:
                chunk_size = (1, Ny, 1)
            else:
                chunk_size = (1, 1, Nz)
    else:
        # throw error for unknown matrix size
        raise ValueError("Input matrix must have 1, 2 or 3 dimensions.")

    # check the format of the matrix is either single precision (float in C++)
    # or uint64 (unsigned long in C++)
    if matrix.dtype == np.float32:
        # set data type flags
        data_type_matlab = h5_literals.MATRIX_DATA_TYPE_MATLAB
        data_type_c = h5_literals.MATRIX_DATA_TYPE_C
    elif matrix.dtype == np.uint64:
        # set data type flags
        data_type_matlab = h5_literals.INTEGER_DATA_TYPE_MATLAB
        data_type_c = h5_literals.INTEGER_DATA_TYPE_C

    else:
        # throw error for unknown data type
        raise ValueError("Input matrix must be of type " "single" " or " "uint64" ".")

    # check if the input matrix is real or complex, if complex, rearrange the
    # data in the C++ format
    if np.isreal(matrix).all():
        # set file tag
        domain_type = "real"  # DOMAIN_TYPE_REAL

    elif dims == 3:
        # set file tag
        domain_type = h5_literals.DOMAIN_TYPE_COMPLEX

        # rearrange the data so the real and imaginary parts are stored in the
        # same matrix
        matrix = np.concatenate(matrix.real, matrix.imag, axis=0)
        matrix = matrix.reshape((Nx, 2, Ny, Nz))
        matrix = np.transpose(matrix, (1, 0, 2, 3))
        matrix = matrix.reshape((2 * Nx, Ny, Nz))

        # update the size of Nx
        Nx = 2 * Nx

    elif dims <= 1:
        # set file tag
        domain_type = h5_literals.DOMAIN_TYPE_COMPLEX

        # rearrange the data so the real and imaginary parts are stored in the
        # same matrix
        nelems = matrix.size
        matrix = matrix.reshape((nelems, 1))
        matrix = np.concatenate(matrix.real, matrix.imag, axis=0)
        matrix = matrix.reshape((nelems, 2, 1, 1))
        matrix = np.transpose(matrix, (1, 0, 2, 3))

        # update the matrix size
        Nx = Nx * (2 - np.array(Nx == 1).astype(float))
        Ny = Ny * (2 - np.array(Ny == 1).astype(float))
        Nz = Nz * (2 - np.array(Nz == 1).astype(float))

        # double store in x-direction if a complex scalar
        if Nx == 1 and Ny == 1 and Nz == 1:
            Nx = 2 * Nx

        # put in correct dimension
        matrix = matrix.reshape((Nx, Ny, Nz))

    else:
        raise NotImplementedError("Currently there is no support for saving 2D complex matrices.")

    # allocate a holder for the new matrix within the file
    opts = {"dtype": data_type_matlab, "chunks": auto_chunk if auto_chunk is True else tuple(chunk_size)}

    if compression_level != 0:
        # use compression
        opts["compression"] = compression_level

    # write the matrix into the file
    with h5py.File(filename, "a") as f:
        f.create_dataset(f"/{matrix_name}", [Nx, Ny, Nz], data=matrix, **opts)

        # set attributes for the matrix (used by k-Wave++)
        assign_str_attr(f[f"/{matrix_name}"].attrs, h5_literals.DOMAIN_TYPE_ATT_NAME, domain_type)
        assign_str_attr(f[f"/{matrix_name}"].attrs, h5_literals.DATA_TYPE_ATT_NAME, data_type_c)


def write_attributes(filename: str, file_description: Optional[str] = None) -> None:
    """
    Write attributes to a HDF5 file.

    This function writes attributes to a HDF5 file using a deprecated legacy method if legacy is set to True, or a new
    typed method if legacy is set to False. The function warns if legacy is set to True and deprecates it. If
    file_description is not provided, a default file description will be used.

    Args:
        filename: The name of the HDF5 file.
        file_description: The description of the file. If not provided, a default description
            will be used.

    """

    # get literals
    h5_literals = get_h5_literals()

    # get computer infor
    comp_info = dotdict(
        {
            "date": datetime.now().strftime("%d-%b-%Y"),
            "computer_name": socket.gethostname(),
            "operating_system_type": platform.system(),
            "operating_system": platform.system() + " " + platform.release() + " " + platform.version(),
            "user_name": os.environ.get("USERNAME"),
            "matlab_version": "N/A",
            "kwave_version": "1.3",
            "kwave_path": "N/A",
        }
    )

    # set file description if not provided by user
    if file_description is None:
        file_description = (
            f"Input data created by {comp_info.user_name} running MATLAB "
            f"{comp_info.matlab_version} on {comp_info.operating_system_type}"
        )

    # set additional file attributes
    with h5py.File(filename, "a") as f:
        # create a dictionary of attributes
        attributes = {
            h5_literals.FILE_MAJOR_VER_ATT_NAME: h5_literals.HDF_FILE_MAJOR_VERSION,
            h5_literals.FILE_MINOR_VER_ATT_NAME: h5_literals.HDF_FILE_MINOR_VERSION,
            h5_literals.CREATED_BY_ATT_NAME: f"k-Wave {kwave.VERSION}",
            h5_literals.FILE_DESCR_ATT_NAME: file_description,
            h5_literals.FILE_TYPE_ATT_NAME: h5_literals.HDF_INPUT_FILE,
            h5_literals.FILE_CREATION_DATE_ATT_NAME: get_date_string(),
        }
        # loop through the attributes dictionary and assign each attribute to the file
        for key, value in attributes.items():
            assign_str_attr(f.attrs, key, value)


def write_flags(filename):
    """
     writeFlags reads the input HDF5 file and derives and writes the
     required source and medium flags based on the datasets present in the
     file. For example, if the file contains a data set named 'BonA', the
     nonlinear_flag will be written as true. Conditional flags are also
     written. The source mode flags are written when appropriate if they
     are not already present in the file. The default source mode is
     'additive'.

     List of flags that are always written
         ux_source_flag
         uy_source_flag
         uz_source_flag
         sxx_source_flag
         sxy_source_flag
         sxz_source_flag
         syy_source_flag
         syz_source_flag
         szz_source_flag
         p_source_flag
         p0_source_flag
         transducer_source_flag
         nonuniform_grid_flag
         nonlinear_flag
         absorbing_flag
         axisymmetric_flag
         elastic_flag
         sensor_mask_type

     List of conditional flags
         u_source_mode
         u_source_many
         p_source_mode
         p_source_many
         s_source_mode
         s_source_many

    Args:
        filename:

    """

    # h5_literals = get_h5_literals()

    with h5py.File(filename, "r") as hf:
        names = hf.keys()

        v_list = [
            ("ux_source", "u_source_many"),
            ("uy_source", "u_source_many"),
            ("uz_source", "u_source_many"),
            ("sxx_source", "s_source_many"),
            ("syy_source", "s_source_many"),
            ("szz_source", "s_source_many"),
            ("sxy_source", "s_source_many"),
            ("sxz_source", "s_source_many"),
            ("syz_source", "s_source_many"),
            ("p_source", "p_source_many"),
        ]
        variable_list = {}
        for prefix, many_flag_key in v_list:
            inp_name = f"{prefix}_input"
            flag_name = f"{prefix}_flag"
            if inp_name in names:
                variable_list[flag_name] = hf[inp_name].shape[1]

                variable_list[many_flag_key] = hf[inp_name].shape[0] != 1
            else:
                variable_list[flag_name] = 0

        # --------------------
        # u source
        # --------------------

        # write u_source mode if not already in file (1 is Additive, 0 is Dirichlet)
        if any(variable_list[flag] for flag in ["ux_source_flag", "uy_source_flag", "uz_source_flag"]) and "u_source_mode" not in names:
            variable_list["u_source_mode"] = 1

        # --------------------
        # s source
        # --------------------

        # write s_source mode if not already in file (1 is Additive, 0 is Dirichlet)
        if (
            any(
                variable_list[flag]
                for flag in [
                    "sxx_source_flag",
                    "syy_source_flag",
                    "szz_source_flag",
                    "sxy_source_flag",
                    "sxz_source_flag",
                    "syz_source_flag",
                ]
            )
            and "s_source_mode" not in names
        ):
            variable_list["s_source_mode"] = 1

        # --------------------
        # p source
        # --------------------

        # write p_source mode if not already in file (1 is Additive, 0 is Dirichlet)
        if any(variable_list[flag] for flag in ["p_source_flag"]) and "p_source_mode" not in names:
            variable_list["p_source_mode"] = 1

        # check for p0_source_input and set p0_source_flag
        variable_list["p0_source_flag"] = "p0_source_input" in names

        # --------------------
        # additional flags
        # --------------------
        # check for transducer_source_input and set transducer_source_flag
        variable_list["transducer_source_flag"] = "transducer_source_input" in names

        # check for BonA and set nonlinear flag
        variable_list["nonlinear_flag"] = "BonA" in names

        # check for alpha_coeff and set absorbing flag
        variable_list["absorbing_flag"] = "alpha_coeff" in names

        # check for lambda and set elastic flag
        variable_list["elastic_flag"] = "lambda" in names

        # set axisymmetric grid flag to false
        variable_list["axisymmetric_flag"] = 0

        # set nonuniform grid flag to false
        variable_list["nonuniform_grid_flag"] = 0

        # check for sensor_mask_index and sensor_mask_corners
        if "sensor_mask_index" in names:
            variable_list["sensor_mask_type"] = 0
        elif "sensor_mask_corners" in names:
            variable_list["sensor_mask_type"] = 1
        else:
            raise ValueError("Either sensor_mask_index or sensor_mask_corners must be defined in the input file")

    # --------------------
    # write flags to file
    # --------------------

    # change all the index variables to be in 64-bit unsigned integers (long in C++) and write to file
    for key, value in variable_list.items():
        # cast matrix to 64-bit unsigned integer
        value = np.array(value, dtype=np.uint64)
        write_matrix(filename, value, key)
        del value


def write_grid(filename, grid_size, grid_spacing, pml_size, pml_alpha, Nt, dt, c_ref):
    """
    Creates and writes the wavenumber grids and PML variables
    required by the k-Wave C++ code to the HDF5 file specified by the
    user.

        List of parameters that are written:
            Nx
            Ny
            Nz
            Nt
            dt
            dx
            dy
            dz
            c_ref
            pml_x_alpha
            pml_y_alpha
            pml_z_alpha
            pml_x_size
            pml_y_size
            pml_z_size

    """

    h5_literals = get_h5_literals()

    # =========================================================================
    # STORE FLOATS
    # =========================================================================
    variable_list = {
        "dt": dt,
        "dx": grid_spacing[0],
        "dy": grid_spacing[1],
        "dz": grid_spacing[2],
        "pml_x_alpha": pml_alpha[0],
        "pml_y_alpha": pml_alpha[1],
        "pml_z_alpha": pml_alpha[2],
        "c_ref": c_ref,
    }

    # change float variables to be in single precision (float in C++), then add to HDF5 file
    for key, value in variable_list.items():
        # cast matrix to single precision
        value = cast_to_type(value, h5_literals.MATRIX_DATA_TYPE_MATLAB)
        write_matrix(filename, value, key)
        del value

    # =========================================================================
    # STORE INTEGERS
    # =========================================================================

    # integer variables
    variable_list = {
        "Nx": grid_size[0],
        "Ny": grid_size[1],
        "Nz": grid_size[2],
        "Nt": Nt,
        "pml_x_size": pml_size[0],
        "pml_y_size": pml_size[1],
        "pml_z_size": pml_size[2],
    }

    # change all the index variables to be in 64-bit unsigned integers (long in C++)
    for key, value in variable_list.items():
        # cast matrix to 64-bit unsigned integer
        value = cast_to_type(value, h5_literals.INTEGER_DATA_TYPE_MATLAB)
        write_matrix(filename, value, key)
        del value


def assign_str_attr(attrs, attr_name, attr_val):
    """
    Assigns HDF5 attribute with value as a fixed-length string

    Args:
        attrs: HDF5 attribute object
        attr_name: name of attribute
        attr_val: value of attribute

    """
    attrs.create(attr_name, attr_val, None, dtype=f"<S{len(attr_val)}")


def load_image(path, is_gray):
    if is_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        raise NotImplementedError
        # im = squeeze(double(im(:, :, 1)) + double(im(:, :, 2)) + double(im(:, :, 3)));
    img = img.astype(float)

    # scale pixel values from 0 -> 1
    img = img.max() - img
    img = img * (1 / img.max())
    return img
