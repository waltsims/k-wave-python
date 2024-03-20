import logging
import time

import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Dict, Union
from jaxtyping import Float

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_SI, scale_time
from kwave.utils.filters import next_pow2
from kwave.utils.matrix import expand_matrix
from kwave.utils.tictoc import TicToc


@typechecker
def angular_spectrum_cw(
    input_plane: Float[np.ndarray, "Dim1 Dim2"],
    dx: float,
    z_pos: float,
    f0: int,
    medium: Union[Dict, int],
    angular_restriction: bool = True,
    grid_expansion: int = 0,
    fft_length: str = "auto",
    data_cast: str = "off",
    data_recast: bool = False,
    reverse_proj: bool = False,
    absorbing: bool = False,
    loops_for_time_est: int = 5,
):
    """
    Projects a 2D input plane (given as a 3D matrix of
    time series at each spatial position) using the angular spectrum
    method. The time series are decomposed into spectral components and
    then each frequency is propagated using the spectral propagator with
    angular restriction described in reference [1].


    Args:
      input_plane: 3D matrix containing the time varying pressure
      over a 2D input plane indexed as (x, y, t) [Pa].
      dx: Spatial step between grid points in the input plane [m].
      z_pos: Vector specifying the relative z-position of
      f0: Source frequency [Hz]
      medium: Medium object.
      angular_restriction: Boolean controlling whether angular
      restriction is used as described in [1] (default = True).
      grid_expansion: Scalar value controlling the expansion of the input
      grid prior to computation (default = 0).
      fft_length: String or integer controlling the length of the FFT used for
      the angular spectrum method. The options are 'auto', which
      uses the next power of 2 greater than or equal to the length
      of the input data, or a specific integer value. The FFT
      length has a significant impact on the computation time and
      memory requirements (default = 'auto').
      data_cast: String input of the data type that variables
      are cast to before computation. For example,
      setting to 'single' will speed up the computation time
      (due to the improved efficiency of fft2 and ifft2)
      but may introduce numerical errors. The default value is 'off'
      which performs no data casting (default = 'off').
      data_recast: Boolean controlling whether the output data is recast to the
      original data type after computation (default = False).
      reverse_proj: Boolean controlling whether the projection is performed
      in the opposite direction (default = False).
      absorbing: Boolean controlling whether the input data has an
      absorbing boundary applied prior to computation (default = False).
      loops_for_time_est: Integer controlling the number of loops used to
      estimate the computation time. This is used to
      display an estimated time remaining during
      computation (default = 5).

     References:
       [1] Zeng, X., & McGough, R. J. (2008). Evaluation of the angular
            spectrum approach for simulations of near-field pressures.
            The Journal of the Acoustical Society of America, 123(1), 68-76.
    """
    TicToc.tic()
    # check list of valid inputs
    if not isinstance(data_cast, str):
        raise ValueError("Optional input 'data_cast' must be a string.")
    elif data_cast not in ["off", "double", "single", "gpuArray-single", "gpuArray-double"]:
        raise ValueError("Invalid input for 'data_cast'.")

    # replace double with off
    if data_cast == "double":
        data_cast = "off"

    # create empty string to hold extra cast variable for use
    # with the parallel computing toolbox
    data_cast_prepend = ""

    # replace PCT options with gpuArray
    if data_cast == "gpuArray-single":
        data_cast = "gpuArray"
        data_cast_prepend = "single"
    elif data_cast == "gpuArray-double":
        data_cast = "gpuArray"

    if data_cast == "gpuArray":
        raise NotImplementedError("processing with GPU is not supported in the Python implementation of the kWave")

    # check for structured medium input
    if isinstance(medium, dict):
        # force the sound speed to be defined
        if "sound_speed" not in medium:
            raise ValueError("medium.sound_speed must be defined when specifying medium properties using a dictionary.")

        # assign the sound speed
        c0 = medium["sound_speed"]

        # assign the absorption
        if "alpha_coeff" in medium or "alpha_power" in medium:
            # enforce both absorption parameters
            if "alpha_coeff" not in medium or "alpha_power" not in medium:
                raise ValueError("Both medium.alpha_coeff and medium.alpha_power must be defined for an absorbing medium.")

            # convert attenuation to Np/m
            alpha_Np = db2neper(alpha=medium["alpha_coeff"], y=medium["alpha_power"]) * (2 * np.pi * f0) ** medium["alpha_power"]

            # check for zero absorption and assign flag
            if alpha_Np != 0:
                absorbing = True
    else:
        # assign the sound speed
        c0 = medium

    # check for maximum supported frequency
    if dx > (c0 / (2 * f0)):
        raise ValueError(f"Input frequency is higher than maximum supported frequency of {scale_SI(c0 / (2 * dx))}Hz.")

    # get grid size
    Nx, Ny = input_plane.shape
    z_pos = np.atleast_1d(z_pos)
    Nz = len(z_pos)

    # get scale factor for grid size
    _, scale, prefix, _ = scale_SI(min(Nx * dx, Ny * dx))

    # update command line status
    logging.log(logging.INFO, "Running CW angular spectrum projection...")
    logging.log(logging.INFO, f"  start time: {TicToc.start_time}")
    logging.log(logging.INFO, f"  input plane size: {Nx} by {Ny} grid points ({scale * Nx * dx} by {scale * Ny * dx} {prefix}m)")
    logging.log(logging.INFO, f"  grid expansion: {grid_expansion} grid points")

    # apply phase conjugation if stepping backwards
    if reverse_proj:
        input_plane = np.conj(input_plane)

    # expand input
    if grid_expansion > 0:
        input_plane = expand_matrix(input_plane, [grid_expansion, grid_expansion], 0)
        Nx, Ny = input_plane.shape

    # get FFT size
    if isinstance(fft_length, str) and fft_length == "auto":
        fft_length = int(2 ** (next_pow2(max([Nx, Ny])) + 1))

    # update command line status
    logging.log(logging.INFO, f"  FFT size: {fft_length} points")
    logging.log(logging.INFO, f"  maximum supported frequency: {scale_SI(c0 / (2 * dx))}Hz")

    # create wavenumber vector
    N = fft_length
    if N % 2 == 0:
        k_vec = np.arange(-N // 2, N // 2) * 2 * np.pi / (N * dx)
    else:
        k_vec = np.arange(-(N - 1) // 2, (N - 1) // 2 + 1) * 2 * np.pi / (N * dx)

    # force middle value
    # force middle value to be zero in case 1/Nx is a recurring
    # number and the series doesn't give exactly zero
    k_vec[N // 2] = 0

    # shift wavenumbers to be in the correct order for FFTW
    k_vec = np.fft.ifftshift(k_vec)

    # compute wavenumber at driving frequency
    k = 2 * np.pi * f0 / c0

    # create wavenumber grids
    ky, kx = np.meshgrid(k_vec, k_vec, indexing="ij")
    kz = np.sqrt(k**2 - (kx**2 + ky**2).astype(complex))

    # precompute term for angular restriction
    sqrt_kx2_ky2 = np.sqrt(kx**2 + ky**2)

    # preallocate maximum pressure output
    pressure = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    pressure[:, :, 0] = input_plane

    # compute forward Fourier transform of input plane
    input_plane_fft = np.fft.fft2(input_plane, (fft_length, fft_length))

    # =========================================================================
    # DATA CASTING
    # =========================================================================

    if data_cast != "off":
        logging.log(logging.INFO, f"  casting variables to {data_cast} type...")

        # List of variables to cast
        cast_variables = ["kz", "z_pos", "input_plane_fft", "pressure"]

        # Additional variables used if absorbing
        if absorbing:
            cast_variables.extend(["alpha_Np", "k"])

        # Additional variables used for angular restriction
        if angular_restriction:
            cast_variables.extend(["sqrt_kx2_ky2", "fft_length", "dx"])

        # Loop through, and change data type
        for var_name in cast_variables:
            exec(f"{var_name} = {data_cast}({data_cast_prepend}({var_name}))")

    # =========================================================================
    # Z-LOOP
    # =========================================================================

    # Update command line status
    logging.log(logging.INFO, f"  precomputation completed in {scale_time(TicToc.toc())}")
    logging.log(logging.INFO, "  starting z-step loop...")

    # Loop over z-positions
    for z_index in range(Nz):
        # Get current z value
        z = z_pos[z_index]

        # If set to zero, just store the input plane
        if z == 0:
            # Store input data
            pressure[:, :, z_index] = input_plane

        else:
            # compute spectral propagator (Eq. 6)
            H = np.conj(np.exp(1j * z * kz))

            # account for attenuation (Eq. 11)
            if absorbing:
                H = H * np.exp(-alpha_Np * z * k / kz)

            # apply angular restriction
            if angular_restriction:
                # size of computational domain [m]
                D = (fft_length - 1) * dx

                # compute angular threshold (Eq. 10)
                kc = k * np.sqrt(0.5 * D**2 / (0.5 * D**2 + z**2))

                # apply threshold to propagator
                H[sqrt_kx2_ky2 > kc] = 0

            # compute projected field and store
            pressure_step = np.fft.ifft2(input_plane_fft * H, (fft_length, fft_length))
            pressure[:, :, z_index] = pressure_step[:Nx, :Ny]

        # Update command line status
        if z_index == loops_for_time_est:
            est_sim_time = scale_time(TicToc.toc() * Nz / z_index)
            logging.log(logging.INFO, f"  estimated simulation time {est_sim_time}  ...")

    # update command line status
    logging.log(logging.INFO, f"  simulation completed in {time.perf_counter() - TicToc.start_time}")

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # POST PROCESSING
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

    # Measure total computation time
    total_computation_time = TicToc.toc()

    # Trim grid expansion
    if grid_expansion > 0:
        pressure = pressure[
            grid_expansion:-grid_expansion,
            grid_expansion:-grid_expansion,
            :,
        ]

    # Reverse time signals and grid if stepping backwards
    if reverse_proj:
        np.flip(np.conj(pressure), axis=2)

    # Cast output back to double precision
    if data_recast:
        # Note: not exactly the same as Matlab implementation
        pressure = float(pressure)

    # Update command line status
    logging.log(logging.INFO, f"Total computation time: {total_computation_time:.2f} seconds")

    return pressure
