import logging
import time

import numpy as np
from matplotlib import pyplot as plt
from beartype import beartype as typechecker
from beartype.typing import Dict, Union
from jaxtyping import Float

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_SI, scale_time
from kwave.utils.filters import next_pow2
from kwave.utils.matrix import expand_matrix
from kwave.utils.tictoc import TicToc


@typechecker
def angular_spectrum(
    input_plane: Float[np.ndarray, "Dim1 Dim2 Dim3"],
    dx: float,
    dt: float,
    z_pos: float,
    medium: Union[Dict, int],
    angular_restriction: bool = True,
    grid_expansion: int = 0,
    fft_length: str = "auto",
    data_cast: str = "off",
    data_recast: bool = False,
    reverse_proj: bool = False,
    absorbing: bool = False,
    plot_updates: bool = False,
    loops_for_time_est: int = 5,
    record_time_series: bool = False,
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
      dt: Temporal step between time points in the input plane [s].
      z_pos: Vector specifying the relative z-position of
      the planes to which the data is projected [m].
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
      plot_updates: Boolean controlling whether a plot is shown during
      computation to display the progress (default = False).
      loops_for_time_est: Integer controlling the number of loops used to
      estimate the computation time. This is used to
      display an estimated time remaining during
      computation (default = 5).
      record_time_series: Boolean controlling whether the time series data is recorded

    Examples:
      >>>    pressure_max = angularSpectrum(input_plane, dx, dt, z_pos, c0)
      >>>    pressure_max = angularSpectrum(input_plane, dx, dt, z_pos, medium)
      >>>    (pressure_max, pressure_time) = angularSpectrum(input_plane, dx, dt, z_pos, c0)
      >>>    (pressure_max, pressure_time) = angularSpectrum(input_plane, dx, dt, z_pos, medium)

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

            # assign flag
            absorbing = True

    else:
        # assign the sound speed
        c0 = medium

    # check time step is sufficient
    if (c0 * dt / dx) > 1:
        raise ValueError(
            "Maximum supported frequency in temporal sampling is lower than maximum supported frequency in spatial sample (CFL > 1)."
        )

    # get grid size
    Nx, Ny, Nt = input_plane.shape
    z_pos = np.atleast_1d(z_pos)
    Nz = len(z_pos)

    # turn off plotting if only one value of z_pos
    if z_pos.size == 1:
        plot_updates = False

    # get scale factor for grid size
    scale, scale, prefix, _ = scale_SI(min(Nx * dx, Ny * dx))

    # update command line status
    logging.log(logging.INFO, "Running angular spectrum projection...")
    logging.log(logging.INFO, f"  start time: {TicToc.start_time}")
    logging.log(logging.INFO, f"  input plane size: {Nx} by {Ny} grid points ({scale * Nx * dx} by {scale * Ny * dx} {prefix}m)")
    logging.log(logging.INFO, f"  grid expansion: {grid_expansion} grid points")

    # reverse time signals if stepping backwards
    if reverse_proj:
        input_plane = np.flip(input_plane, 2)

    # expand input
    if grid_expansion > 0:
        input_plane = expand_matrix(input_plane, [grid_expansion, grid_expansion, 0], 0)
        Nx, Ny, Nt = input_plane.shape

    # get FFT size
    if isinstance(fft_length, str) and fft_length == "auto":
        fft_length = int(2 ** (next_pow2(max([Nx, Ny])) + 1))

    # update command line status
    logging.log(logging.INFO, f"  FFT size: {fft_length} points")
    logging.log(logging.INFO, f"  maximum supported frequency: {scale_SI(c0 / (2 * dx))}Hz")
    logging.log(logging.INFO, f"  input signal length: {Nt} time points ({scale_SI(Nt * dt)}s)")

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

    # create wavenumber grids
    ky, kx = np.meshgrid(k_vec, k_vec, indexing="ij")

    # precompute term for angular restriction
    sqrt_kx2_ky2 = np.sqrt(kx**2 + ky**2)

    # preallocate maximum pressure output
    pressure_max = np.zeros((Nx, Ny, Nz))

    # preallocate time series output
    if record_time_series:
        pressure_time = np.zeros((Nx, Ny, Nt, Nz))

    # Compute the FFT
    input_plane_w_fft = np.fft.fft(input_plane, axis=2)

    # Reduce to a single sided spectrum where the number of unique points for
    # even numbered FFT lengths is given by N/2 + 1, and for odd (N + 1)/2
    num_unique_pts = int(np.ceil((Nt + 1) / 2))
    input_plane_w_fft = input_plane_w_fft[:, :, :num_unique_pts]

    # Create the frequency axis variable
    f_vec = np.arange(input_plane_w_fft.shape[2]) / (dt * Nt)

    # Compute frequencies to propagate
    f_vec_prop = f_vec[f_vec < (c0 / (2 * dx))]

    # Preallocate loop variable
    pressure_time_step = np.zeros((Nx, Ny, len(f_vec)), dtype=np.complex128)

    if data_cast != "off":
        logging.log(logging.INFO, f"  casting variables to {data_cast} type...")

        # List of variables to cast
        cast_variables = ["kx", "ky", "z_pos", "input_plane_w_fft", "pressure_max", "pressure_time_step"]

        # Additional variable if storing snapshots
        if record_time_series:
            cast_variables.append("pressure_time")

        # Additional variables used if absorbing
        if absorbing:
            cast_variables.append("k")

        # Additional variables used for angular restriction
        if angular_restriction:
            cast_variables.extend(["sqrt_kx2_ky2", "fft_length", "dx"])

        # Loop through, and change data type
        for var_name in cast_variables:
            exec(f"{var_name} = {data_cast}({data_cast_prepend}({var_name}))")

    # Open figure window
    if plot_updates:
        sim_fig = plt.figure()
        plt.show()

    # Update command line status
    logging.log(logging.INFO, f"  precomputation completed in {scale_time(TicToc.toc())}")
    logging.log(logging.INFO, "  starting z-step loop...")

    # Loop over z-positions
    for z_index in range(Nz):
        # Get current z value
        z = z_pos[z_index]

        # If set to zero, just store the input plane
        if z == 0:
            # Store maximum pressure
            pressure_max[:, :, z_index] = np.max(input_plane, axis=2)

            # Store time series data if required
            if record_time_series:
                pressure_time[:, :, :, z_index] = input_plane

        else:
            # Loop over frequencies
            for f_index in range(len(f_vec_prop)):
                # Compute wavenumber at driving frequency
                k = 2 * np.pi * f_vec[f_index] / c0

                # Compute wavenumber grid
                kz = np.sqrt(k**2 - ((kx**2) + (ky**2)).astype(complex))

                # Compute spectral propagator (Eq. 6)
                H = np.conj(np.exp(1j * z * kz))

                # Account for attenuation (Eq. 11)
                if absorbing:
                    # Convert attenuation to Np/m
                    alpha_Np = db2neper(medium.alpha_coeff, medium.alpha_power) * (2 * np.pi * f_vec[f_index]) ** medium.alpha_power

                    # Apply attenuation to propagator
                    if alpha_Np != 0:
                        H = H * np.exp(-alpha_Np * z * k / kz)

                # Apply angular restriction
                if angular_restriction:
                    # Size of computational domain [m]
                    D = (fft_length - 1) * dx

                    # Compute angular threshold (Eq. 10)
                    kc = k * np.sqrt(0.5 * D**2 / (0.5 * D**2 + z**2))

                    # Apply threshold to propagator
                    H[sqrt_kx2_ky2 > kc] = 0

                # Compute forward Fourier transform of input plane
                input_plane_xy_fft = np.fft.fft2(input_plane_w_fft[:, :, f_index], s=(fft_length, fft_length))

                # Compute phase shift for retarded time
                ret_time = np.exp(1j * 2 * np.pi * f_vec[f_index] * z / c0)

                # Compute projected field
                # Compute projected field
                pressure_step = np.fft.ifft2(input_plane_xy_fft * H, s=(fft_length, fft_length))
                pressure_time_step[:, :, f_index] = pressure_step[:Nx, :Ny] * ret_time

            # Form double sided amplitude spectrum in correct order for FFTW
            # (FFT of real data is conjugate symmetric)
            if Nt % 2:
                pressure_time_step_exp = np.concatenate(
                    (pressure_time_step, np.flip(np.conj(pressure_time_step[:, :, 1:]), axis=2)), axis=2
                )
            else:
                pressure_time_step_exp = np.concatenate(
                    (pressure_time_step, np.flip(np.conj(pressure_time_step[:, :, 1:-2]), axis=2)), axis=2
                )

            # Take inverse Fourier transform to recover time domain data
            pressure_time_step_exp = np.real(np.fft.ifft(pressure_time_step_exp, axis=2))

            # Store maximum pressure
            pressure_max[:, :, z_index] = np.max(pressure_time_step_exp, axis=2)

            # Store time series data if required
            if record_time_series:
                pressure_time[:, :, :, z_index] = pressure_time_step_exp

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

    # Close plot
    if plot_updates:
        plt.close(sim_fig)

    # Trim grid expansion
    if grid_expansion > 0:
        pressure_max = pressure_max[
            grid_expansion:-grid_expansion,
            grid_expansion:-grid_expansion,
            :,
        ]
        if record_time_series:
            pressure_time = pressure_time[
                grid_expansion:-grid_expansion,
                grid_expansion:-grid_expansion,
                :,
                :,
            ]

    # Reverse time signals and grid if stepping backwards
    if reverse_proj:
        pressure_max = np.flip(pressure_max, axis=2)
        if record_time_series:
            pressure_time = np.flip(pressure_time, axis=2)

    # Cast output back to double precision
    if data_recast:
        # Recast data
        pressure_max = np.double(pressure_max)
        if record_time_series:
            pressure_time = np.double(pressure_time)

    # Update command line status
    logging.log(logging.INFO, f"Total computation time: {total_computation_time:.2f} seconds")

    # Assign outputs
    if record_time_series:
        return pressure_max, pressure_time
    else:
        return pressure_max
