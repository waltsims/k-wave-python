
import numpy as np

from kwave.kgrid import kWaveGrid


def bioheat_exact(
    T0,
    S,
    material,
    dx,
    t,
    device="cpu",
    return_numpy=False,
):
    """
    Fourier solution of Pennes' bioheat equation for a constant source.

    Uses NumPy on the CPU or CuPy on the GPU. Does not require PyTorch.

    Equation:
        dT/dt = D * Laplacian(T) - P * (T - Ta) + S

    Parameters
    ----------
    T0 : array-like
        Initial temperature field, 1D, 2D, or 3D.
    S : array-like or scalar
        Constant heat source [K/s], or scalar zero.
    material : array-like
        [D, P, Ta], where D is thermal diffusivity [m²/s],
        P is the perfusion coefficient [1/s], and Ta is
        the reference temperature [K].
    dx : float or sequence
        Grid spacing in metres.
    t : float
        Evaluation time in seconds.
    device : {"cpu", "gpu"}
        NumPy CPU execution or CuPy GPU execution.
    return_numpy : bool
        Return a NumPy array on the CPU, including when device="gpu".

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Temperature field at time t.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'.")

    # Select the array backend.
    if device == "gpu":
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "GPU execution requires CuPy and a compatible CUDA runtime."
            ) from exc

        try:
            cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise RuntimeError(
                "No usable CUDA GPU was found for CuPy."
            ) from exc

        xp = cp
    else:
        xp = np

    T0_np = np.asarray(T0)

    if T0_np.ndim not in (1, 2, 3) or T0_np.size <= 1:
        raise ValueError("T0 must be a non-scalar 1D, 2D, or 3D array.")

    S_np = np.asarray(S)
    source_is_zero = S_np.ndim == 0 and S_np.item() == 0

    if not source_is_zero and S_np.shape != T0_np.shape:
        raise ValueError("S must be zero or have the same shape as T0.")

    material = np.asarray(material, dtype=float).ravel()
    if material.size != 3:
        raise ValueError("material must contain [D, P, Ta].")

    D, P, Ta = map(float, material)

    if not np.all(np.isfinite(material)) or D < 0 or P < 0:
        raise ValueError("Material parameters must be finite; D and P must be non-negative.")

    t = float(t)
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and non-negative.")

    if np.isscalar(dx):
        spacing = [float(dx)] * T0_np.ndim
    else:
        spacing = np.asarray(dx, dtype=float).ravel().tolist()

    if len(spacing) != T0_np.ndim:
        raise ValueError(f"dx must be a scalar or contain {T0_np.ndim} values.")

    if not np.all(np.isfinite(spacing)) or any(d <= 0 for d in spacing):
        raise ValueError("Grid spacing must be finite and positive.")

    # Transfer fields to the selected array backend.
    T0_arr = xp.asarray(T0_np, dtype=xp.float64)
    S_arr = None if source_is_zero else xp.asarray(S_np, dtype=xp.float64)

    # Build squared angular wavenumber magnitude in FFT ordering.
    # No fftshift/ifftshift is required with fftfreq.
    k_axes = [
        2.0 * np.pi * xp.fft.fftfreq(n, d=d)
        for n, d in zip(T0_np.shape, spacing)
    ]

    k_mesh = xp.meshgrid(*k_axes, indexing="ij")
    k_squared = sum(ki**2 for ki in k_mesh)

    denominator = D * k_squared + P
    decay = xp.exp(-denominator * t)

    # Stable evaluation of (1 - exp(-a*t)) / a.
    # Handle a == 0 explicitly.
    nonzero = denominator != 0
    safe_denominator = xp.where(nonzero, denominator, 1.0)
    Q = xp.where(
        nonzero,
        -xp.expm1(-denominator * t) / safe_denominator,
        t,
    )

    # Fourier solution.
    T0_fft = xp.fft.fftn(T0_arr - Ta)

    if S_arr is None:
        T_fft = decay * T0_fft
    else:
        T_fft = decay * T0_fft + Q * xp.fft.fftn(S_arr)

    T = xp.fft.ifftn(T_fft).real + Ta

    if return_numpy:
        return xp.asnumpy(T) if device == "gpu" else T

    return T
