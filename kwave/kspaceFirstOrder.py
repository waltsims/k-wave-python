import copy
import warnings
from typing import Optional, Union

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.utils.matrix import expand_matrix
from kwave.utils.pml import get_optimal_pml_size


def _normalize_pml(val, ndim, name="pml_size"):
    if isinstance(val, str):
        valid = "an integer, tuple of integers, or 'auto'" if name == "pml_size" else "a float or tuple of floats"
        raise ValueError(f"{name} must be {valid}, got {val!r}")
    if isinstance(val, (int, float)):
        return (val,) * ndim
    t = tuple(val)
    if len(t) == 0 or len(t) > ndim:
        raise ValueError(f"{name} must be a scalar or 1-to-{ndim}-element sequence, got {len(t)} elements")
    return t + (t[-1],) * (ndim - len(t))


def _is_cartesian_mask(mask, ndim):
    """True if mask is a Cartesian coordinate array (ndim, n_points)."""
    arr = np.asarray(mask)
    return arr.ndim == 2 and arr.shape[0] == ndim


def _expand_obj(obj, attrs, pml_size, edge_val=None):
    """Shallow-copy obj and expand grid-shaped array attributes via expand_matrix.

    Scalars (size == 1) are intentionally skipped — they broadcast to any grid
    shape and don't need expansion (e.g. homogeneous sound_speed=1500).
    """
    exp_coeff = list(pml_size)
    out = copy.copy(obj)
    for attr in attrs:
        val = getattr(out, attr, None)
        if val is not None:
            arr = np.asarray(val)
            if arr.size > 1:  # scalar/homogeneous values broadcast — skip
                setattr(out, attr, expand_matrix(arr, exp_coeff, edge_val))
    return out


def _expand_for_pml_outside(kgrid, medium, source, sensor, pml_size):
    """Expand grid/medium/source/sensor so PML sits outside the user domain."""
    new_N = tuple(int(n) + 2 * int(p) for n, p in zip(kgrid.N, pml_size))
    expanded_kgrid = kWaveGrid(new_N, kgrid.spacing)
    expanded_kgrid.setTime(kgrid.Nt, kgrid.dt)

    expanded_medium = _expand_obj(medium, ("sound_speed", "sound_speed_ref", "density", "alpha_coeff", "BonA"), pml_size)
    expanded_medium = _expand_obj(expanded_medium, ("alpha_filter",), pml_size, edge_val=0)  # zero-pad, not edge-extend
    # s_mask intentionally excluded — stress sources are not yet implemented (raises NotImplementedError)
    expanded_source = _expand_obj(source, ("p0", "p_mask", "u_mask"), pml_size, edge_val=0)

    expanded_sensor = sensor
    if sensor is not None and getattr(sensor, "mask", None) is not None:
        if not _is_cartesian_mask(sensor.mask, kgrid.dim):
            expanded_sensor = copy.copy(sensor)
            expanded_sensor.mask = expand_matrix(np.asarray(sensor.mask), list(pml_size), 0)

    return expanded_kgrid, expanded_medium, expanded_source, expanded_sensor


_FULL_GRID_SUFFIXES = ("_max", "_min", "_rms", "_max_all", "_min_all", "_rms_all")
# Python solver pre-strips _final fields; C++ binary does not
_FULL_GRID_SUFFIXES_CPP = ("_final",) + _FULL_GRID_SUFFIXES


def _strip_pml(result, pml_size, ndim, suffixes=_FULL_GRID_SUFFIXES):
    """Remove PML padding from full-grid fields in the result dict."""
    slices = tuple(slice(int(p), -int(p) if int(p) else None) for p in pml_size[:ndim])
    return {
        key: val[slices] if isinstance(val, np.ndarray) and val.ndim == ndim and any(key.endswith(s) for s in suffixes) else val
        for key, val in result.items()
    }


def _resolve_dtype(value):
    """Normalize a dtype-like input to ``np.float32`` or ``np.float64``.

    Accepts numpy dtypes/types (``np.float32``, ``np.float64``), strings
    (``"float32"`` etc., plus MATLAB aliases ``"single"`` / ``"double"``),
    Python ``float``, ``None`` (default → float64), and the legacy MATLAB
    ``"off"`` alias for float64.  Anything that resolves to a non-float32 /
    non-float64 dtype raises ``ValueError`` — the solver isn't validated
    for ``float16`` / complex dtypes.

    Cupy dtypes (``cp.float32``, ``cp.float64``) work for free because cupy
    re-exports numpy's scalar types.  Torch / JAX dtypes are not accepted —
    they live in different ecosystems and don't translate via ``np.dtype()``;
    the error message points the user at the equivalent numpy dtype.
    """
    if value is None or value == "off":
        return np.float64
    try:
        resolved = np.dtype(value).type
    except TypeError as e:
        framework = getattr(type(value), "__module__", "").split(".")[0]
        hint = ""
        if framework in ("torch", "jax", "jaxlib", "tensorflow"):
            hint = f" {framework}.dtype objects aren't supported; pass the equivalent numpy dtype (np.float32 / np.float64)."
        raise ValueError(
            f"dtype must be a numpy dtype, type, or string (e.g. 'float32', 'single'), got {value!r}.{hint}"
        ) from e
    if resolved is np.float32:
        return np.float32
    if resolved is np.float64:
        return np.float64
    raise ValueError(f"dtype must resolve to float32 or float64; got {resolved.__name__} from {value!r}")


def kspaceFirstOrder(
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    source: kSource,
    sensor: Union[kSensor, NotATransducer, None] = None,
    *,
    pml_size: Union[int, tuple, str] = 20,
    pml_alpha: Union[float, tuple] = 2.0,
    pml_inside: bool = False,
    use_sg: bool = True,
    use_kspace: bool = True,
    smooth_p0: bool = True,
    backend: str = "python",
    device: str = "cpu",
    dtype=None,
    save_only: bool = False,
    data_path: Optional[str] = None,
    quiet: bool = False,
    debug: bool = False,
    num_threads: Optional[int] = None,
    device_num: Optional[int] = None,
    binary_path: Optional[str] = None,
) -> dict:
    """Run a k-Wave simulation.

    Unified entry point replacing the legacy kspaceFirstOrder2D / 3D
    functions.  Works with 1-D, 2-D, and 3-D grids.

    Args:
        kgrid: Simulation grid (defines dimensionality, spacing, and time
            steps).
        medium: Acoustic medium properties (sound speed, density, absorption,
            nonlinearity).
        source: Pressure and/or velocity source terms.
        sensor: Sensor mask defining where the field is recorded.  ``None``
            records the entire grid.

    Keyword Args:
        pml_size: Perfectly-matched-layer thickness in grid points.  A scalar
            applies to all dimensions; a tuple sets each dimension
            independently.  ``"auto"`` selects an optimal size via FFT-based
            analysis.  Default ``20``.
        pml_alpha: PML absorption coefficient (Nepers per grid point).
            Scalar or per-dimension tuple.  Default ``2.0``.
        pml_inside: When ``False`` (default), the grid is automatically
            expanded by ``2 * pml_size`` so the PML sits outside the user
            domain; full-grid output fields (``_final``, ``_max``, etc.)
            are cropped back to the original size.  When ``True``, the PML
            occupies the outermost grid points of the user-supplied grid,
            which saves memory but means PML absorption will modify field
            values near the domain boundary.
        use_sg: Use a staggered grid for velocity fields.  Default ``True``.
        use_kspace: Apply the k-space correction to the time-stepping scheme.
            Default ``True``.
        smooth_p0: Smooth the initial pressure distribution to suppress
            staircasing artifacts.  Default ``True``.
        backend: Simulation engine.  ``"python"`` runs a pure-Python /
            NumPy / CuPy solver; ``"cpp"`` serializes to HDF5 and invokes
            the compiled C++ binary.  Default ``"python"``.
        device: ``"cpu"`` or ``"gpu"``.  For ``backend="python"`` this
            selects NumPy (cpu) vs CuPy (gpu).  For ``backend="cpp"`` it
            selects the OMP vs CUDA binary.  Default ``"cpu"``.
        dtype: Numerical precision for state arrays in the Python backend.
            Accepts dtype-like input — a numpy dtype (``np.float32``,
            ``np.float64``; cupy aliases like ``cp.float32`` work since cupy
            re-exports numpy's scalar types), a string (``"float32"``,
            ``"float64"``, ``"single"``, ``"double"``), a Python type
            (``float``), or ``None`` for the default (float64).  The
            MATLAB-style alias ``"off"`` is accepted as a synonym for
            float64 to ease migration from the legacy
            ``SimulationOptions.data_cast``.  Torch / JAX dtypes are not
            accepted; pass the numpy equivalent (e.g. ``np.float32`` for
            ``torch.float32``).
            ``np.float32`` uses roughly half the memory and is faster on
            most hardware, at the cost of reduced numerical accuracy.
            Only ``float32`` and ``float64`` are supported; other dtypes
            raise ``ValueError``.  Has no effect on ``backend="cpp"`` (the
            C++ binary uses fixed internal precision regardless); a warning
            is emitted if ``dtype`` resolves to anything other than float64
            with the C++ backend.  Default ``None`` (float64).
        save_only: When ``True`` (``backend="cpp"`` only), write the HDF5
            input file and return without running the binary.  Useful for
            cluster submission.  Default ``False``.
        data_path: Directory for HDF5 input/output files (``backend="cpp"``
            only).  If ``None`` a temporary directory is created and cleaned
            up automatically after the run.  Set explicitly to inspect or
            reuse the HDF5 files.  Default ``None``.
        quiet: Suppress progress output.  Default ``False``.
        debug: Print detailed diagnostic output.  Default ``False``.
        num_threads: Thread count for the C++ OMP binary.  ``None`` uses all
            available cores.  Default ``None``.
        device_num: GPU device index for CUDA execution.  Default ``None``.
        binary_path: Path to a custom C++ binary.  When ``None`` (default),
            the binary bundled with ``k-wave-data`` is used.  Only applies
            when ``backend="cpp"``.

    Returns:
        dict: Recorded sensor data keyed by field name (e.g.
        ``"p"``, ``"p_final"``, ``"ux"``, ``"uy"``).

        All time-series are ``(n_sensor, Nt)`` with sensor points in
        C-flattened order.  Use :func:`reshape_to_grid` to recover spatial
        structure for full-grid masks.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if backend not in ("python", "cpp"):
        raise ValueError(f"Unknown backend: {backend!r}. Use 'python' or 'cpp'.")
    dtype = _resolve_dtype(dtype)

    if isinstance(pml_size, str) and pml_size.lower() == "auto":
        pml_size = tuple(int(x) for x in get_optimal_pml_size(kgrid))
    pml_size = _normalize_pml(pml_size, kgrid.dim)
    pml_alpha = _normalize_pml(pml_alpha, kgrid.dim, "pml_alpha")

    from kwave.solvers.validation import validate_simulation

    validate_simulation(kgrid, medium, source, sensor, pml_size=pml_size)

    # --- Shared pre-processing (both backends) ---

    if not pml_inside:
        kgrid, medium, source, sensor = _expand_for_pml_outside(kgrid, medium, source, sensor, pml_size)

    # Smooth initial pressure (after expansion so Blackman window covers PML transition)
    if smooth_p0 and source.p0 is not None and kgrid.dim >= 2:
        from kwave.utils.filters import smooth

        source = copy.copy(source)
        source.p0 = smooth(np.asarray(source.p0, dtype=float).reshape(tuple(int(n) for n in kgrid.N)), restore_max=True)

    # --- Backend dispatch ---

    if backend == "python":
        from kwave.solvers.kspace_solver import Simulation

        result = Simulation(
            kgrid,
            medium,
            source,
            sensor,
            device=device,
            use_sg=use_sg,
            use_kspace=use_kspace,
            smooth_p0=False,
            pml_size=pml_size,
            pml_alpha=pml_alpha,
            quiet=quiet,
            dtype=dtype,
        ).run()

    elif backend == "cpp":
        from kwave.solvers.cpp_simulation import CppSimulation
        from kwave.utils.checks import check_alpha_mode_cpp_compatible, warn_alpha_power_near_unity_cpp

        check_alpha_mode_cpp_compatible(medium)
        warn_alpha_power_near_unity_cpp(medium)

        if dtype is not np.float64:
            warnings.warn(
                f"dtype={np.dtype(dtype).name!r} has no effect with backend='cpp'; the C++ binary "
                "uses fixed internal precision regardless. Use backend='python' to control "
                "computational precision.",
                stacklevel=2,
            )

        if not use_kspace:
            warnings.warn(
                "use_kspace=False has no effect with backend='cpp'; the C++ binary always applies k-space correction.",
                stacklevel=2,
            )
        if sensor is not None and getattr(sensor, "record", None) is not None:
            warnings.warn(
                "sensor.record is not yet supported for backend='cpp'; the C++ binary will record its default fields.",
                stacklevel=2,
            )
        if sensor is not None and getattr(sensor, "record_start_index", 1) != 1:
            warnings.warn(
                "sensor.record_start_index is not yet supported for backend='cpp'; the C++ binary records from the first time step.",
                stacklevel=2,
            )

        # Convert Cartesian sensor mask to binary grid (cpp binary requires binary masks)
        if sensor is not None and sensor.mask is not None and _is_cartesian_mask(sensor.mask, kgrid.dim):
            from kwave.utils.conversion import cart2grid

            sensor = copy.copy(sensor)
            sensor.mask, _, _ = cart2grid(kgrid, np.asarray(sensor.mask), order="C")

        cpp_sim = CppSimulation(kgrid, medium, source, sensor, pml_size=pml_size, pml_alpha=pml_alpha, use_sg=use_sg)
        if save_only:
            if data_path is None:
                raise ValueError("data_path must be provided when save_only=True (the HDF5 files must persist for cluster submission).")
            input_file, output_file = cpp_sim.prepare(data_path=data_path)
            result = {"input_file": input_file, "output_file": output_file}
            if not pml_inside:
                result["pml_size"] = pml_size
            return result
        result = cpp_sim.run(device=device, num_threads=num_threads, device_num=device_num, quiet=quiet, debug=debug, data_path=data_path, binary_path=binary_path)

    # --- Post-processing: strip PML from full-grid fields ---

    if not pml_inside:
        # Python solver pre-strips _final fields; C++ binary does not
        suffixes = _FULL_GRID_SUFFIXES_CPP if backend == "cpp" else _FULL_GRID_SUFFIXES
        result = _strip_pml(result, pml_size, kgrid.dim, suffixes=suffixes)

    return result


def reshape_to_grid(data, grid_shape):
    """Reshape flat sensor data to grid shape.

    Convenience helper for full-grid sensor masks where ``n_sensor``
    equals the total number of grid points.

    Args:
        data: sensor array — ``(n_sensor, Nt)`` time-series or
            ``(n_sensor,)`` aggregate.
        grid_shape: tuple of grid dimensions, e.g. ``(Nx, Ny)``.

    Returns:
        For time-series: ``(*grid_shape, Nt)``
        For aggregates:  ``(*grid_shape)``
    """
    data = np.asarray(data)
    if data.ndim == 2:
        n_sensor, Nt = data.shape
        return data.reshape(*grid_shape, Nt)
    elif data.ndim == 1:
        return data.reshape(grid_shape)
    return data
