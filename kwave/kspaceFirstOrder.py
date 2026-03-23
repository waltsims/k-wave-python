import copy
import os
import warnings
from typing import Optional, Union

import numpy as np
from beartype import beartype as typechecker

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
    # Pad short tuples by repeating last value (e.g. (20, 15) in 3-D → (20, 15, 15))
    return t + (t[-1],) * (ndim - len(t))


def _is_binary_mask(mask, ndim):
    """True if mask is a grid-shaped binary array (not Cartesian coordinates)."""
    arr = np.asarray(mask)
    if arr.ndim == 2 and arr.shape[0] == ndim:
        return False  # Cartesian: (ndim, n_points)
    return True


def _expand_for_pml_outside(kgrid, medium, source, sensor, pml_size):
    """Expand grid/medium/source/sensor so PML sits outside the user domain."""
    ndim = kgrid.dim
    new_N = tuple(int(n) + 2 * int(p) for n, p in zip(kgrid.N, pml_size))
    expanded_kgrid = kWaveGrid(new_N, kgrid.spacing)
    expanded_kgrid.setTime(kgrid.Nt, kgrid.dt)

    # Medium: edge-extend non-scalar arrays
    expanded_medium = copy.copy(medium)
    for attr in ("sound_speed", "density", "alpha_coeff", "BonA"):
        val = getattr(expanded_medium, attr, None)
        if val is not None and np.atleast_1d(val).size > 1:
            setattr(expanded_medium, attr, expand_matrix(np.atleast_1d(val), list(pml_size)))

    # Source: zero-pad spatial arrays
    expanded_source = copy.copy(source)
    for attr in ("p0", "p_mask", "u_mask"):
        val = getattr(expanded_source, attr, None)
        if val is not None and np.atleast_1d(val).size > 1:
            setattr(expanded_source, attr, expand_matrix(np.atleast_1d(val), list(pml_size), 0))

    # Sensor: zero-pad binary masks, leave Cartesian unchanged
    expanded_sensor = sensor
    if sensor is not None and getattr(sensor, "mask", None) is not None:
        if _is_binary_mask(sensor.mask, ndim):
            expanded_sensor = copy.copy(sensor)
            expanded_sensor.mask = expand_matrix(np.atleast_1d(sensor.mask), list(pml_size), 0)

    return expanded_kgrid, expanded_medium, expanded_source, expanded_sensor


def _strip_pml(result, pml_size, ndim, expanded_grid_shape=None):
    """Remove PML padding from full-grid fields in the result dict.

    Strips spatial fields (_final, _max, _min, _rms) using interior slices.
    If expanded_grid_shape is given, also strips time-series that were
    recorded over the full expanded grid (sensor=None with pml_inside=False).
    """
    slices = tuple(slice(int(p), -int(p) if int(p) else None) for p in pml_size[:ndim])
    full_grid_suffixes = ("_final", "_max", "_min", "_rms")
    stripped = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray) and val.ndim == ndim and any(key.endswith(s) for s in full_grid_suffixes):
            stripped[key] = val[slices]
        elif expanded_grid_shape is not None and isinstance(val, np.ndarray) and val.ndim == 2:
            expanded_numel = int(np.prod(expanded_grid_shape))
            if val.shape[0] == expanded_numel:
                # Build interior boolean mask on expanded grid, apply to sensor axis
                interior = np.ones(expanded_grid_shape, dtype=bool)
                for ax in range(ndim):
                    sl = [slice(None)] * ndim
                    sl[ax] = slice(None, int(pml_size[ax]))
                    interior[tuple(sl)] = False
                    sl[ax] = slice(-int(pml_size[ax]), None)
                    interior[tuple(sl)] = False
                stripped[key] = val[interior.flatten(order="F")]
            else:
                stripped[key] = val
        else:
            stripped[key] = val
    return stripped


@typechecker
def kspaceFirstOrder(
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    source: Union[kSource, NotATransducer],
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
    save_only: bool = False,
    data_path: Optional[str] = None,
    quiet: bool = False,
    debug: bool = False,
    num_threads: Optional[int] = None,
    device_num: Optional[int] = None,
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

    Returns:
        dict: Recorded sensor data keyed by field name (e.g.
        ``"p"``, ``"p_final"``, ``"ux"``, ``"uy"``).

    Environment variables:
        ``KWAVE_BACKEND``: Override ``backend`` (e.g. ``"python"``).
        ``KWAVE_DEVICE``: Override ``device`` (e.g. ``"cpu"``).
    """
    # Environment overrides — lets CI run GPU/cpp examples on CPU/python
    backend = os.environ.get("KWAVE_BACKEND", backend)
    device = os.environ.get("KWAVE_DEVICE", device)

    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if backend not in ("python", "cpp"):
        raise ValueError(f"Unknown backend: {backend!r}. Use 'python' or 'cpp'.")

    if isinstance(pml_size, str) and pml_size.lower() == "auto":
        pml_size = tuple(int(x) for x in get_optimal_pml_size(kgrid))
    pml_size = _normalize_pml(pml_size, kgrid.dim)
    pml_alpha = _normalize_pml(pml_alpha, kgrid.dim, "pml_alpha")

    from kwave.solvers.validation import validate_simulation

    validate_simulation(kgrid, medium, source, sensor, pml_size=pml_size)

    # --- Shared pre-processing (both backends) ---

    # Expand grid when PML sits outside the user domain
    if pml_inside:
        warnings.warn(
            f"pml_inside=True: the outermost {pml_size} grid points per side will be used for PML absorption. "
            "Sources, sensors, and medium properties near the boundary will be affected. "
            "Set pml_inside=False (default) to expand the grid automatically instead.",
            stacklevel=2,
        )
    sensor_was_none = sensor is None or (hasattr(sensor, "mask") and sensor.mask is None)
    if not pml_inside:
        kgrid, medium, source, sensor = _expand_for_pml_outside(kgrid, medium, source, sensor, pml_size)

    # Smooth initial pressure (after expansion so Blackman window covers PML transition)
    if smooth_p0 and source.p0 is not None and kgrid.dim >= 2:
        from kwave.utils.filters import smooth

        source = copy.copy(source)
        grid_shape = tuple(int(n) for n in kgrid.N)
        source.p0 = smooth(np.asarray(source.p0, dtype=float).reshape(grid_shape, order="F"), restore_max=True)

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
            smooth_p0=False,  # already handled above
            pml_size=pml_size,
            pml_alpha=pml_alpha,
        ).run()

    elif backend == "cpp":
        from kwave.solvers.cpp_simulation import CppSimulation

        if not use_kspace:
            warnings.warn(
                "use_kspace=False has no effect with backend='cpp'; " "the C++ binary always applies k-space correction.",
                stacklevel=2,
            )
        if sensor is not None and getattr(sensor, "record", None) is not None:
            warnings.warn(
                "sensor.record is not yet supported for backend='cpp'; " "the C++ binary will record its default fields.",
                stacklevel=2,
            )
        if sensor is not None and getattr(sensor, "record_start_index", 1) != 1:
            warnings.warn(
                "sensor.record_start_index is not yet supported for backend='cpp'; " "the C++ binary records from the first time step.",
                stacklevel=2,
            )

        # Convert Cartesian sensor mask to binary grid (cpp binary requires binary masks)
        if sensor is not None and sensor.mask is not None:
            mask_arr = np.asarray(sensor.mask)
            if mask_arr.ndim == 2 and mask_arr.shape[0] == kgrid.dim:
                from kwave.utils.conversion import cart2grid

                sensor = copy.copy(sensor)
                sensor.mask, _, _ = cart2grid(kgrid, mask_arr)

        cpp_sim = CppSimulation(kgrid, medium, source, sensor, pml_size=pml_size, pml_alpha=pml_alpha, use_sg=use_sg)
        if save_only:
            if data_path is None:
                raise ValueError("data_path must be provided when save_only=True " "(the HDF5 files must persist for cluster submission).")
            input_file, output_file = cpp_sim.prepare(data_path=data_path)
            return {"input_file": input_file, "output_file": output_file}
        result = cpp_sim.run(device=device, num_threads=num_threads, device_num=device_num, quiet=quiet, debug=debug, data_path=data_path)

    # --- Shared post-processing ---

    # Strip PML from full-grid output fields when PML was outside the user domain
    if not pml_inside and isinstance(result, dict):
        expanded_shape = tuple(int(n) for n in kgrid.N) if sensor_was_none else None
        result = _strip_pml(result, pml_size, kgrid.dim, expanded_grid_shape=expanded_shape)

    return result
