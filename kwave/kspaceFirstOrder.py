import copy
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.utils.pml import get_optimal_pml_size


def _normalize_pml(val, ndim, name="pml_size"):
    if isinstance(val, str):
        raise ValueError(f"{name} should already be resolved before calling _normalize_pml")
    if isinstance(val, (int, float)):
        return (val,) * ndim
    t = tuple(val)
    if len(t) == 0 or len(t) > ndim:
        raise ValueError(f"{name} must be a scalar or 1-to-{ndim}-element sequence, got {len(t)} elements")
    # Pad short tuples by repeating last value (e.g. (20, 15) in 3-D → (20, 15, 15))
    return t + (t[-1],) * (ndim - len(t))


def kspaceFirstOrder(
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    source: kSource,
    sensor: Union[kSensor, NotATransducer, None] = None,
    *,
    pml_size: Union[int, tuple, str] = 20,
    pml_alpha: Union[float, tuple] = 2.0,
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
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

    if isinstance(pml_size, str) and pml_size.lower() == "auto":
        pml_size = tuple(int(x) for x in get_optimal_pml_size(kgrid))
    pml_size = _normalize_pml(pml_size, kgrid.dim)
    pml_alpha = _normalize_pml(pml_alpha, kgrid.dim, "pml_alpha")

    from kwave.solvers.validation import validate_simulation

    validate_simulation(kgrid, medium, source, sensor, pml_size=pml_size)

    if backend == "python":
        from kwave.solvers.kwave_adapter import run_simulation_native

        opts = SimpleNamespace(
            pml_size=list(pml_size),
            pml_alpha=list(pml_alpha),
            use_sg=use_sg,
            use_kspace=use_kspace,
            smooth_p0=smooth_p0,
        )
        return run_simulation_native(kgrid, medium, source, sensor, opts, device=device)

    if backend == "cpp":
        import warnings

        from kwave.solvers.cpp_simulation import CppSimulation

        if not use_kspace:
            warnings.warn(
                "use_kspace=False has no effect with backend='cpp'; " "the C++ binary always applies k-space correction.",
                stacklevel=2,
            )

        # Apply p0 smoothing before HDF5 serialization (matches MATLAB legacy path)
        if smooth_p0 and source.p0 is not None and kgrid.dim >= 2:
            from kwave.utils.filters import smooth

            source = copy.copy(source)
            grid_shape = tuple(int(n) for n in kgrid.N)
            source.p0 = smooth(np.asarray(source.p0, dtype=float).reshape(grid_shape, order="F"), restore_max=True)

        cpp_sim = CppSimulation(kgrid, medium, source, sensor, pml_size=pml_size, pml_alpha=pml_alpha, use_sg=use_sg)
        if save_only:
            input_file, output_file = cpp_sim.prepare(data_path=data_path)
            return {"input_file": input_file, "output_file": output_file}
        return cpp_sim.run(device=device, num_threads=num_threads, device_num=device_num, quiet=quiet, debug=debug, data_path=data_path)

    raise ValueError(f"Unknown backend: {backend!r}. Use 'python' or 'cpp'.")
