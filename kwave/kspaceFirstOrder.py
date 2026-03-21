"""
Unified entry point for k-Wave simulations.

Provides a single function `kspaceFirstOrder()` that dispatches to the
appropriate backend (native Python/CuPy or C++ OMP/CUDA).
"""
from types import SimpleNamespace
from typing import Union

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.utils.pml import get_optimal_pml_size


def _normalize_pml(pml_value, ndim, name="pml_size"):
    """Normalize PML parameter to a per-dimension tuple."""
    if isinstance(pml_value, str):
        raise ValueError(f"{name} should already be resolved before calling _normalize_pml")
    if isinstance(pml_value, (int, float)):
        return tuple([pml_value] * ndim)
    if hasattr(pml_value, "__len__"):
        val = tuple(pml_value)
        if len(val) == 1:
            return val * ndim
        if len(val) == ndim:
            return val
        raise ValueError(f"{name} must be a scalar or {ndim}-element sequence, got {len(val)} elements")
    return tuple([pml_value] * ndim)


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
    backend: str = "native",
    use_gpu: bool = False,
    save_only: bool = False,
    data_path: str = None,
    quiet: bool = False,
    debug: bool = False,
    num_threads: int = None,
    device_num: int = None,
) -> dict:
    """
    Run a k-Wave simulation.

    Args:
        kgrid: k-Wave grid object with time stepping configured
        medium: k-Wave medium object
        source: k-Wave source object
        sensor: k-Wave sensor object (None records everywhere)
        pml_size: PML size in grid points. int, per-dimension tuple, or "auto"
        pml_alpha: PML absorption in Nepers per grid point
        use_sg: Use staggered grid (default True)
        use_kspace: Use k-space correction (default True)
        smooth_p0: Smooth initial pressure distribution (default True)
        backend: "native" (Python/CuPy) or "cpp" (C++ binary)
        use_gpu: Use GPU acceleration
        save_only: Only write HDF5 input file (cpp backend)
        data_path: Directory for HDF5 files (cpp backend)
        quiet: Suppress output
        debug: Show detailed output
        num_threads: Number of threads (cpp backend)
        device_num: GPU device number

    Returns:
        dict with 'p' (sensor data), 'p_final' (final pressure field),
        and optionally 'input_file' (if save_only=True)
    """
    ndim = kgrid.dim

    # Resolve pml_size="auto"
    if isinstance(pml_size, str) and pml_size.lower() == "auto":
        pml_size = tuple(int(x) for x in get_optimal_pml_size(kgrid))
    pml_size = _normalize_pml(pml_size, ndim, "pml_size")
    pml_alpha = _normalize_pml(pml_alpha, ndim, "pml_alpha")

    # Validate inputs
    from kwave.solvers.validation import validate_simulation

    validate_simulation(kgrid, medium, source, sensor, pml_size=pml_size)

    if backend == "native":
        return _run_native(kgrid, medium, source, sensor, pml_size, pml_alpha, use_gpu, quiet, debug)
    elif backend == "cpp":
        return _run_cpp(
            kgrid,
            medium,
            source,
            sensor,
            pml_size=pml_size,
            pml_alpha=pml_alpha,
            use_gpu=use_gpu,
            save_only=save_only,
            data_path=data_path,
            quiet=quiet,
            debug=debug,
            num_threads=num_threads,
            device_num=device_num,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'native' or 'cpp'.")


def _run_native(kgrid, medium, source, sensor, pml_size, pml_alpha, use_gpu, quiet, debug):
    """Run simulation using the native Python/CuPy solver."""
    from kwave.solvers.kwave_adapter import (
        _convert_kgrid,
        _convert_medium,
        _convert_sensor,
        _convert_source,
        run_simulation_native,
    )

    # Create a minimal options-like object for the adapter
    opts = SimpleNamespace(pml_size=list(pml_size), pml_alpha=list(pml_alpha))

    kgrid_ns = _convert_kgrid(kgrid, opts)
    medium_ns = _convert_medium(medium)
    source_ns = _convert_source(source)
    sensor_ns = _convert_sensor(sensor)

    from kwave.solvers.kspace_solver import Simulation

    backend_str = "gpu" if use_gpu else "cpu"
    sim = Simulation(kgrid_ns, medium_ns, source_ns, sensor_ns, backend=backend_str)
    results = sim.run()

    from kwave.utils.dotdictionary import dotdict

    output = dotdict()
    output["p"] = results["sensor_data"]
    output["p_final"] = results["pressure"]
    return output


def _run_cpp(kgrid, medium, source, sensor, *, pml_size, pml_alpha, use_gpu, save_only, data_path, quiet, debug, num_threads, device_num):
    """Run simulation using the C++ binary backend."""
    from kwave.solvers.cpp_simulation import CppSimulation

    cpp_sim = CppSimulation(
        kgrid,
        medium,
        source,
        sensor,
        pml_size=pml_size,
        pml_alpha=pml_alpha,
    )

    if save_only:
        input_file, output_file = cpp_sim.prepare(data_path=data_path)
        return {"input_file": input_file, "output_file": output_file}

    return cpp_sim.run(
        use_gpu=use_gpu,
        num_threads=num_threads,
        device_num=device_num,
        quiet=quiet,
        debug=debug,
    )
