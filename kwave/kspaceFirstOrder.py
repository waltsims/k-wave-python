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
    if len(t) == 1:
        return t * ndim
    if len(t) == ndim:
        return t
    raise ValueError(f"{name} must be a scalar or {ndim}-element sequence, got {len(t)} elements")


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
        from kwave.solvers.cpp_simulation import CppSimulation

        # Apply p0 smoothing before HDF5 serialization (matches MATLAB legacy path)
        if smooth_p0 and source.p0 is not None:
            p0 = np.asarray(source.p0, dtype=float)
            if p0.ndim >= 2:
                from kwave.utils.filters import smooth

                source = copy.copy(source)
                source.p0 = smooth(p0, restore_max=True)

        cpp_sim = CppSimulation(kgrid, medium, source, sensor, pml_size=pml_size, pml_alpha=pml_alpha, use_sg=use_sg)
        if save_only:
            input_file, output_file = cpp_sim.prepare(data_path=data_path)
            return {"input_file": input_file, "output_file": output_file}
        return cpp_sim.run(device=device, num_threads=num_threads, device_num=device_num, quiet=quiet, debug=debug)

    raise ValueError(f"Unknown backend: {backend!r}. Use 'python' or 'cpp'.")
