"""Legacy bridge for kspaceFirstOrder2D/3D → Simulation."""

import copy

import numpy as np

from kwave.solvers.kspace_solver import Simulation


def _per_axis(value, ndim, cast):
    """Expand a legacy scalar / 1-element / per-axis option to one value per axis.

    Mirrors ``SimulationOptions.option_factory``: a full per-axis sequence is used as given,
    anything shorter applies its first element to every axis.
    """
    values = [cast(v) for v in np.atleast_1d(value).ravel()]
    return tuple(values) if len(values) == ndim else (values[0],) * ndim


def run_python_backend(kgrid, medium, source, sensor, simulation_options, execution_options):
    """Dispatch to Simulation from legacy kspaceFirstOrder2D/3D."""
    # Imported here: kwave.kspaceFirstOrder imports the solver package lazily, keep it that way round
    from kwave.kspaceFirstOrder import _FULL_GRID_SUFFIXES, _expand_for_pml_outside, _strip_pml

    device = "gpu" if execution_options.is_gpu_simulation else "cpu"
    pml_size = simulation_options.pml_size
    pml_size = (20,) * kgrid.dim if pml_size is None else _per_axis(pml_size, kgrid.dim, int)
    pml_alpha = getattr(simulation_options, "pml_alpha", None)
    pml_alpha = (2.0,) * kgrid.dim if pml_alpha is None else _per_axis(pml_alpha, kgrid.dim, float)
    # pml_inside=True: the PML occupies the user's grid and *_final fields cover the whole grid.
    # pml_inside=False: pad the grid so the PML sits outside it, exactly as kspaceFirstOrder() does;
    # the solver then crops *_final fields back to the user's grid.
    pml_inside = getattr(simulation_options, "pml_inside", False)
    ndim = kgrid.dim
    smooth_p0 = getattr(simulation_options, "smooth_p0", True)
    if not pml_inside:
        if smooth_p0 and getattr(source, "p0", None) is not None and np.size(source.p0) > 1:
            # Legacy k-Wave smooths p0 on the user's grid *before* padding it (see kWaveSimulation.smooth_and_enlarge)
            from kwave.utils.filters import smooth

            source = copy.copy(source)
            source.p0 = smooth(np.asarray(source.p0, dtype=float), restore_max=True)
            smooth_p0 = False
        kgrid, medium, source, sensor = _expand_for_pml_outside(kgrid, medium, source, sensor, pml_size)
    result = Simulation(
        kgrid,
        medium,
        source,
        sensor,
        device=device,
        use_sg=getattr(simulation_options, "use_sg", True),
        use_kspace=getattr(simulation_options, "use_kspace", True),
        smooth_p0=smooth_p0,
        pml_size=pml_size,
        pml_alpha=pml_alpha,
        pml_inside=pml_inside,
    ).run()
    if not pml_inside:
        result = _strip_pml(result, pml_size, ndim, suffixes=_FULL_GRID_SUFFIXES)
    return result
