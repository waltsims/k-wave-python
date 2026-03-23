"""Legacy bridge for kspaceFirstOrder2D/3D → Simulation."""

from kwave.solvers.kspace_solver import Simulation


def run_python_backend(kgrid, medium, source, sensor, simulation_options, execution_options):
    """Dispatch to Simulation from legacy kspaceFirstOrder2D/3D."""
    device = "gpu" if execution_options.is_gpu_simulation else "cpu"
    pml_size = simulation_options.pml_size
    if pml_size is None:
        pml_size = (20,) * kgrid.dim
    elif not hasattr(pml_size, "__len__"):
        pml_size = (int(pml_size),) * kgrid.dim
    pml_alpha = getattr(simulation_options, "pml_alpha", None)
    if pml_alpha is None:
        pml_alpha = (2.0,) * kgrid.dim
    elif not hasattr(pml_alpha, "__len__"):
        pml_alpha = (float(pml_alpha),) * kgrid.dim
    return Simulation(
        kgrid,
        medium,
        source,
        sensor,
        device=device,
        use_sg=getattr(simulation_options, "use_sg", True),
        use_kspace=getattr(simulation_options, "use_kspace", True),
        smooth_p0=getattr(simulation_options, "smooth_p0", True),
        pml_size=tuple(pml_size),
        pml_alpha=tuple(pml_alpha),
    ).run()
