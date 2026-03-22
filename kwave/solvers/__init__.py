"""
k-Wave Python Solvers Module.

This module provides multiple backends for k-space pseudospectral wave propagation:

Backends:
    - native: Pure Python/CuPy solver (no external binaries required)
    - OMP: C++ OpenMP binary (CPU, high performance)
    - CUDA: C++ CUDA binary (GPU, highest performance)

Usage:
    # Get a solver by backend name
    from kwave.solvers import get_solver, Backend

    solver = get_solver("native")  # Python solver
    solver = get_solver("OMP")     # C++ CPU solver
    solver = get_solver("CUDA")    # C++ GPU solver

    # Or use the enum
    solver = get_solver(Backend.NATIVE)

    # For native solver, can run directly:
    result = solver.run(kgrid, medium, source, sensor, sim_opts, exec_opts)

    # For C++ solvers, use kspaceFirstOrder2D/3D with backend parameter:
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
    result = kspaceFirstOrder2D(..., execution_options=SimulationExecutionOptions(backend="OMP"))

Low-level API (for debugging/MATLAB interop):
    from kwave.solvers import Simulation, simulate_from_dicts

    # Step-by-step simulation
    sim = Simulation(kgrid, medium, source, sensor)
    sim.setup()
    sim.step()
    result = sim.run()

    # Or from dictionaries
    result = simulate_from_dicts(kgrid_dict, medium_dict, source_dict, sensor_dict)
"""

from kwave.solvers.base import Backend, Solver
from kwave.solvers.cpp import CppSolver
from kwave.solvers.kspace_solver import (
    Simulation,
    create_simulation,
    simulate_from_dicts,
)
from kwave.solvers.native import NativeSolver


def get_solver(backend: str | Backend = "native", use_gpu: bool = False) -> Solver:
    """
    Factory function to get a solver by backend name.

    Args:
        backend: Backend type - "native", "OMP", "CUDA", or Backend enum
        use_gpu: For native backend, whether to use GPU (CuPy) acceleration

    Returns:
        Solver instance

    Examples:
        >>> solver = get_solver("native")           # CPU Python
        >>> solver = get_solver("native", use_gpu=True)  # GPU Python (CuPy)
        >>> solver = get_solver("OMP")              # C++ CPU
        >>> solver = get_solver("CUDA")             # C++ GPU
        >>> solver = get_solver(Backend.NATIVE)     # Using enum
    """
    if isinstance(backend, str):
        backend = Backend(backend)

    if backend == Backend.NATIVE:
        return NativeSolver(use_gpu=use_gpu)
    elif backend in (Backend.OMP, Backend.CUDA):
        return CppSolver(backend=backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    # Factory and enums
    "get_solver",
    "Backend",
    "Solver",
    # Solver classes
    "NativeSolver",
    "CppSolver",
    # Low-level API
    "Simulation",
    "create_simulation",
    "simulate_from_dicts",
]
