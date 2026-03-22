from kwave.solvers.base import Backend, Solver
from kwave.solvers.cpp import CppSolver
from kwave.solvers.kspace_solver import Simulation, create_simulation, simulate_from_dicts
from kwave.solvers.native import NativeSolver


def get_solver(backend: str | Backend = "native", use_gpu: bool = False) -> Solver:
    """Return a solver instance for the given backend name or Backend enum."""
    if isinstance(backend, str):
        backend = Backend(backend)
    if backend == Backend.NATIVE:
        return NativeSolver(use_gpu=use_gpu)
    if backend in (Backend.OMP, Backend.CUDA):
        return CppSolver(backend=backend)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "get_solver",
    "Backend",
    "Solver",
    "NativeSolver",
    "CppSolver",
    "Simulation",
    "create_simulation",
    "simulate_from_dicts",
]
