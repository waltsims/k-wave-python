from kwave.solvers.base import Backend, Solver
from kwave.solvers.cpp import CppSolver
from kwave.solvers.kspace_solver import Simulation, create_simulation, simulate_from_dicts
from kwave.solvers.native import PythonSolver


def get_solver(backend: str | Backend = "python", device: str = "cpu") -> Solver:
    if isinstance(backend, str):
        backend = Backend(backend)
    if backend == Backend.PYTHON:
        return PythonSolver(device=device)
    if backend == Backend.CPP:
        return CppSolver(device=device)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "get_solver",
    "Backend",
    "Solver",
    "PythonSolver",
    "CppSolver",
    "Simulation",
    "create_simulation",
    "simulate_from_dicts",
]
