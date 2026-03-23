from enum import Enum

from kwave.solvers.kspace_solver import Simulation, create_simulation, simulate_from_dicts


class Backend(Enum):
    PYTHON = "python"
    CPP = "cpp"


__all__ = [
    "Backend",
    "Simulation",
    "create_simulation",
    "simulate_from_dicts",
]
