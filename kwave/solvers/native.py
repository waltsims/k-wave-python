from kwave.solvers.base import Backend, Solver
from kwave.solvers.kwave_adapter import run_simulation_native


class PythonSolver(Solver):
    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def backend(self) -> Backend:
        return Backend.PYTHON

    @property
    def device(self) -> str:
        return self._device

    def run(self, kgrid, medium, source, sensor, simulation_options, execution_options) -> dict:
        return run_simulation_native(kgrid, medium, source, sensor, simulation_options, device=self._device)


def run_python_backend(kgrid, medium, source, sensor, simulation_options, execution_options):
    """Dispatch to PythonSolver from legacy kspaceFirstOrder2D/3D."""
    device = "gpu" if execution_options.is_gpu_simulation else "cpu"
    return PythonSolver(device=device).run(kgrid, medium, source, sensor, simulation_options, execution_options)
