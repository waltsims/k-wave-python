from kwave.solvers.base import Backend, Solver
from kwave.solvers.kwave_adapter import run_simulation_native


class NativeSolver(Solver):
    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def backend(self) -> Backend:
        return Backend.NATIVE

    @property
    def device(self) -> str:
        return self._device

    def run(self, kgrid, medium, source, sensor, simulation_options, execution_options) -> dict:
        return run_simulation_native(kgrid, medium, source, sensor, simulation_options, device=self._device)
