from kwave.solvers.base import Backend, Solver
from kwave.solvers.kwave_adapter import run_simulation_native


class NativeSolver(Solver):
    def __init__(self, use_gpu: bool = False):
        self._use_gpu = use_gpu

    @property
    def backend(self) -> Backend:
        return Backend.NATIVE

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    def run(self, kgrid, medium, source, sensor, simulation_options, execution_options) -> dict:
        return run_simulation_native(kgrid, medium, source, sensor, simulation_options, use_gpu=self._use_gpu)
