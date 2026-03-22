from kwave.executor import Executor
from kwave.solvers.base import Backend, Solver
from kwave.utils.dotdictionary import dotdict


class CppSolver(Solver):
    """C++ binary k-space solver. Use kspaceFirstOrder() or run_from_files()."""

    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def backend(self) -> Backend:
        return Backend.CPP

    @property
    def device(self) -> str:
        return self._device

    def run(self, kgrid, medium, source, sensor, simulation_options, execution_options) -> dict:
        raise NotImplementedError("Use kspaceFirstOrder(backend='cpp') instead.")

    def run_from_files(self, input_filename, output_filename, simulation_options, execution_options, sensor) -> dotdict:
        execution_options.is_gpu_simulation = self._device == "gpu"
        executor = Executor(simulation_options=simulation_options, execution_options=execution_options)
        return executor.run_simulation(input_filename, output_filename, options=execution_options.as_list(sensor=sensor))
