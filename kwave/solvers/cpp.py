from typing import Union

from kwave.executor import Executor
from kwave.solvers.base import Backend, Solver
from kwave.utils.dotdictionary import dotdict


class CppSolver(Solver):
    """C++ binary k-space solver (OMP or CUDA).

    Note: run() cannot be used directly — preprocessing is handled by kspaceFirstOrder2D/3D.
    Use run_from_files() after preprocessing is complete.
    """

    def __init__(self, backend: Union[str, Backend] = Backend.OMP):
        if isinstance(backend, str):
            backend = Backend(backend)
        if backend not in (Backend.OMP, Backend.CUDA):
            raise ValueError(f"CppSolver only supports OMP or CUDA backends, got {backend}")
        self._backend = backend

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def is_gpu(self) -> bool:
        return self._backend == Backend.CUDA

    def run(self, kgrid, medium, source, sensor, simulation_options, execution_options) -> dict:
        raise NotImplementedError("Use kspaceFirstOrder2D/3D with backend='OMP' or 'CUDA'.")

    def run_from_files(self, input_filename, output_filename, simulation_options, execution_options, sensor) -> dotdict:
        execution_options.is_gpu_simulation = self.is_gpu
        executor = Executor(simulation_options=simulation_options, execution_options=execution_options)
        return executor.run_simulation(input_filename, output_filename, options=execution_options.as_list(sensor=sensor))
