"""
C++ binary solver backends (OMP and CUDA).

This module provides wrappers around the C++ k-Wave binaries for CPU (OpenMP)
and GPU (CUDA) execution. These solvers require external binaries and disk I/O.

Note: The C++ solvers require preprocessing done by kWaveSimulation and
save_to_disk_func before execution. This is currently handled in the
kspaceFirstOrder2D/3D entry points. This wrapper is provided for API
consistency with the native solver.
"""
from typing import Union

from kwave.executor import Executor
from kwave.solvers.base import Backend, Solver
from kwave.utils.dotdictionary import dotdict


class CppSolver(Solver):
    """
    C++ binary k-space solver (OMP or CUDA).

    This solver wraps the external C++ binaries for high-performance
    simulation. It requires:
    - External binaries (automatically downloaded by k-wave-python)
    - Disk I/O (input/output via HDF5 files)
    - Preprocessing via kWaveSimulation

    Note: Due to the architecture of the C++ backend, this solver cannot
    be used standalone. Use kspaceFirstOrder2D/3D with backend="OMP" or
    backend="CUDA" instead, which handle the required preprocessing.
    """

    def __init__(self, backend: Union[str, Backend] = Backend.OMP):
        """
        Initialize the C++ solver.

        Args:
            backend: Backend type - "OMP" for CPU or "CUDA" for GPU
        """
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

    def run(
        self,
        kgrid,
        medium,
        source,
        sensor,
        simulation_options,
        execution_options,
    ) -> dict:
        """
        Run the simulation using the C++ binary.

        Note: This method is not intended for direct use. The C++ backend
        requires preprocessing that must be done by kspaceFirstOrder2D/3D.
        Use those entry points with backend="OMP" or backend="CUDA" instead.

        Raises:
            NotImplementedError: Always raised - use kspaceFirstOrder2D/3D instead
        """
        raise NotImplementedError(
            "CppSolver cannot be used directly. The C++ backend requires "
            "preprocessing handled by kspaceFirstOrder2D/3D. Use those "
            "entry points with backend='OMP' or backend='CUDA' instead."
        )

    def run_from_files(
        self,
        input_filename: str,
        output_filename: str,
        simulation_options,
        execution_options,
        sensor,
    ) -> dotdict:
        """
        Run simulation from pre-saved HDF5 input file.

        This is the actual execution method used internally after
        preprocessing is complete.

        Args:
            input_filename: Path to HDF5 input file
            output_filename: Path for HDF5 output file
            simulation_options: Simulation options
            execution_options: Execution options
            sensor: Sensor object for record options

        Returns:
            dotdict with simulation results
        """
        # Set GPU flag based on backend
        execution_options.is_gpu_simulation = self.is_gpu

        # Create executor and run
        executor = Executor(simulation_options=simulation_options, execution_options=execution_options)
        executor_options = execution_options.as_list(sensor=sensor)
        return executor.run_simulation(input_filename, output_filename, options=executor_options)
