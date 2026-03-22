"""
Native Python/CuPy solver backend.

This module provides a pure Python implementation of the k-space pseudospectral
method with optional GPU acceleration via CuPy.
"""
from typing import Union

from kwave.solvers.base import Backend, Solver
from kwave.solvers.kspace_solver import Simulation
from kwave.solvers.kwave_adapter import (
    _convert_kgrid,
    _convert_medium,
    _convert_sensor,
    _convert_source,
)


class NativeSolver(Solver):
    """
    Pure Python/CuPy k-space pseudospectral solver.

    This solver runs entirely in Python using NumPy for CPU computation
    or CuPy for GPU acceleration. No external binaries are required.

    Attributes:
        use_gpu: Whether to use GPU acceleration (requires CuPy)
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the native solver.

        Args:
            use_gpu: Whether to use GPU acceleration via CuPy
        """
        self._use_gpu = use_gpu

    @property
    def backend(self) -> Backend:
        return Backend.NATIVE

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

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
        Run the simulation using the native Python/CuPy backend.

        Args:
            kgrid: k-Wave grid object
            medium: k-Wave medium object
            source: k-Wave source object
            sensor: k-Wave sensor object
            simulation_options: Simulation options
            execution_options: Execution options

        Returns:
            Dictionary with 'p' (sensor data) and 'p_final' (final pressure field)
        """
        # Convert kwave classes to SimpleNamespace for Simulation
        kgrid_ns = _convert_kgrid(kgrid, simulation_options)
        medium_ns = _convert_medium(medium)
        source_ns = _convert_source(source)
        sensor_ns = _convert_sensor(sensor)

        # Select backend
        backend_str = "gpu" if self._use_gpu else "cpu"

        # Create and run simulation
        sim = Simulation(kgrid_ns, medium_ns, source_ns, sensor_ns, backend=backend_str)
        results = sim.run()

        return results
