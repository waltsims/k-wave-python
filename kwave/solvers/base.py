"""
Base solver interface for k-Wave backends.

All solver backends (native Python, C++ OMP, C++ CUDA) implement this interface.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.ktransducer import NotATransducer
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions


class Backend(Enum):
    """Available simulation backends."""

    OMP = "OMP"  # C++ OpenMP binary
    CUDA = "CUDA"  # C++ CUDA binary
    NATIVE = "native"  # Pure Python/CuPy

    @classmethod
    def from_string(cls, value: str) -> "Backend":
        """Convert string to Backend enum."""
        if value is None:
            return None
        return cls(value)


class Solver(ABC):
    """Abstract base class for k-Wave solvers."""

    @abstractmethod
    def run(
        self,
        kgrid: "kWaveGrid",
        medium: "kWaveMedium",
        source: "kSource",
        sensor: Union["kSensor", "NotATransducer", None],
        simulation_options: "SimulationOptions",
        execution_options: "SimulationExecutionOptions",
    ) -> dict:
        """
        Run the simulation.

        Args:
            kgrid: k-Wave grid object
            medium: k-Wave medium object
            source: k-Wave source object
            sensor: k-Wave sensor object
            simulation_options: Simulation options
            execution_options: Execution options

        Returns:
            Dictionary with simulation results (sensor_data, pressure fields, etc.)
        """
        pass

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """Return the backend type."""
        pass

    @property
    def requires_binary(self) -> bool:
        """Whether this solver requires external binaries."""
        return self.backend in (Backend.OMP, Backend.CUDA)

    @property
    def requires_disk_io(self) -> bool:
        """Whether this solver requires saving to disk."""
        return self.requires_binary
