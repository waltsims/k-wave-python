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
    OMP = "OMP"
    CUDA = "CUDA"
    NATIVE = "native"


class Solver(ABC):
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
        ...

    @property
    @abstractmethod
    def backend(self) -> Backend:
        ...

    @property
    def requires_binary(self) -> bool:
        return self.backend in (Backend.OMP, Backend.CUDA)

    @property
    def requires_disk_io(self) -> bool:
        return self.requires_binary
