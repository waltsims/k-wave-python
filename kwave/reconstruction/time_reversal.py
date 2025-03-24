"""Time reversal reconstruction for acoustic wave propagation.

This module provides functionality for time reversal reconstruction of acoustic wave fields.
"""

from typing import Callable, Union

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


class TimeReversal:
    """
    Time reversal reconstruction class for photoacoustic imaging.

    This class handles the time reversal reconstruction process using the acoustic wave equation.
    It supports both 2D and 3D reconstructions and includes compensation for half-plane recording.
    """

    def __init__(self, kgrid: kWaveGrid, medium: kWaveMedium, sensor: kSensor, compensation_factor: float = 2.0) -> None:
        """
        Args:
            kgrid: Computational grid for the simulation
            medium: Medium properties for wave propagation
            sensor: Sensor object containing the sensor mask
            compensation_factor: Factor to compensate for half-plane recording (default: 2.0)

        Raises:
            ValueError: If inputs are invalid for time reversal

        Note:
            Future versions may support:
            - GPU acceleration via use_gpu parameter
            - Differentiable operations via differentiable parameter
            - Custom boundary conditions via boundary_condition parameter
            - Elastic wave propagation via elastic parameter
        """
        self.kgrid = kgrid
        self.medium = medium
        self.sensor = sensor
        self.compensation_factor = compensation_factor
        self._source = None
        self._new_sensor = None

        # Validate inputs
        if sensor.mask is None:
            raise ValueError("Sensor mask must be set for time reversal")

        # Check for valid time array
        if kgrid.t_array is None:
            raise ValueError("t_array must be explicitly set for time reversal")
        if isinstance(kgrid.t_array, str):
            if kgrid.t_array == "auto":
                raise ValueError("t_array must be explicitly set for time reversal")
            else:
                raise ValueError(f"Invalid t_array value: {kgrid.t_array}")

    def __call__(
        self, simulation_function: Callable, simulation_options: SimulationOptions, execution_options: SimulationExecutionOptions
    ) -> np.ndarray:
        """
        Run the time reversal reconstruction.

        Args:
            simulation_function: Function to use for simulation (e.g., kspaceFirstOrder2D)
            simulation_options: Options for the simulation
            execution_options: Options for execution

        Returns:
            Reconstructed initial pressure distribution

        Raises:
            ValueError: If simulation_function is not a valid simulation function
            ValueError: If simulation_options is None
            ValueError: If execution_options is None
        """
        # Validate inputs
        if simulation_function is None:
            raise ValueError("simulation_function must be provided")
        if simulation_options is None:
            raise ValueError("simulation_options must be provided")
        if execution_options is None:
            raise ValueError("execution_options must be provided")

        # Create source from sensor mask
        self._source = kSource()
        self._source.p_mask = self.sensor.mask

        # Create new sensor for recording
        self._new_sensor = kSensor()
        self._new_sensor.mask = np.ones(self.kgrid.N, dtype=bool)

        # Run simulation
        output = simulation_function(self.kgrid, self._source, self._new_sensor, self.medium, simulation_options, execution_options)

        # Extract and process result
        p0_recon = output["p_final"]

        # Apply compensation factor
        p0_recon *= self.compensation_factor

        return p0_recon
