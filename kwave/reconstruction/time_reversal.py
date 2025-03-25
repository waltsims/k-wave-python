"""
Time reversal reconstruction for photoacoustic imaging.

This class handles time reversal reconstruction of initial pressure distribution
from sensor data. It supports both 2D and 3D simulations and automatically
applies compensation for half-plane recording.

Example:
    >>> tr = TimeReversal(kgrid, medium, sensor)
    >>> p0_recon = tr(kspaceFirstOrder2D, simulation_options, execution_options)
"""

from typing import Any, Callable, Dict

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.options import SimulationExecutionOptions, SimulationOptions


class TimeReversal:
    """
    Time reversal reconstruction for photoacoustic imaging.

    This class handles time reversal reconstruction of initial pressure distribution
    from sensor data. It supports both 2D and 3D simulations and automatically
    applies compensation for half-plane recording.

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

    def __init__(self, kgrid: kWaveGrid, medium: kWaveMedium, sensor: kSensor, compensation_factor: float = 2.0) -> None:
        """
        Initialize time reversal reconstruction.

        Args:
            kgrid: Computational grid for the simulation
            medium: Medium properties for wave propagation
            sensor: Sensor object containing the sensor mask
            compensation_factor: Factor to compensate for half-plane recording (default: 2.0)

        Raises:
            ValueError: If inputs are invalid for time reversal
        """
        self.kgrid = kgrid
        self.medium = medium
        self.sensor = sensor
        self.compensation_factor = compensation_factor
        self._source = None
        self._new_sensor = None

        # Validate inputs
        if sensor.mask is None:
            raise ValueError("Sensor mask must be set for time reversal. Use sensor.mask = ...")

        # Check for valid time array
        if kgrid.t_array is None:
            raise ValueError("t_array must be explicitly set for time reversal")
        if isinstance(kgrid.t_array, str):
            if kgrid.t_array == "auto":
                raise ValueError("t_array must be explicitly set for time reversal")
            else:
                raise ValueError(f"Invalid t_array value: {kgrid.t_array}")

        # Validate compensation factor
        if compensation_factor <= 0:
            raise ValueError("compensation_factor must be positive")

        # Validate sensor mask has at least one active point
        if not np.any(sensor.mask):
            raise ValueError("Sensor mask must have at least one active point")

        # Validate sensor mask shape matches grid dimensions
        if not np.array_equal(sensor.mask.shape, kgrid.N):
            raise ValueError(f"Sensor mask shape {sensor.mask.shape} does not match grid dimensions {kgrid.N}")

    def __call__(
        self, simulation_function: Callable, simulation_options: SimulationOptions, execution_options: SimulationExecutionOptions
    ) -> np.ndarray:
        """
        Run time reversal reconstruction.

        Args:
            simulation_function: Function to run the simulation (e.g., kspaceFirstOrder2D)
            simulation_options: Options for the simulation
            execution_options: Options for execution

        Returns:
            Reconstructed initial pressure distribution

        Raises:
            ValueError: If simulation_function, simulation_options, or execution_options are None,
                      or if sensor does not have recorded pressure data
        """
        if simulation_function is None:
            raise ValueError("simulation_function must be provided")
        if simulation_options is None:
            raise ValueError("simulation_options must be provided")
        if execution_options is None:
            raise ValueError("execution_options must be provided")

        # Validate sensor has recorded pressure data
        if not hasattr(self.sensor, "recorded_pressure") or self.sensor.recorded_pressure is None:
            raise ValueError("Sensor must have recorded pressure data. Run a forward simulation first.")

        # Create source and sensor for reconstruction
        self._source = kSource()
        self._source.p_mask = self.sensor.mask  # Use sensor mask as source mask
        self._source.p = np.flip(self.sensor.recorded_pressure, axis=1)  # Time-reverse the recorded pressure
        self._source.p_mode = "dirichlet"  # Use dirichlet boundary condition
        _passed_record = self.sensor.record
        if "p_final" not in _passed_record:
            _passed_record.append("p_final")
        self._new_sensor = kSensor(mask=self.sensor.mask, record=_passed_record)

        # Run reconstruction
        result = simulation_function(self.kgrid, self._source, self._new_sensor, self.medium, simulation_options, execution_options)

        # Process result
        if isinstance(result, dict):
            p0_recon = result["p_final"]
        else:
            p0_recon = result

        # Apply compensation factor and positivity condition
        p0_recon = self.compensation_factor * p0_recon
        p0_recon[p0_recon < 0] = 0  # Apply positivity condition

        return p0_recon
