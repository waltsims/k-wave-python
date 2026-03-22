"""
Adapter to convert k-Wave classes to Simulation inputs.

This module bridges the gap between kwave's dataclass-based API
(kWaveGrid, kWaveMedium, kSource, kSensor) and the Simulation class.
"""
from types import SimpleNamespace
from typing import Union

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.options.simulation_options import SimulationOptions
from kwave.solvers.kspace_solver import Simulation


def _convert_kgrid(kgrid: kWaveGrid, simulation_options: SimulationOptions) -> SimpleNamespace:
    """Convert kWaveGrid to SimpleNamespace for Simulation."""
    ns = SimpleNamespace()

    # Grid dimensions
    ns.Nx = int(kgrid.N[0])
    ns.dx = float(kgrid.spacing[0])

    if kgrid.dim >= 2:
        ns.Ny = int(kgrid.N[1])
        ns.dy = float(kgrid.spacing[1])

    if kgrid.dim >= 3:
        ns.Nz = int(kgrid.N[2])
        ns.dz = float(kgrid.spacing[2])

    # Time stepping
    ns.Nt = int(kgrid.Nt)
    ns.dt = float(kgrid.dt)

    # PML parameters from simulation options
    pml_size = simulation_options.pml_size
    pml_alpha = simulation_options.pml_alpha

    # Handle PML size (can be scalar or per-dimension)
    if hasattr(pml_size, "__len__"):
        ns.pml_size_x = int(pml_size[0])
        if len(pml_size) >= 2:
            ns.pml_size_y = int(pml_size[1])
        if len(pml_size) >= 3:
            ns.pml_size_z = int(pml_size[2])
    else:
        ns.pml_size_x = int(pml_size)
        if kgrid.dim >= 2:
            ns.pml_size_y = int(pml_size)
        if kgrid.dim >= 3:
            ns.pml_size_z = int(pml_size)

    # Handle PML alpha (can be scalar or per-dimension)
    if hasattr(pml_alpha, "__len__"):
        ns.pml_alpha_x = float(pml_alpha[0])
        if len(pml_alpha) >= 2:
            ns.pml_alpha_y = float(pml_alpha[1])
        if len(pml_alpha) >= 3:
            ns.pml_alpha_z = float(pml_alpha[2])
    else:
        ns.pml_alpha_x = float(pml_alpha)
        if kgrid.dim >= 2:
            ns.pml_alpha_y = float(pml_alpha)
        if kgrid.dim >= 3:
            ns.pml_alpha_z = float(pml_alpha)

    return ns


def _convert_medium(medium: kWaveMedium) -> SimpleNamespace:
    """Convert kWaveMedium to SimpleNamespace for Simulation."""
    ns = SimpleNamespace()

    ns.sound_speed = np.asarray(medium.sound_speed)
    ns.density = np.asarray(medium.density) if medium.density is not None else 1000.0

    if medium.alpha_coeff is not None:
        ns.alpha_coeff = np.asarray(medium.alpha_coeff)
    if medium.alpha_power is not None:
        ns.alpha_power = float(np.asarray(medium.alpha_power).flat[0])

    if medium.BonA is not None:
        ns.BonA = np.asarray(medium.BonA)

    return ns


def _convert_source(source: kSource) -> SimpleNamespace:
    """Convert kSource to SimpleNamespace for Simulation."""
    ns = SimpleNamespace()

    # Initial pressure
    if source.p0 is not None:
        ns.p0 = np.asarray(source.p0)

    # Time-varying pressure source
    if source.p is not None:
        ns.p = np.asarray(source.p)
    if source.p_mask is not None:
        ns.p_mask = np.asarray(source.p_mask)
    if source.p_mode is not None:
        ns.p_mode = source.p_mode

    # Time-varying velocity sources
    if source.ux is not None:
        ns.ux = np.asarray(source.ux)
    if source.uy is not None:
        ns.uy = np.asarray(source.uy)
    if source.uz is not None:
        ns.uz = np.asarray(source.uz)
    if source.u_mask is not None:
        ns.u_mask = np.asarray(source.u_mask)
    if source.u_mode is not None:
        ns.u_mode = source.u_mode

    return ns


def _convert_sensor(sensor: Union[kSensor, NotATransducer, None]) -> SimpleNamespace:
    """Convert kSensor to SimpleNamespace for Simulation."""
    ns = SimpleNamespace()

    if sensor is None:
        # Record everywhere
        ns.mask = None
    elif hasattr(sensor, "mask") and sensor.mask is not None:
        ns.mask = np.asarray(sensor.mask)

    if hasattr(sensor, "record") and sensor.record is not None:
        ns.record = tuple(sensor.record)

    if hasattr(sensor, "record_start_index") and sensor.record_start_index is not None:
        ns.record_start_index = int(sensor.record_start_index)

    return ns


def run_simulation_native(
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    source: kSource,
    sensor: Union[kSensor, NotATransducer, None],
    simulation_options: SimulationOptions,
    use_gpu: bool = False,
) -> dict:
    """
    Run simulation using the native Python/CuPy solver.

    Args:
        kgrid: k-Wave grid object
        medium: k-Wave medium object
        source: k-Wave source object
        sensor: k-Wave sensor object
        simulation_options: Simulation options
        use_gpu: Whether to use GPU acceleration (requires CuPy)

    Returns:
        Dictionary with sensor data fields (p, p_final, etc.)
    """
    # Convert kwave classes to SimpleNamespace
    kgrid_ns = _convert_kgrid(kgrid, simulation_options)
    medium_ns = _convert_medium(medium)
    source_ns = _convert_source(source)
    sensor_ns = _convert_sensor(sensor)

    # Select backend
    backend = "gpu" if use_gpu else "cpu"

    # Create and run simulation
    sim = Simulation(kgrid_ns, medium_ns, source_ns, sensor_ns, backend=backend)
    return sim.run()
