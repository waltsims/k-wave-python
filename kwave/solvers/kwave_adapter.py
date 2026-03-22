"""Adapter: convert kwave classes → Simulation inputs."""
from types import SimpleNamespace
from typing import Union

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.solvers.kspace_solver import Simulation


def _convert_kgrid(kgrid: kWaveGrid, opts) -> SimpleNamespace:
    ns = SimpleNamespace(Nx=int(kgrid.N[0]), dx=float(kgrid.spacing[0]), Nt=int(kgrid.Nt), dt=float(kgrid.dt))
    if kgrid.dim >= 2:
        ns.Ny, ns.dy = int(kgrid.N[1]), float(kgrid.spacing[1])
    if kgrid.dim >= 3:
        ns.Nz, ns.dz = int(kgrid.N[2]), float(kgrid.spacing[2])

    pml_size = opts.pml_size if opts.pml_size is not None else [20] * kgrid.dim
    pml_alpha = opts.pml_alpha if opts.pml_alpha is not None else [2.0] * kgrid.dim
    for attr, src, cast in [("pml_size", pml_size, int), ("pml_alpha", pml_alpha, float)]:
        dims = ["x", "y", "z"][: kgrid.dim]
        if hasattr(src, "__len__"):
            for i, ax in enumerate(dims):
                if i < len(src):
                    setattr(ns, f"{attr}_{ax}", cast(src[i]))
        else:
            for ax in dims:
                setattr(ns, f"{attr}_{ax}", cast(src))
    return ns


def _convert_medium(medium: kWaveMedium) -> SimpleNamespace:
    ns = SimpleNamespace(
        sound_speed=np.asarray(medium.sound_speed),
        density=np.asarray(medium.density) if medium.density is not None else 1000.0,
    )
    if medium.alpha_coeff is not None:
        ns.alpha_coeff = np.asarray(medium.alpha_coeff)
    if medium.alpha_power is not None:
        ns.alpha_power = float(np.asarray(medium.alpha_power).flat[0])
    if medium.BonA is not None:
        ns.BonA = np.asarray(medium.BonA)
    return ns


def _convert_source(source: kSource) -> SimpleNamespace:
    ns = SimpleNamespace()
    for attr in ("p0", "p", "p_mask", "p_mode", "ux", "uy", "uz", "u_mask", "u_mode"):
        val = getattr(source, attr, None)
        if val is not None:
            setattr(ns, attr, np.asarray(val) if attr not in ("p_mode", "u_mode") else val)
    return ns


def _convert_sensor(sensor: Union[kSensor, NotATransducer, None]) -> SimpleNamespace:
    ns = SimpleNamespace(mask=None)
    if sensor is None:
        return ns
    if hasattr(sensor, "mask") and sensor.mask is not None:
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
    simulation_options,
    device: str = "cpu",
) -> dict:
    return Simulation(
        _convert_kgrid(kgrid, simulation_options),
        _convert_medium(medium),
        _convert_source(source),
        _convert_sensor(sensor),
        backend=device,
        use_sg=getattr(simulation_options, "use_sg", True),
        use_kspace=getattr(simulation_options, "use_kspace", True),
        smooth_p0=getattr(simulation_options, "smooth_p0", True),
    ).run()
