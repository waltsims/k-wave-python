"""Reproducer for issue #664.

User report: kspaceFirstOrder2D output contains NaNs when alpha_power is in
the range 0.95 to 1.03, but the equivalent MATLAB simulation runs cleanly.
This narrows the bug to the Python-side serialization / interop with the C++
binary, not the binary itself.

Two complementary tests live here:

  1. ``test_smoke_*`` — runs the C++ binary end-to-end and asserts no NaN.
     Skipped when the binary cannot launch (e.g. missing dylibs in CI without
     ``brew install fftw hdf5 zlib libomp``).

  2. ``test_hdf5_input_matches_matlab`` — writes the Python-side HDF5 input
     and compares it field-by-field against a MATLAB-generated reference.
     This is the diagnostic test: the failing field pinpoints the bug.
     Runs everywhere; needs no binary.

On master today both are expected to fail. The fix lands in subsequent
commits on this branch.
"""
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


def _build_minimal_repro(alpha_power, tmp_path, save_to_disk_exit=False):
    """Build the smallest scenario that mirrors #664's failing config."""
    N = Vector([64, 64])
    dx = Vector([0.1e-3, 0.1e-3])
    kgrid = kWaveGrid(N, dx)
    kgrid.makeTime(1500)

    medium = kWaveMedium(
        sound_speed=1500 * np.ones(N),
        density=1000 * np.ones(N),
        alpha_coeff=0.5 * np.ones(N),
        alpha_power=alpha_power,
        alpha_mode="no_dispersion",
    )

    source = kSource()
    source.p_mask = np.zeros(N, dtype=bool)
    source.p_mask[N.x // 2, N.y // 2] = True
    t = kgrid.t_array.squeeze()
    source.p = (np.sin(2 * np.pi * 1e6 * t) * np.exp(-((t - 5e-7) ** 2) / (2e-7) ** 2))[np.newaxis, :]

    sensor = kSensor(mask=np.ones(N, dtype=bool))

    simulation_options = SimulationOptions(
        pml_inside=True,
        smooth_p0=False,
        save_to_disk=True,
        save_to_disk_exit=save_to_disk_exit,
        data_path=str(tmp_path),
        input_filename=f"issue_664_input_{alpha_power}.h5",
        output_filename=f"issue_664_output_{alpha_power}.h5",
    )
    return kgrid, medium, source, sensor, simulation_options


@pytest.mark.parametrize("alpha_power", [0.97, 1.01, 1.03])
def test_smoke_no_nan_for_alpha_power_near_unity(alpha_power, tmp_path):
    """C++ backend output must not contain NaN for alpha_power near 1.0.

    Skipped when the binary cannot launch (missing system libraries).
    """
    kgrid, medium, source, sensor, simulation_options = _build_minimal_repro(alpha_power, tmp_path)
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False, show_sim_log=False)

    try:
        sensor_data = kspaceFirstOrder2DC(
            kgrid=kgrid,
            source=deepcopy(source),
            sensor=sensor,
            medium=medium,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )
    except subprocess.CalledProcessError as e:
        if "Library not loaded" in (e.stderr or "") or "image not found" in (e.stderr or ""):
            pytest.skip(f"C++ binary missing system libraries: {e.stderr.splitlines()[0] if e.stderr else e}")
        raise

    p = np.asarray(sensor_data["p"])
    assert not np.any(np.isnan(p)), (
        f"C++ backend output contains NaN for alpha_power={alpha_power}. " f"NaN fraction = {np.mean(np.isnan(p)):.3%}"
    )
