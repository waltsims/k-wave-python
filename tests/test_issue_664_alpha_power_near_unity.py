"""Reproducer + regression test for issue #664.

User report: ``kspaceFirstOrder2D`` output contains NaNs when ``alpha_power`` is
in the range 0.95 to 1.03, but the equivalent MATLAB simulation runs cleanly.

Diagnosis (see ``plans/issue-664.md``):

  - ``medium.alpha_mode='no_dispersion'`` is the documented escape hatch for the
    ``tan(pi * alpha_power / 2)`` singularity at ``alpha_power == 1``.
  - The C++ binary's HDF5 input format does NOT carry ``alpha_mode``.  The
    legacy ``kspaceFirstOrder2D`` / ``kspaceFirstOrder3D`` Python entries write
    only ``alpha_coeff`` and ``alpha_power``, so the binary always applies the
    full power-law absorption + dispersion math.  Near ``alpha_power = 1`` that
    math overflows in ``float32`` and the binary returns NaN.
  - The MATLAB k-Wave equivalent compares clean only because the user was
    running MATLAB's pure-MATLAB ``kspaceFirstOrder2D`` (which honors
    ``alpha_mode``); the MATLAB ``kspaceFirstOrder2DC`` C++ path has the same
    HDF5 limitation.
  - The new Python solver (``kwave/solvers/kspace_solver.py``) historically
    ignored ``alpha_mode`` and would diverge for the same reason.

Fix in this branch:

  1. ``kspace_solver.py`` now honors ``alpha_mode='no_dispersion'`` and
     ``'no_absorption'`` — matching the legacy MATLAB and Python paths.
  2. ``kspaceFirstOrder()`` (modern API) raises ``ValueError`` when
     ``backend='cpp'`` is combined with one of those alpha_modes, pointing the
     user at ``backend='python'``.
  3. ``kspaceFirstOrder2D`` / ``kspaceFirstOrder3D`` (legacy API) raise the
     same error before invoking the C++ binary.
"""

import subprocess
from copy import deepcopy

import numpy as np
import pytest
from scipy.signal import gausspulse

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


def _build_repro(alpha_power, *, alpha_mode="no_dispersion"):
    """Heterogeneous-medium repro that mirrors issue #664's failing config.

    The user's full scenario is a 1600x1600 Shepp-Logan phantom with a 512-element
    ring transducer; we shrink to 64x64 and a single-point dirichlet source while
    keeping the features the bug requires: heterogeneous sound speed, heterogeneous
    absorption, ``pml_inside=False``, and a high ``pml_alpha``.
    """
    N = Vector([64, 64])
    dx = Vector([0.15e-3, 0.15e-3])
    c = 1420 + 220 * np.random.default_rng(0).random(tuple(N))
    atten = 0.0025 + 6e-2 * np.abs(c - c[0, 0])

    kgrid = kWaveGrid(N, dx)
    kgrid.makeTime(c)

    medium = kWaveMedium(
        sound_speed=c,
        density=1000 * np.ones(tuple(N)),
        alpha_coeff=atten,
        alpha_power=alpha_power,
        alpha_mode=alpha_mode,
    )

    source = kSource()
    source.p_mask = np.zeros(tuple(N), dtype=bool)
    source.p_mask[N.x // 4, N.y // 4] = True
    t = kgrid.t_array.squeeze()
    source.p = gausspulse(t - 5e-7, fc=1e6, bw=0.75)[np.newaxis, :]
    source.p_mode = "dirichlet"

    sensor = kSensor(mask=np.ones(tuple(N), dtype=bool))
    return kgrid, medium, source, sensor


@pytest.mark.parametrize("alpha_power", [0.97, 0.99, 1.005, 1.03])
def test_python_backend_honors_no_dispersion(alpha_power):
    """The Python solver must apply ``alpha_mode='no_dispersion'`` and stay finite."""
    kgrid, medium, source, sensor = _build_repro(alpha_power)
    result = kspaceFirstOrder(
        kgrid=kgrid,
        medium=medium,
        source=deepcopy(source),
        sensor=sensor,
        backend="python",
        device="cpu",
        pml_inside=False,
        smooth_p0=False,
        pml_alpha=10,
        quiet=True,
    )
    p = np.asarray(result["p"])
    assert not np.any(np.isnan(p)), f"Python backend NaN'd at alpha_power={alpha_power}"
    assert np.nanmax(np.abs(p)) < 1e3, (
        f"Python backend diverged at alpha_power={alpha_power}: "
        f"max|p| = {np.nanmax(np.abs(p)):.3g} — alpha_mode='no_dispersion' was likely ignored."
    )


@pytest.mark.parametrize("mode", ["no_dispersion", "no_absorption", "stokes"])
def test_modern_api_rejects_alpha_mode_on_cpp_backend(mode, tmp_path):
    """``kspaceFirstOrder(..., backend='cpp')`` must refuse alpha_mode it can't honor."""
    kgrid, medium, source, sensor = _build_repro(0.99, alpha_mode=mode)
    with pytest.raises(ValueError, match="alpha_mode"):
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="cpp",
            device="cpu",
            pml_inside=False,
            smooth_p0=False,
            pml_alpha=10,
            data_path=str(tmp_path),
            quiet=True,
        )


@pytest.mark.parametrize("mode", ["no_dispersion", "no_absorption", "stokes"])
def test_legacy_api_rejects_alpha_mode_on_cpp_backend(mode, tmp_path):
    """Legacy ``kspaceFirstOrder2DC`` must refuse alpha_mode it can't honor."""
    kgrid, medium, source, sensor = _build_repro(0.99, alpha_mode=mode)
    so = SimulationOptions(
        pml_inside=False,
        smooth_p0=False,
        save_to_disk=True,
        pml_alpha=10,
        data_cast="single",
        data_path=str(tmp_path),
        input_filename=f"issue_664_input_{mode}.h5",
        output_filename=f"issue_664_output_{mode}.h5",
    )
    eo = SimulationExecutionOptions(is_gpu_simulation=False, show_sim_log=False)
    with pytest.raises(ValueError, match="alpha_mode"):
        kspaceFirstOrder2DC(
            kgrid=kgrid,
            source=deepcopy(source),
            sensor=sensor,
            medium=medium,
            simulation_options=so,
            execution_options=eo,
        )


@pytest.mark.parametrize("alpha_power", [0.97, 1.03])
def test_modern_api_warns_when_alpha_mode_unset_near_unity(alpha_power, tmp_path):
    """``kspaceFirstOrder(..., backend='cpp')`` must warn when alpha_power is near 1
    even with the default ``alpha_mode=None`` — the silent-NaN path the original
    issue reporter hit before they discovered the ``no_dispersion`` escape hatch.
    """
    N = Vector([64, 64])
    dx = Vector([0.1e-3, 0.1e-3])
    kgrid = kWaveGrid(N, dx)
    kgrid.makeTime(1500)
    medium = kWaveMedium(
        sound_speed=1500 * np.ones(tuple(N)),
        density=1000 * np.ones(tuple(N)),
        alpha_coeff=0.5 * np.ones(tuple(N)),
        alpha_power=alpha_power,
        # alpha_mode left at default (None)
    )
    assert medium.alpha_mode is None
    source = kSource()
    source.p_mask = np.zeros(tuple(N), dtype=bool)
    source.p_mask[N.x // 2, N.y // 2] = True
    t = kgrid.t_array.squeeze()
    source.p = (np.sin(2 * np.pi * 1e6 * t) * np.exp(-((t - 5e-7) ** 2) / (2e-7) ** 2))[np.newaxis, :]
    sensor = kSensor(mask=np.ones(tuple(N), dtype=bool))

    with pytest.warns(UserWarning, match="dispersion-singular"):
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="cpp",
            device="cpu",
            pml_inside=True,
            smooth_p0=False,
            save_only=True,
            data_path=str(tmp_path),
            quiet=True,
        )


@pytest.mark.parametrize("alpha_power", [0.96, 0.99, 1.005, 1.04])
def test_warn_helper_fires_in_singular_range(alpha_power):
    """``warn_alpha_power_near_unity_cpp`` must warn for ``alpha_power`` in [0.95, 1.05]."""
    from kwave.utils.checks import warn_alpha_power_near_unity_cpp

    medium = kWaveMedium(sound_speed=1500.0, density=1000.0, alpha_coeff=0.5, alpha_power=alpha_power)
    with pytest.warns(UserWarning, match="dispersion-singular"):
        warn_alpha_power_near_unity_cpp(medium)


@pytest.mark.parametrize("alpha_power", [0.5, 0.94, 1.06, 1.5, 2.0])
def test_warn_helper_silent_outside_singular_range(alpha_power, recwarn):
    """``warn_alpha_power_near_unity_cpp`` must stay silent outside [0.95, 1.05]."""
    from kwave.utils.checks import warn_alpha_power_near_unity_cpp

    medium = kWaveMedium(sound_speed=1500.0, density=1000.0, alpha_coeff=0.5, alpha_power=alpha_power)
    warn_alpha_power_near_unity_cpp(medium)
    assert not any("dispersion-singular" in str(w.message) for w in recwarn.list)


def test_warn_helper_silent_when_no_absorption(recwarn):
    """``warn_alpha_power_near_unity_cpp`` must stay silent when ``alpha_coeff`` is zero/unset."""
    from kwave.utils.checks import warn_alpha_power_near_unity_cpp

    warn_alpha_power_near_unity_cpp(kWaveMedium(sound_speed=1500.0, density=1000.0))
    warn_alpha_power_near_unity_cpp(kWaveMedium(sound_speed=1500.0, density=1000.0, alpha_coeff=0.0, alpha_power=1.0))
    assert not any("dispersion-singular" in str(w.message) for w in recwarn.list)


def test_warn_helper_handles_array_alpha_power_with_singular_element():
    """Heterogeneous ``alpha_power`` must trigger the warning if *any* element is singular."""
    from kwave.utils.checks import warn_alpha_power_near_unity_cpp

    # First element is safe (1.5), but the third is in the singular range — must still warn.
    alpha_power = np.array([1.5, 1.5, 1.0, 1.5])
    medium = kWaveMedium(sound_speed=1500.0, density=1000.0, alpha_coeff=0.5, alpha_power=alpha_power)
    with pytest.warns(UserWarning, match="dispersion-singular"):
        warn_alpha_power_near_unity_cpp(medium)


def test_warn_helper_silent_for_array_alpha_power_all_safe(recwarn):
    """Heterogeneous ``alpha_power`` with no element in the singular range must stay silent."""
    from kwave.utils.checks import warn_alpha_power_near_unity_cpp

    alpha_power = np.array([1.5, 1.7, 0.5, 2.0])
    medium = kWaveMedium(sound_speed=1500.0, density=1000.0, alpha_coeff=0.5, alpha_power=alpha_power)
    warn_alpha_power_near_unity_cpp(medium)
    assert not any("dispersion-singular" in str(w.message) for w in recwarn.list)


@pytest.mark.parametrize("alpha_power", [1.5, 1.1])
def test_legacy_cpp_unaffected_when_alpha_mode_unset(alpha_power, tmp_path):
    """Default path (no alpha_mode) must still dispatch normally to the C++ binary.

    Uses ``alpha_power`` values comfortably away from 1.0 — the dispersion-singular
    region is exactly the case ``alpha_mode='no_dispersion'`` exists to handle and is
    not what this test exercises.
    """
    N = Vector([64, 64])
    dx = Vector([0.1e-3, 0.1e-3])
    kgrid = kWaveGrid(N, dx)
    kgrid.makeTime(1500)
    medium = kWaveMedium(
        sound_speed=1500 * np.ones(tuple(N)),
        density=1000 * np.ones(tuple(N)),
        alpha_coeff=0.5 * np.ones(tuple(N)),
        alpha_power=alpha_power,
    )
    source = kSource()
    source.p_mask = np.zeros(tuple(N), dtype=bool)
    source.p_mask[N.x // 2, N.y // 2] = True
    t = kgrid.t_array.squeeze()
    source.p = (np.sin(2 * np.pi * 1e6 * t) * np.exp(-((t - 5e-7) ** 2) / (2e-7) ** 2))[np.newaxis, :]
    sensor = kSensor(mask=np.ones(tuple(N), dtype=bool))

    so = SimulationOptions(
        pml_inside=True,
        smooth_p0=False,
        save_to_disk=True,
        data_path=str(tmp_path),
        input_filename=f"baseline_{alpha_power}.h5",
        output_filename=f"baseline_out_{alpha_power}.h5",
    )
    eo = SimulationExecutionOptions(is_gpu_simulation=False, show_sim_log=False)

    try:
        sensor_data = kspaceFirstOrder2DC(
            kgrid=kgrid,
            source=deepcopy(source),
            sensor=sensor,
            medium=medium,
            simulation_options=so,
            execution_options=eo,
        )
    except FileNotFoundError as e:
        pytest.skip(f"C++ binary not installed: {e}")
    except subprocess.CalledProcessError as e:
        # macOS dyld:  "Library not loaded" / "image not found"
        # Linux ld.so: "error while loading shared libraries"
        # Windows:     "system error" / DLL missing dialog
        msg = e.stderr or ""
        if any(s in msg for s in ("Library not loaded", "image not found", "error while loading shared libraries")):
            pytest.skip(f"C++ binary missing system libraries: {msg.splitlines()[0] if msg else e}")
        raise

    p = np.asarray(sensor_data["p"])
    assert not np.any(np.isnan(p)), f"Default path NaN'd at alpha_power={alpha_power}"
