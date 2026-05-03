"""Tests for alpha_power near 1.0 (tan(pi*y/2) singularity)."""
import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder


def _make_sim():
    """Create a minimal 2D simulation with a point source."""
    N = (64, 64)
    dx = (1e-4, 1e-4)
    kgrid = kWaveGrid(N, dx)
    kgrid.makeTime(1500)

    source = kSource()
    source.p0 = np.zeros(N)
    source.p0[32, 32] = 1.0

    sensor = kSensor(mask=np.ones(N))
    return kgrid, source, sensor


def test_alpha_power_near_unity_raises_without_no_dispersion():
    """alpha_power close to 1.0 must raise when dispersion is enabled."""
    kgrid, source, sensor = _make_sim()
    medium = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.5, alpha_power=1.01)

    with pytest.raises(ValueError, match="too close to 1.0"):
        kspaceFirstOrder(kgrid, medium, source, sensor, pml_inside=True, quiet=True)


def test_alpha_power_near_unity_works_with_no_dispersion():
    """alpha_power close to 1.0 must produce valid output when dispersion is disabled."""
    kgrid, source, sensor = _make_sim()
    medium = kWaveMedium(
        sound_speed=1500,
        density=1000,
        alpha_coeff=0.5,
        alpha_power=1.01,
        alpha_mode="no_dispersion",
    )

    result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_inside=True, quiet=True)
    p = np.asarray(result["p"])
    assert not np.any(np.isnan(p)), "Output contains NaN with alpha_power=1.01 and no_dispersion"
    assert not np.all(p == 0), "Output is all zeros — absorption had no effect"


def test_alpha_mode_no_absorption():
    """alpha_mode='no_absorption' should disable absorption but keep dispersion."""
    kgrid, source, sensor = _make_sim()
    medium_lossy = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.5, alpha_power=1.5)
    medium_no_abs = kWaveMedium(
        sound_speed=1500,
        density=1000,
        alpha_coeff=0.5,
        alpha_power=1.5,
        alpha_mode="no_absorption",
    )

    result_lossy = kspaceFirstOrder(kgrid, medium_lossy, source, sensor, pml_inside=True, quiet=True)
    result_no_abs = kspaceFirstOrder(kgrid, medium_no_abs, source, sensor, pml_inside=True, quiet=True)

    p_lossy = np.asarray(result_lossy["p"])
    p_no_abs = np.asarray(result_no_abs["p"])

    assert not np.any(np.isnan(p_no_abs))
    # Disabling absorption should change the output
    assert not np.allclose(p_lossy, p_no_abs), "Disabling absorption should change the output"


def test_alpha_mode_no_dispersion():
    """alpha_mode='no_dispersion' should disable dispersion but keep absorption."""
    kgrid, source, sensor = _make_sim()
    medium_lossy = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.5, alpha_power=1.5)
    medium_no_disp = kWaveMedium(
        sound_speed=1500,
        density=1000,
        alpha_coeff=0.5,
        alpha_power=1.5,
        alpha_mode="no_dispersion",
    )

    result_lossy = kspaceFirstOrder(kgrid, medium_lossy, source, sensor, pml_inside=True, quiet=True)
    result_no_disp = kspaceFirstOrder(kgrid, medium_no_disp, source, sensor, pml_inside=True, quiet=True)

    p_lossy = np.asarray(result_lossy["p"])
    p_no_disp = np.asarray(result_no_disp["p"])

    assert not np.any(np.isnan(p_no_disp))
    # Both should have absorption so similar amplitude, but waveforms differ
    assert not np.allclose(p_lossy, p_no_disp), "Disabling dispersion should change the output"


def test_alpha_power_normal_range_unaffected():
    """alpha_power=1.5 (well away from singularity) should work as before."""
    kgrid, source, sensor = _make_sim()
    medium = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.5, alpha_power=1.5)

    result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_inside=True, quiet=True)
    p = np.asarray(result["p"])
    assert not np.any(np.isnan(p))
    assert p.shape[0] > 0


def test_cpp_backend_warns_on_alpha_mode(tmp_path):
    """C++ backend should warn when alpha_mode is set (it cannot honor it)."""
    kgrid, source, sensor = _make_sim()
    medium = kWaveMedium(
        sound_speed=1500,
        density=1000,
        alpha_coeff=0.5,
        alpha_power=1.5,
        alpha_mode="no_dispersion",
    )

    with pytest.warns(UserWarning, match="not supported by the C\\+\\+ backend"):
        kspaceFirstOrder(
            kgrid,
            medium,
            source,
            sensor,
            pml_inside=True,
            quiet=True,
            backend="cpp",
            save_only=True,
            data_path=str(tmp_path),
        )


def test_cpp_backend_skips_near_unity_guard(tmp_path):
    """C++ backend uses its own dispersion formulation; the Python near-unity guard
    must not block the call. Otherwise C++ users have no valid escape hatch
    (alpha_mode='no_dispersion' is silently ignored by the binary)."""
    kgrid, source, sensor = _make_sim()
    medium = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.5, alpha_power=0.97)

    # Should not raise — only save_only path is exercised so we don't need the binary.
    kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        pml_inside=True,
        quiet=True,
        backend="cpp",
        save_only=True,
        data_path=str(tmp_path),
    )
