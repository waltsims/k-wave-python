
import numpy as np
import pytest

from kwave.kgrid import kWaveGrid

from kwave.kwaveDiffusion import (
    DiffusionMedium,
    DiffusionSource,
    kWaveDiffusion,
)


@pytest.fixture
def grid():
    N = 32
    dx = 0.5e-3
    return kWaveGrid([N], [dx])

def to_numpy(value):

    if hasattr(value, "get"):
        return value.get()

    return np.asarray(value)

def test_uniform_heating(grid):

    rho = 1000.0
    specific_heat = 4000.0

    heating_rate = 1.0
    dt = 2.0

    medium = DiffusionMedium(
        diffusion_coeff=1.4e-7,
        density=rho,
        specific_heat=specific_heat,
    )

    source = DiffusionSource(
        T0=37.0,
        Q=heating_rate * rho * specific_heat,
    )

    solver = kWaveDiffusion(
        grid,
        medium,
        source,
        use_kspace=True,
        display_updates=False,
    )

    T = to_numpy(solver.take_time_step(1, dt))

    expected = 37.0 + heating_rate * dt

    np.testing.assert_allclose(
        T,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_fourier_mode_decay(grid):

    N = 32
    dx = 0.5e-3

    D = 1.4e-7

    x = np.arange(N)

    mode = 3

    T0 = (
        37.0
        + np.cos(2.0 * np.pi * mode * x / N)
    )

    medium = DiffusionMedium(
        diffusion_coeff=D,
    )

    source = DiffusionSource(T0=T0)

    solver = kWaveDiffusion(
        grid,
        medium,
        source,
        use_kspace=True,
        display_updates=False,
    )

    dt = 2.0

    T = to_numpy(solver.take_time_step(1, dt))

    k = 2.0 * np.pi * mode / (N * dx)

    expected = (
        37.0
        + np.cos(2.0 * np.pi * mode * x / N)
        * np.exp(-D * k**2 * dt)
    )

    np.testing.assert_allclose(
        T,
        expected,
        rtol=1e-9,
        atol=1e-9,
    )


def test_scalar_and_constant_array_agree(grid):

    N = 32

    D = 1.4e-7

    T0 = 37.0 + np.cos(
        2.0 * np.pi * 3 * np.arange(N) / N
    )

    source = DiffusionSource(T0=T0)

    solver_scalar = kWaveDiffusion(
        grid,
        DiffusionMedium(diffusion_coeff=D),
        source,
        display_updates=False,
    )

    solver_array = kWaveDiffusion(
        grid,
        DiffusionMedium(
            diffusion_coeff=np.full(N, D)
        ),
        source,
        display_updates=False,
    )

    T_scalar = solver_scalar.take_time_step(1, 2.0)
    T_array = solver_array.take_time_step(1, 2.0)

    np.testing.assert_allclose(
        T_scalar,
        T_array,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sensor_recording_multiple_calls(grid):

    N = 32

    mask = np.zeros(N)
    mask[[5, 10, 15]] = 1

    medium = DiffusionMedium(
        diffusion_coeff=1.4e-7,
    )

    source = DiffusionSource(T0=37.0)

    solver = kWaveDiffusion(
        grid,
        medium,
        source,
        sensor=mask,
        display_updates=False,
    )

    to_numpy(solver.take_time_step(1, 0.1))
    to_numpy(solver.take_time_step(2, 0.1))

    assert solver.sensor_data.shape == (3, 3)
    assert solver.time_steps_taken == 3


def test_zero_source_can_be_activated(grid):

    rho = 1000.0
    specific_heat = 4000.0

    medium = DiffusionMedium(
        diffusion_coeff=1.4e-7,
        density=rho,
        specific_heat=specific_heat,
    )

    source = DiffusionSource(
        T0=37.0,
        Q=0.0,
    )

    solver = kWaveDiffusion(
        grid,
        medium,
        source,
        display_updates=False,
    )

    solver.Q = 1.0e6

    T = to_numpy(solver.take_time_step(1, 1.0))

    assert np.max(T) > 37.0


def test_dt_limit_uncorrected(grid):

    D = 1.4e-7

    solver = kWaveDiffusion(
        grid,
        DiffusionMedium(diffusion_coeff=D),
        DiffusionSource(T0=37.0),
        use_kspace=False,
        display_updates=False,
    )

    assert np.isfinite(solver.dt_limit)
    assert solver.dt_limit > 0

def test_nyquist_mode_decay(grid):

    N = 32
    D = 1.4e-7

    x = np.arange(N)

    T0 = 37.0 + (-1.0) ** x

    solver = kWaveDiffusion(
        grid,
        DiffusionMedium(
            diffusion_coeff=np.full(N, D)
        ),
        DiffusionSource(T0=T0),
        display_updates=False,
    )

    T = to_numpy(solver.take_time_step(1, 2.0))

    k_nyquist = np.pi / grid.dx

    expected = (
        37.0
        + (-1.0) ** x
        * np.exp(-D * k_nyquist**2 * 2.0)
    )

    np.testing.assert_allclose(
        T,
        expected,
        rtol=1e-8,
        atol=1e-8,
    )

def test_dt_limit_zero_reference(grid):

    D = 1.4e-7

    solver = kWaveDiffusion(
        grid,
        DiffusionMedium(
            diffusion_coeff=D,
            diffusion_coeff_ref=0.0,
        ),
        DiffusionSource(T0=37.0),
        use_kspace=True,
        display_updates=False,
    )

    expected = 2.0 / (
        D * np.max(solver.k) ** 2
    )

    np.testing.assert_allclose(
        solver.dt_limit,
        expected,
        rtol=1e-10,
    )

def test_cpu_backend(grid):

    solver = kWaveDiffusion(
        grid,
        DiffusionMedium(
            diffusion_coeff=1.4e-7,
        ),
        DiffusionSource(T0=37.0),
        device="cpu",
        display_updates=False,
    )

    T = to_numpy(solver.take_time_step(1, 1.0))

    assert isinstance(T, np.ndarray)

    assert np.all(np.isfinite(T))

def test_temperature_remains_constant(grid):

    solver = kWaveDiffusion(
        grid,
        DiffusionMedium(
            diffusion_coeff=1.4e-7,
        ),
        DiffusionSource(T0=37.0),
        display_updates=False,
    )

    T = to_numpy(solver.take_time_step(10, 0.1))

    np.testing.assert_allclose(
        T,
        37.0,
        atol=1e-12,
    )

def test_heterogeneous_diffusion_changes_temperature(grid):

    N = 32
    D = np.full(N, 1.4e-7)
    D[8:16] = 2.8e-7

    x = np.arange(N)

    T0 = 37.0 + np.cos(
        2.0 * np.pi * 3 * x / N
    )

    solver = kWaveDiffusion(
        grid,
        DiffusionMedium(diffusion_coeff=D),
        DiffusionSource(T0=T0),
        display_updates=False,
    )

    T = to_numpy(solver.take_time_step(1, 2.0))

    assert np.all(np.isfinite(T))
    assert not np.allclose(T, T0)
