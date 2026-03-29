"""
Monopole Point Source In A Homogeneous Propagation Medium Example

Ported from: k-Wave/examples/example_tvsp_homogeneous_medium_monopole.m

Simulates a time-varying sinusoidal pressure source (monopole) within a
two-dimensional homogeneous propagation medium with power-law absorption.
A single source point emits a filtered 0.25 MHz tone burst which propagates
outward as an expanding circular wavefront. A single sensor point records
the arriving pressure waveform.

This is the first time-varying source example and builds on the initial
value problem demonstrations.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import filter_time_series


def setup():
    """Set up simulation physics (grid, medium, source).

    Grid: 128 x 128, dx = dy = 50e-3/128 m (~0.391 mm).
    Medium: c = 1500 m/s, alpha_coeff = 0.75 dB/(MHz^y cm), alpha_power = 1.5.
    Source: single point at (95, 63) [0-based], i.e. MATLAB (end-Nx/4, Ny/2),
            emitting a 0.25 MHz sinusoid filtered by filterTimeSeries.

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 128  # number of grid points in the x direction
    Ny = 128  # number of grid points in the y direction
    dx = 50e-3 / Nx  # grid point spacing in the x direction [m]
    dy = dx  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a single source point
    # MATLAB: source.p_mask(end - Nx/4, Ny/2) = 1  ->  (96, 64) 1-based  ->  (95, 63) 0-based
    source = kSource()
    p_mask = np.zeros((Nx, Ny), dtype=float)
    p_mask[95, 63] = 1
    source.p_mask = p_mask

    # define a time varying sinusoidal source
    source_freq = 0.25e6  # [Hz]
    source_mag = 2  # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with the original single-point sensor from the MATLAB example.

    Sensor: single point at (31, 63) [0-based], i.e. MATLAB (Nx/4, Ny/2).
    Records: p and p_final.

    Returns:
        dict: Simulation results with keys 'p' (1 x Nt) and 'p_final' (Nx x Ny).
    """
    kgrid, medium, source = setup()

    # define a single sensor point
    # MATLAB: sensor.mask(Nx/4, Ny/2) = 1  ->  (32, 64) 1-based  ->  (31, 63) 0-based
    Nx, Ny = 128, 128
    sensor_mask = np.zeros((Nx, Ny), dtype=float)
    sensor_mask[31, 63] = 1
    sensor = kSensor(mask=sensor_mask, record=["p", "p_final"])

    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=True,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kgrid, medium, source = setup()
    result = run(quiet=False)

    p_final = np.asarray(result["p_final"])
    p_sensor = np.asarray(result["p"])

    # --- Figure 1: final wave-field ---
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    extent = [
        kgrid.y_vec[0] * 1e3,
        kgrid.y_vec[-1] * 1e3,
        kgrid.x_vec[-1] * 1e3,
        kgrid.x_vec[0] * 1e3,
    ]
    ax1.imshow(p_final, extent=extent, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xlabel("y-position [mm]")
    ax1.set_ylabel("x-position [mm]")
    ax1.set_title("Final Wave Field + Source/Sensor Positions")
    ax1.set_aspect("equal")

    # --- Figure 2: input and sensor signals ---
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6  # convert to microseconds

    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6))

    ax_top.plot(t_us, np.asarray(source.p).ravel(), "k-")
    ax_top.set_xlabel("Time [us]")
    ax_top.set_ylabel("Signal Amplitude")
    ax_top.set_title("Input Pressure Signal")
    ax_top.set_xlim(t_us[0], t_us[-1])

    ax_bot.plot(t_us, p_sensor.ravel(), "r-")
    ax_bot.set_xlabel("Time [us]")
    ax_bot.set_ylabel("Signal Amplitude")
    ax_bot.set_title("Sensor Pressure Signal")
    ax_bot.set_xlim(t_us[0], t_us[-1])

    fig2.suptitle("Monopole Point Source Example")
    fig2.tight_layout()
    plt.show()
