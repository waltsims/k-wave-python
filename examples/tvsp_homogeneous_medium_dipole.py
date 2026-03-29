"""
Dipole Point Source In A Homogeneous Propagation Medium Example

Ported from: k-Wave/examples/example_tvsp_homogeneous_medium_dipole.m

Simulates the detection of a time-varying velocity dipole source within a
two-dimensional homogeneous propagation medium with power-law absorption.
The source is a sinusoidal particle-velocity signal injected in the
x-direction at a single grid point; this creates a characteristic dipole
radiation pattern. A single-point sensor records the resulting pressure
and the final pressure field.
"""
# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import filter_time_series


# %%
def setup():
    """Set up simulation physics. Returns (kgrid, medium, source)."""

    # create the computational grid
    Nx = 128  # number of grid points in the x direction
    Ny = 128  # number of grid points in the y direction
    dx = 50e-3 / Nx  # grid point spacing in the x direction [m]
    dy = dx  # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
        density=1000,  # [kg/m^3]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a single source point
    # MATLAB: source.u_mask(end - Nx/4, Ny/2) = 1  -> (96, 64) 1-based -> (95, 63) 0-based
    u_mask = np.zeros((Nx, Ny))
    u_mask[95, 63] = 1

    # define a time varying sinusoidal velocity source in the x-direction
    source_freq = 0.25e6  # [Hz]
    source_mag = 2 / (medium.sound_speed * medium.density)
    ux = -source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array.squeeze())

    # filter the source to remove high frequencies not supported by the grid
    source = kSource()
    source.u_mask = u_mask
    source.ux = filter_time_series(kgrid, medium, ux.reshape(1, -1))

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run with original sensor from MATLAB example."""
    kgrid, medium, source = setup()

    # define a single sensor point
    # MATLAB: sensor.mask(Nx/4, Ny/2) = 1  -> (32, 64) 1-based -> (31, 63) 0-based
    Nx, Ny = 128, 128
    sensor_mask = np.zeros((Nx, Ny))
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


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    p = np.asarray(result["p"]).squeeze()
    p_final = np.asarray(result["p_final"])

    kgrid, medium, source = setup()
    t_array = kgrid.t_array.squeeze()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # plot the input velocity signal
    ax = axes[0]
    ax.plot(t_array * 1e6, source.ux.squeeze(), "k-")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Signal Amplitude")
    ax.set_title("Input Velocity Signal")
    ax.set_xlim(t_array[0] * 1e6, t_array[-1] * 1e6)

    # plot the recorded pressure at the sensor
    ax = axes[1]
    ax.plot(t_array * 1e6, p, "r-")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Signal Amplitude")
    ax.set_title("Sensor Pressure Signal")
    ax.set_xlim(t_array[0] * 1e6, t_array[-1] * 1e6)

    fig.suptitle("Dipole Point Source In A Homogeneous Propagation Medium")
    fig.tight_layout()
    plt.show()
