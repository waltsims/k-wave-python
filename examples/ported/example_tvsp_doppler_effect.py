"""
The Doppler Effect Example

Ported from: k-Wave/examples/example_tvsp_doppler_effect.m

Demonstrates the Doppler effect: a stationary sensor records a frequency
shift as a moving pressure source travels past. The source is interpolated
between grid points to simulate smooth motion at 150 m/s through a
homogeneous medium (c = 1500 m/s, i.e. Mach 0.1). Power-law absorption
is included (alpha = 0.75 dB/(MHz^y cm), y = 1.5).

The time array is set manually (Nt = 4500, dt = 20 ns) rather than via
kgrid.makeTime, since the example requires fine temporal resolution to
resolve the moving source position.

It builds on the Monopole Point Source In A Homogeneous Propagation Medium
Example.
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

    Grid: 64 x 128, dx = dy = 20e-3/128 m (~0.156 mm).
    Medium: c = 1500 m/s, alpha_coeff = 0.75 dB/(MHz^y cm), alpha_power = 1.5.
    Source: a row of points along y (inside PML margins) with a sinusoidal
            pressure signal (0.75 MHz) that slides along the row to simulate
            a source moving at 150 m/s.  Linear interpolation distributes
            the signal between adjacent grid points at each time step.

    The time array is manually set: Nt = 4500, dt = 20 ns.

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 64  # number of grid points in the x (row) direction
    Ny = Nx * 2  # number of grid points in the y (column) direction = 128
    dy = 20e-3 / Ny  # grid point spacing in the y direction [m]
    dx = dy  # grid point spacing in the x direction [m]
    pml_size = 20  # PML thickness [grid points]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # set the velocity of the moving source
    source_vel = 150  # [m/s]

    # manually create the time array (Nt = 4500, dt = 20 ns)
    Nt = 4500
    dt = 20e-9  # [s]
    kgrid.setTime(Nt, dt)

    # define a single time-varying sinusoidal source
    source_freq = 0.75e6  # [Hz]
    source_mag = 3  # [Pa]
    t_array = np.asarray(kgrid.t_array).ravel()  # shape (Nt,)
    source_pressure = source_mag * np.sin(2 * np.pi * source_freq * t_array)

    # filter the source to remove high frequencies not supported by the grid
    # filter_time_series expects a 2D (num_signals x Nt) array
    source_pressure = filter_time_series(kgrid, medium, source_pressure.reshape(1, -1)).ravel()

    # define a line of source points along y (inside PML margins)
    # MATLAB: source.p_mask(end - pml_size - source_x_pos, 1+pml_size:end-pml_size) = 1
    #   end - pml_size - source_x_pos  = 64 - 20 - 5 = 39  (1-based)  -> 38 (0-based)
    #   1+pml_size:end-pml_size        = 21:108  (1-based)  -> 20:108 (0-based)
    source_x_pos = 5  # [grid points]
    source = kSource()
    p_mask = np.zeros((Nx, Ny), dtype=float)
    p_mask[Nx - 1 - pml_size - source_x_pos, pml_size : Ny - pml_size] = 1
    source.p_mask = p_mask

    # preallocate an empty pressure source matrix
    num_source_positions = int(np.sum(p_mask))  # = Ny - 2*pml_size = 88
    source_p = np.zeros((num_source_positions, Nt))

    # move the source along the source mask by interpolating the pressure
    # series between the source elements
    # NOTE: MATLAB uses 1-based indexing; here sensor_index and t_index are 0-based
    sensor_index = 0  # current left grid-point pair index (0-based)
    t_index = 0  # current time step (0-based)

    # TODO: vectorize this loop — the entire source_p matrix can be computed
    # with NumPy fancy indexing instead of iterating over 4500 time steps.
    while t_index < Nt and sensor_index < num_source_positions - 2:
        # check if the source has moved to the next pair of grid points
        # MATLAB: kgrid.t_array(t_index) > (sensor_index * dy / source_vel)
        # In MATLAB sensor_index starts at 1, so condition uses sensor_index*dy.
        # Here sensor_index is 0-based, equivalent MATLAB sensor_index = sensor_index+1.
        if t_array[t_index] > ((sensor_index + 1) * dy / source_vel):
            sensor_index += 1

        # calculate the position of source between the two current grid points
        exact_pos = source_vel * t_array[t_index]
        discrete_pos = (sensor_index + 1) * dy  # MATLAB: sensor_index * dy (1-based)
        pos_ratio = (discrete_pos - exact_pos) / dy

        # update the pressure at the two current grid points using linear interpolation
        # MATLAB: source.p(sensor_index, t_index) and source.p(sensor_index+1, t_index)
        source_p[sensor_index, t_index] = pos_ratio * source_pressure[t_index]
        source_p[sensor_index + 1, t_index] = (1 - pos_ratio) * source_pressure[t_index]

        t_index += 1

    source.p = source_p

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with a single-point sensor and full-grid for p_final.

    Sensor: single point at (Nx-1 - pml_size - source_x_pos - source_sensor_x_distance,
    Ny//2 - 1) [0-based].
    MATLAB: (end - pml_size - source_x_pos - source_sensor_x_distance, Ny/2)
          = (64 - 20 - 5 - 5, 64) = (34, 64) 1-based  -> (33, 63) 0-based.

    Records: p and p_final (full grid).

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx = 64
    Ny = 128
    pml_size = 20
    source_x_pos = 5
    source_sensor_x_distance = 5

    # full-grid binary sensor to capture p_final and p at all points
    sensor_mask = np.ones((Nx, Ny), dtype=bool)
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

    source_p = np.asarray(source.p)
    p_sensor = np.asarray(result["p"])

    # --- Figure 1: moving pressure field ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.imshow(source_p, aspect="auto", cmap="RdBu_r")
    ax1.set_ylabel("Source Position")
    ax1.set_xlabel("Time Step")
    ax1.set_title("Input Pressure Signal")

    ax2.plot(np.sum(source_p, axis=0))
    ax2.set_ylabel("Pressure [au]")
    ax2.set_xlabel("Time Step")
    ax2.set_title("Sum Of Input Pressure Signal Across All Source Positions")
    fig1.tight_layout()

    # --- Figure 2: sensor pressure ---
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6

    fig2, ax = plt.subplots(figsize=(8, 4))
    # extract the single sensor point from full-grid data
    # sensor point is at (33, 63) in the full grid
    # For full-grid sensor, p has shape (Nx*Ny, Nt); index = 33*Ny + 63
    sensor_idx = 33 * 128 + 63
    ax.plot(t_us, p_sensor[sensor_idx, :], "r-")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Signal Amplitude")
    ax.set_title("Sensor Pressure Signal")
    ax.set_xlim(t_us[0], t_us[-1])

    plt.tight_layout()
    plt.show()
