"""
Steering A Linear Array Example

Ported from: k-Wave/examples/example_tvsp_steering_linear_array.m

Demonstrates how to use k-Wave to steer a tone burst from a linear array
transducer in 2D.  A 21-element linear source at x_offset=25 (1-based) is
driven with per-element time delays computed via geometric beamforming for
a 30-degree steering angle.

The original MATLAB example passes an empty sensor (full-domain movie).
Here we record p and p_final on a full binary grid so the result can be
compared quantitatively with the MATLAB reference.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst


def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 128
    Ny = 128
    dx = 50e-3 / Nx  # [m]
    dy = dx
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500,  # [m/s]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define source mask for a linear transducer with an odd number of
    # elements   (MATLAB 1-based indexing -> Python 0-based)
    num_elements = 21
    x_offset = 25  # 1-based grid index
    start_index = Ny // 2 - (num_elements + 1) // 2 + 1  # 1-based: 54 (MATLAB round)
    source_mask = np.zeros((Nx, Ny), dtype=bool)
    # 0-based: row 24, columns 53..73 (inclusive)
    source_mask[x_offset - 1, start_index - 1 : start_index - 1 + num_elements] = True

    # define tone burst properties
    sampling_freq = 1 / kgrid.dt
    steering_angle = 30  # [deg]
    element_spacing = dx  # [m]
    tone_burst_freq = 1e6  # [Hz]
    tone_burst_cycles = 8

    # element index relative to centre element
    element_index = np.arange(-(num_elements - 1) / 2, (num_elements - 1) / 2 + 1)

    # geometric beamforming offsets
    tone_burst_offset = 40 + element_spacing * element_index * np.sin(steering_angle * np.pi / 180) / (medium.sound_speed * kgrid.dt)

    # create the tone burst signals  (num_elements x signal_length)
    signals = tone_burst(
        sampling_freq,
        tone_burst_freq,
        tone_burst_cycles,
        signal_offset=np.round(tone_burst_offset).astype(int),
    )

    source = kSource()
    source.p_mask = source_mask
    source.p = signals

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run the simulation with a full-grid binary sensor.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    Nx, Ny = 128, 128
    sensor = kSensor()
    sensor.mask = np.ones((Nx, Ny), dtype=bool)
    sensor.record = ["p", "p_final"]

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

    kgrid, _, source = setup()
    result = run(quiet=False)
    pf = np.asarray(result["p_final"])
    side = int(np.sqrt(pf.size))
    p_final = pf.reshape(side, side)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    im = ax.imshow(
        p_final.T,
        extent=[
            kgrid.x_vec[0] * 1e3,
            kgrid.x_vec[-1] * 1e3,
            kgrid.y_vec[-1] * 1e3,
            kgrid.y_vec[0] * 1e3,
        ],
        cmap="RdBu_r",
    )
    ax.set_xlabel("x-position [mm]")
    ax.set_ylabel("y-position [mm]")
    ax.set_title("Final Pressure Field")
    fig.colorbar(im, ax=ax)

    # plot source signals (stacked)
    ax = axes[1]
    num_source_time_points = source.p.shape[1]
    t_us = np.asarray(kgrid.t_array).squeeze()[:num_source_time_points] * 1e6
    for i in range(source.p.shape[0]):
        ax.plot(t_us, source.p[i, :] + i, linewidth=0.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Input Signals (stacked)")
    ax.set_title("Tone Burst Signals")

    fig.suptitle("Steering A Linear Array Example")
    fig.tight_layout()
    plt.show()
