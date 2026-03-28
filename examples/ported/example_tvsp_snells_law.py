"""
Snell's Law And Critical Angle Reflection Example

Ported from: k-Wave/examples/example_tvsp_snells_law.m

Illustrates Snell's law by steering a tone burst from a 61-element linear
array transducer at 35 degrees into a two-layer medium (1500/3000 m/s).
It builds on the Steering A Linear Array example.

You should observe the transmitted beam refracting according to Snell's
law at the interface, and the reflected beam at the complementary angle.
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
    Nx = 128  # number of grid points in the x direction
    Ny = Nx  # number of grid points in the y direction
    dx = 50e-3 / Nx  # grid point spacing [m]
    dy = dx
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define sound speed in the two layers
    c1 = 1500  # [m/s]
    c2 = 3000  # [m/s]

    # define the properties of the propagation medium
    c = c1 * np.ones((Nx, Ny))  # [m/s]
    c[Nx // 2 - 1 :, :] = c2  # MATLAB: Nx/2:end (1-based 64:128 = 0-based 63:)
    medium = kWaveMedium(
        sound_speed=c,
        density=1000,  # [kg/m^3]
        alpha_coeff=0.75,  # [dB/(MHz^y cm)]
        alpha_power=1.5,
    )

    # create the time array
    kgrid.makeTime(c)

    # define a source mask for a 61-element linear transducer
    num_elements = 61
    x_offset = 25  # [grid points, 1-based]
    y_offset = 20  # [grid points]
    start_index = Ny // 2 - (num_elements + 1) // 2 + 1 - y_offset  # 1-based (MATLAB round)
    source = kSource()
    source.p_mask = np.zeros((Nx, Ny))
    # Convert to 0-based: x_offset-1=24, start_index-1 for range start
    source.p_mask[x_offset - 1, start_index - 1 : start_index - 1 + num_elements] = 1

    # define the properties of the tone burst
    sampling_freq = 1 / kgrid.dt  # [Hz]
    steering_angle = 35  # [deg]
    element_spacing = dx  # [m]
    tone_burst_freq = 1e6  # [Hz]
    tone_burst_cycles = 8

    # create element index relative to centre element
    element_index = np.arange(-(num_elements - 1) / 2, (num_elements - 1) / 2 + 1)

    # geometric beam forming: per-element time delays
    tone_burst_offset = 200 + element_spacing * element_index * np.sin(steering_angle * np.pi / 180) / (c1 * kgrid.dt)

    # create the tone burst signals
    source.p = tone_burst(
        sampling_freq,
        tone_burst_freq,
        tone_burst_cycles,
        signal_offset=np.round(tone_burst_offset).astype(int),
    )

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with a full-grid sensor recording p_final.

    The MATLAB original passes an empty sensor (visualisation only).
    Here we record p_final to show the refracted beam pattern.

    Returns:
        dict: Simulation results with key 'p_final'.
    """
    kgrid, medium, source = setup()
    Nx = 128
    Ny = 128
    sensor = kSensor(mask=np.ones((Nx, Ny), dtype=bool))
    sensor.record = ["p_final"]
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

    result = run(quiet=False)
    p_final = np.asarray(result["p_final"])

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.max(np.abs(p_final))
    ax.imshow(p_final, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xlabel("y [grid points]")
    ax.set_ylabel("x [grid points]")
    ax.set_title("Snell's Law — Final Pressure Field")
    plt.tight_layout()
    plt.show()
