"""
Filtering A Delta Function Input Signal Example Part 1

Ported from: k-Wave/examples/example_na_filtering_part_1.m

Illustrates the numerical aliasing that results from applying a temporal
delta-function pressure pulse without filtering or smoothing. A single-point
source emits an unfiltered impulse in a 1-D homogeneous medium. Because the
impulse excites all temporal frequencies equally — including those above the
spatial Nyquist limit of the grid — the recorded signal at the opposite end of
the domain contains significant high-frequency aliasing artefacts.

The default CFL case is ported here: dt = 7 ns fixed, Nt = 1024 steps.
The PML is placed *outside* the computational domain (PMLInside = false)
with alpha = 0 (no absorption) so it acts purely as a grid extension.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder


def setup():
    """Set up the simulation physics (grid, medium, source).

    Grid: 256 points, dx = 10e-3 / 256 m.
    Medium: homogeneous, c = 1500 m/s.
    Source: single point at grid index 50 (0-based), delta-function pulse
            of magnitude 2 Pa at time step 99 (0-based).
    Time array: dt = 7 ns, Nt = 1024 (set explicitly, not via makeTime).

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 256  # number of grid points [grid points]
    dx = 10e-3 / Nx  # grid point spacing [m]
    kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create time array (fixed dt and Nt, not CFL-derived)
    dt = 7e-9  # time step [s]
    Nt = 1024  # number of time steps
    kgrid.setTime(Nt, dt)

    # define a single element source
    # MATLAB: source.p_mask(1 + source_offset, 1) = 1  ->  index 51 (1-based)  ->  50 (0-based)
    source_offset = 50
    source = kSource()
    p_mask = np.zeros(Nx, dtype=float)
    p_mask[source_offset] = 1
    source.p_mask = p_mask

    # define a delta function input pulse
    # MATLAB: source.p(temporal_offset) = source_magnitude  ->  index 100 (1-based)  ->  99 (0-based)
    temporal_offset = 100  # [time steps, 1-based in MATLAB]
    source_magnitude = 2  # [Pa]
    source.p = np.zeros(Nt, dtype=float)
    source.p[temporal_offset - 1] = source_magnitude

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with a full-grid binary sensor.

    The MATLAB original uses PMLInside=false, PMLSize=128, PMLAlpha=0.
    The Python solver only supports pml_inside=True, so we use that and
    let the solver handle PML internally.

    Returns:
        dict: Simulation results with keys 'p' and 'p_final'.
    """
    kgrid, medium, source = setup()

    # full-grid binary sensor recording p and p_final
    Nx = 256
    sensor = kSensor(
        mask=np.ones(Nx, dtype=float),
        record=["p", "p_final"],
    )

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

    p = np.asarray(result["p"])
    t_array = np.asarray(kgrid.t_array).ravel()

    # plot input and recorded time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    t_us = t_array * 1e6  # convert to microseconds
    source_p = np.asarray(source.p).ravel()

    # find the sensor point closest to the original single-element position
    # MATLAB: sensor.mask(end - source_offset) = 1  ->  index 206 (1-based)  ->  row 205 (0-based)
    sensor_row = 205
    ax1.plot(t_us, source_p, "k-", label="Input pulse")
    ax1.plot(t_us, p[sensor_row, :], "b-", label="Recorded pulse")
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [au]")
    ax1.legend()
    ax1.set_title("Input & Recorded Time Series (Part 1: No Filtering)")

    # compute and plot amplitude spectra
    dt = 7e-9
    Fs = 1 / dt
    N = len(source_p)
    f = np.fft.rfftfreq(N, d=dt)
    input_as = np.abs(np.fft.rfft(source_p)) / N
    output_as = np.abs(np.fft.rfft(p[sensor_row, :])) / N

    f_max = float(kgrid.k_max * np.min(medium.sound_speed) / (2 * np.pi))
    f_MHz = f * 1e-6

    ax2.plot(f_MHz, input_as, "k-", label="Input spectrum")
    ax2.plot(f_MHz, output_as, "b-", label="Recorded spectrum")
    ax2.axvline(f_max * 1e-6, color="k", linestyle="--", label="Max grid frequency")
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Amplitude [au]")
    ax2.legend(loc="lower left")
    ax2.set_title("Amplitude Spectra")

    fig.tight_layout()
    plt.show()
