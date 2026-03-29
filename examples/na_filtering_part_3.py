"""
Filtering A Delta Function Input Signal Example Part 3

Ported from: k-Wave/examples/example_na_filtering_part_3.m

Illustrates how to temporally filter an input signal using
``filter_time_series`` before feeding it to the solver. A delta-function
impulse is passed through a causal Kaiser-windowed low-pass filter whose
cut-off is set by the maximum frequency the grid can support (default:
PPW = 3). This removes the above-Nyquist energy that would otherwise alias,
producing a clean recorded waveform at the sensor.

The default case (example_number = 1) is ported: causal filter with default
PPW and transition width.
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
    """Set up the simulation physics (grid, medium, source).

    Grid: 256 points, dx = 10e-3 / 256 m.
    Medium: homogeneous, c = 1500 m/s.
    Source: single point at grid index 50 (0-based), temporally filtered
            delta-function pulse of magnitude 2 Pa.

    The delta pulse is filtered with ``filter_time_series(kgrid, medium, ...)``
    using default settings (causal, PPW = 3, transition_width = 0.1).

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

    # define a delta function input pulse
    # MATLAB: source_func(temporal_offset) = source_magnitude
    temporal_offset = 100  # [time steps, 1-based in MATLAB]
    source_magnitude = 2  # [Pa]
    source_func = np.zeros((1, Nt), dtype=float)  # row vector for filter_time_series
    source_func[0, temporal_offset - 1] = source_magnitude

    # filter the input signal (default causal filter)
    # MATLAB: source_func_filtered = filterTimeSeries(kgrid, medium, source_func)
    source_func_filtered = filter_time_series(kgrid, medium, source_func)

    # define a single element source
    # MATLAB: source.p_mask(1 + source_offset, 1) = 1  ->  index 51 (1-based)  ->  50 (0-based)
    source_offset = 50
    source = kSource()
    p_mask = np.zeros(Nx, dtype=float)
    p_mask[source_offset] = 1
    source.p_mask = p_mask

    # assign filtered input signal
    source.p = source_func_filtered

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with a full-grid binary sensor.

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
    t_us = t_array * 1e6

    # extract filtered source signal
    source_p = np.asarray(source.p).ravel()

    # sensor point matching MATLAB: sensor.mask(end - source_offset)
    sensor_row = 205

    # --- Figure 1: input signals and spectra ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # original (unfiltered) delta pulse for comparison
    Nt = 1024
    dt = 7e-9
    source_func_orig = np.zeros(Nt)
    source_func_orig[99] = 2  # temporal_offset - 1

    ax1.plot(t_us, source_func_orig, "k-", label="Original input")
    ax1.plot(t_us, source_p, "b-", label="Filtered input")
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [au]")
    ax1.legend()
    ax1.set_xlim(0, t_us[-1] * 0.5)
    ax1.set_title("Original & Filtered Input Signals")

    # amplitude spectra
    f = np.fft.rfftfreq(Nt, d=dt)
    orig_as = np.abs(np.fft.rfft(source_func_orig)) / Nt
    filt_as = np.abs(np.fft.rfft(source_p)) / Nt

    f_max = float(kgrid.k_max * np.min(medium.sound_speed) / (2 * np.pi))
    f_MHz = f * 1e-6

    ax2.plot(f_MHz, orig_as, "k-", label="Original spectrum")
    ax2.plot(f_MHz, filt_as, "b-", label="Filtered spectrum")
    ax2.axvline(f_max * 1e-6, color="k", linestyle="--", label="Max grid frequency")
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Amplitude [au]")
    ax2.legend(loc="lower left")
    ax2.set_title("Amplitude Spectra of Input Signals")

    fig1.tight_layout()

    # --- Figure 2: recorded signal ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))

    ax3.plot(t_us, source_p, "k-", label="Input pulse")
    ax3.plot(t_us, p[sensor_row, :], "b-", label="Recorded pulse")
    ax3.set_xlabel("Time [us]")
    ax3.set_ylabel("Pressure [au]")
    ax3.legend()
    ax3.set_title("Input & Recorded Time Series (Part 3: Filtered Source)")

    # amplitude spectra of recorded signal
    output_as = np.abs(np.fft.rfft(p[sensor_row, :])) / Nt

    ax4.plot(f_MHz, filt_as, "k-", label="Input spectrum")
    ax4.plot(f_MHz, output_as, "b-", label="Recorded spectrum")
    ax4.axvline(f_max * 1e-6, color="k", linestyle="--", label="Max grid frequency")
    ax4.set_xlabel("Frequency [MHz]")
    ax4.set_ylabel("Amplitude [au]")
    ax4.legend(loc="lower left")
    ax4.set_title("Amplitude Spectra")

    fig2.tight_layout()
    plt.show()
