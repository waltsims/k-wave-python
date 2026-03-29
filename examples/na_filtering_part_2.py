"""
Filtering A Delta Function Input Signal Example Part 2

Ported from: k-Wave/examples/example_na_filtering_part_2.m

Illustrates how numerical aliasing can be avoided by spatially smoothing the
source mask. Instead of a single-point (binary) source, the source mask is
smoothed with a Blackman window (via ``smooth(mask, restore_max=True)``),
thresholded to discard small values, and then the source time series is scaled
per-element to reproduce the original amplitude distribution. This spreads
the spatial spectrum of the source below the grid Nyquist limit.

The default CFL case is ported: dt = 7 ns fixed, Nt = 1024 steps.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import smooth


def setup():
    """Set up the simulation physics (grid, medium, source).

    Grid: 256 points, dx = 10e-3 / 256 m.
    Medium: homogeneous, c = 1500 m/s.
    Source: spatially smoothed single-element mask at grid index 50 (0-based),
            delta-function pulse of magnitude 2 Pa at time step 99 (0-based),
            scaled per source element.

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
    p_mask = np.zeros((Nx, 1), dtype=float)  # column vector for smooth()
    p_mask[source_offset, 0] = 1

    # spatially smooth the source mask maintaining the maximum magnitude
    # MATLAB: source.p_mask = smooth(source.p_mask, true)
    p_mask = smooth(p_mask, restore_max=True)

    # threshold out small values
    # MATLAB: source.p_mask(source.p_mask < 0.05) = 0
    p_mask[p_mask < 0.05] = 0

    # flatten to 1D for the solver
    p_mask = p_mask.ravel()

    # define a delta function input pulse
    # MATLAB: source_func(temporal_offset) = source_magnitude
    temporal_offset = 100  # [time steps, 1-based in MATLAB]
    source_magnitude = 2  # [Pa]
    source_func = np.zeros(Nt, dtype=float)
    source_func[temporal_offset - 1] = source_magnitude

    # assign and scale the input pulse based on the source mask
    # MATLAB: source.p(1:sum(source.p_mask ~= 0), :) = source.p_mask(source.p_mask ~= 0) * source_func
    nonzero_vals = p_mask[p_mask != 0]
    num_sources = len(nonzero_vals)
    # each row = weight * source_func
    source_p = nonzero_vals[:, np.newaxis] * source_func[np.newaxis, :]

    source = kSource()
    source.p = source_p  # (num_sources, Nt)

    # force the source mask to be binary
    # MATLAB: source.p_mask(source.p_mask ~= 0) = 1
    p_mask[p_mask != 0] = 1
    source.p_mask = p_mask

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

    # find the sensor point closest to the original single-element sensor
    # MATLAB: sensor.mask(end - source_offset) = 1  ->  index 206 (1-based)  ->  row 205 (0-based)
    sensor_row = 205

    # plot recorded time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # extract the middle source element's signal for display
    source_p = np.asarray(source.p)
    mid_idx = source_p.shape[0] // 2
    ax1.plot(t_us, source_p[mid_idx, :], "k-", label="Input pulse (middle element)")
    ax1.plot(t_us, p[sensor_row, :], "b-", label="Recorded pulse")
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [au]")
    ax1.legend()
    ax1.set_title("Input & Recorded Time Series (Part 2: Smoothed Source Mask)")

    # compute and plot amplitude spectrum of recorded signal
    dt = 7e-9
    N = p.shape[1]
    f = np.fft.rfftfreq(N, d=dt)
    output_as = np.abs(np.fft.rfft(p[sensor_row, :])) / N

    f_max = float(np.max(kgrid.k_max) * np.min(medium.sound_speed) / (2 * np.pi))
    f_MHz = f * 1e-6

    ax2.plot(f_MHz, output_as, "b-", label="Recorded spectrum")
    ax2.axvline(f_max * 1e-6, color="k", linestyle="--", label="Max grid frequency")
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Amplitude [au]")
    ax2.legend()
    ax2.set_title("Amplitude Spectrum of Recorded Signal")

    fig.tight_layout()
    plt.show()
