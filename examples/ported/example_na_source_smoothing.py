"""
Source Smoothing Example

Ported from: k-Wave/examples/example_na_source_smoothing.m

Illustrates how spatial smoothing can reduce discrete sampling artifacts.
The original MATLAB example uses kspaceSecondOrder; this port uses
kspaceFirstOrder (the only solver available in the Python backend) with
equivalent 1D parameters.

Three cases are run: no window, Hanning, and Blackman. Each applies a
frequency-domain window to a delta-function initial pressure before
propagating.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import get_win


def setup():
    """Set up the simulation physics (grid, medium, source) with no window.

    Returns:
        tuple: (kgrid, medium, source)
    """

    # create the computational grid
    Nx = 256
    dx = 0.05e-3  # [m]
    kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # define time array (matching MATLAB: dt=2e-9, t_end=4.26e-6)
    dt = 2e-9  # [s]
    t_end = 4.26e-6  # [s]
    Nt = np.round(t_end / dt).astype(int) + 1
    kgrid.setTime(Nt, dt)

    # create delta function source at centre
    source_pos = Nx // 2  # 0-based: 128
    source = kSource()
    source.p0 = np.zeros(Nx)
    source.p0[source_pos] = 1.0

    return kgrid, medium, source


def _apply_window(p0, Nx, window_type):
    """Apply a frequency-domain window to the initial pressure."""
    win, cg = get_win(Nx, window_type, rotation=True, symmetric=True)
    win = win.flatten()  # get_win returns (N,1) for 1D; flatten for broadcasting
    return np.real(np.fft.ifft(np.fft.fftshift(np.fft.fftshift(np.fft.fft(p0)) * win))) / cg


def run(backend="python", device="cpu", quiet=True):
    """Run three simulations (no window, Hanning, Blackman).

    Returns:
        dict: {'no_window': result, 'hanning': result, 'blackman': result}
              Each result has keys 'p' and 'p_final'.
    """
    results = {}

    for label, window_type in [("no_window", None), ("hanning", "Hanning"), ("blackman", "Blackman")]:
        kgrid, medium, source = setup()
        Nx = 256
        source_pos = Nx // 2

        if window_type is not None:
            source.p0 = _apply_window(source.p0, Nx, window_type)

        # sensor at a small distance from source
        source_sensor_dist = 2e-6  # [m]
        sensor_pos = source_pos + np.round(medium.sound_speed * source_sensor_dist / kgrid.dx).astype(int)
        sensor_mask = np.zeros(Nx, dtype=bool)
        sensor_mask[sensor_pos] = True
        sensor = kSensor(mask=sensor_mask)
        sensor.record = ["p", "p_final"]

        results[label] = kspaceFirstOrder(
            kgrid,
            medium,
            source,
            sensor,
            backend=backend,
            device=device,
            quiet=quiet,
            pml_inside=True,
            smooth_p0=False,
        )

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    results = run(quiet=False)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for i, (label, title) in enumerate(
        [
            ("no_window", "No Window"),
            ("hanning", "Hanning"),
            ("blackman", "Blackman"),
        ]
    ):
        r = results[label]
        p = np.asarray(r["p"]).flatten()

        # plot the time signal
        ax = axes[i, 0]
        ax.plot(p, "k-")
        ax.set_ylabel("Amplitude [au]")
        ax.set_xlabel("Time Step")
        ax.set_title(f"{title}: Recorded Time Pulse")

        # plot the amplitude spectrum
        ax = axes[i, 1]
        spectrum = np.abs(np.fft.rfft(p))
        spectrum[0] = 2 * spectrum[0]
        freqs = np.fft.rfftfreq(len(p))
        ax.plot(freqs, spectrum / spectrum.max(), "k-")
        ax.set_xlabel("Normalised Frequency")
        ax.set_ylabel("Relative Amplitude")
        ax.set_title(f"{title}: Frequency Response")

    fig.suptitle("Source Smoothing Example")
    fig.tight_layout()
    plt.show()
