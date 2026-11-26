"""
Defining An Ultrasound Transducer Example

Ported from: k-Wave/examples/example_us_defining_transducer.m

Illustrates how kWaveTransducerSimple and NotATransducer can be used to
define a linear ultrasound transducer with electronic focusing in azimuth
and elevation.  The field is recorded at three points along the beam axis
in a nonlinear, absorbing medium, and the amplitude spectra of the recorded
signals are compared with the spectrum of the input signal.

kspaceFirstOrder() does not take a transducer object as its source, so the
transducer is converted to an equivalent velocity source: the source mask is
the active-element mask, and each source point is driven by the transducer
input signal shifted by that point's beamforming delay (azimuth + elevation)
and scaled by the transmit apodization.  As in k-Wave, the transducer signal
is added to ux without the k-space source correction.
"""

# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.utils.signals import tone_burst

# size of the PML in grid points (added outside the grid)
PML_size = (20, 10, 10)

# input signal properties
source_strength = 1e6  # [Pa]
tone_burst_freq = 0.5e6  # [Hz]
tone_burst_cycles = 5


# %%
def transducer_source(transducer):
    """Convert a NotATransducer into an equivalent kSource velocity source.

    Each active grid point is driven by the (zero-padded) transducer input
    signal delayed by its beamforming delay and scaled by the transmit
    apodization, mirroring how k-Wave applies transducer sources.

    Returns:
        kSource: source with u_mask, ux (one signal per source point, C-order)
        and u_mode="additive-no-correction".
    """
    mask = transducer.active_elements_mask != 0
    signal = np.asarray(transducer.input_signal).squeeze()
    delays = np.asarray(transducer.delay_mask())[mask].astype(int)

    if transducer.transmit_apodization == "Rectangular":
        apodization = np.ones(delays.size)
    else:
        apodization = np.asarray(transducer.transmit_apodization_mask)[mask]

    # source point i at time step t uses signal[delay_i + t]
    num_time_points = signal.size - delays.max()
    ux = signal[delays[:, None] + np.arange(num_time_points)[None, :]] * apodization[:, None]

    source = kSource()
    source.u_mask = mask
    source.ux = ux
    source.u_mode = "additive-no-correction"
    return source


# %%
def setup():
    """Set up the simulation physics (grid, medium, transducer source).

    Returns:
        tuple: (kgrid, medium, source, not_transducer)
    """
    # create the computational grid
    Nx = 128 - 2 * PML_size[0]  # [grid points]
    Ny = 128 - 2 * PML_size[1]  # [grid points]
    Nz = 64 - 2 * PML_size[2]  # [grid points]
    dx = 40e-3 / Nx  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    # create the time array
    kgrid.makeTime(medium.sound_speed, t_end=40e-6)

    # define the input signal (converted from pressure to particle velocity)
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
    input_signal = (source_strength / (medium.sound_speed * medium.density)) * input_signal

    # physical properties of the transducer
    number_elements = 72  # total number of transducer elements
    element_width = 1  # width of each element [grid points]
    element_length = 12  # length of each element [grid points]
    element_spacing = 0  # spacing (kerf width) between the elements [grid points]
    transducer_width = number_elements * element_width + (number_elements - 1) * element_spacing

    # position the transducer in the middle of the computational grid
    transducer = kWaveTransducerSimple(
        kgrid,
        number_elements=number_elements,
        element_width=element_width,
        element_length=element_length,
        element_spacing=element_spacing,
        radius=np.inf,  # radius of curvature of the transducer [m]
        position=np.round([1, Ny / 2 - transducer_width / 2, Nz / 2 - element_length / 2]),
    )

    # properties used to derive the beamforming delays
    active_elements = np.zeros((number_elements, 1))
    active_elements[21:52] = 1
    not_transducer = NotATransducer(
        transducer,
        kgrid,
        sound_speed=float(medium.sound_speed),  # sound speed [m/s]
        focus_distance=20e-3,  # focus distance [m]
        elevation_focus_distance=19e-3,  # focus distance in the elevation plane [m]
        steering_angle=0,  # steering angle [degrees]
        transmit_apodization="Rectangular",
        receive_apodization="Rectangular",
        active_elements=active_elements,
        input_signal=input_signal,
    )

    source = transducer_source(not_transducer)

    return kgrid, medium, source, not_transducer


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the simulation recording the pressure at three on-axis points.

    Returns:
        dict: Simulation results with key 'p' (3 x Nt).
    """
    kgrid, medium, source, _ = setup()
    Nx, Ny, Nz = (int(n) for n in kgrid.N)

    # define a sensor mask through the central plane of the transducer
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[Nx // 4, Ny // 2, Nz // 2] = True
    sensor_mask[Nx // 2, Ny // 2, Nz // 2] = True
    sensor_mask[3 * Nx // 4, Ny // 2, Nz // 2] = True
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]

    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=False,
        pml_size=PML_size,
    )


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from kwave.utils.filters import spect
    from kwave.utils.plot import voxel_plot

    kgrid, medium, source, not_transducer = setup()

    # plot the transducer elements
    voxel_plot(np.single(not_transducer.all_elements_mask))

    result = run(quiet=False)
    p = np.asarray(result["p"])
    fs = 1 / kgrid.dt
    c0 = float(medium.sound_speed)
    f_max = c0 / (2 * kgrid.dx)

    # compute the amplitude spectra of the input signal and the sensor data
    input_signal = np.asarray(not_transducer.stored_input_signal).squeeze()
    padded_input_signal = np.concatenate((input_signal, np.zeros(2 * input_signal.size)))
    f_input, as_input, _ = spect(padded_input_signal, fs)
    f, as_sensor, _ = spect(p, fs, dim=1)

    # plot the input signal and its frequency spectrum
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(np.arange(input_signal.size) * kgrid.dt * 1e6, input_signal, "k-")
    axes[0].set_xlabel(r"Time [$\mu$s]")
    axes[0].set_ylabel("Particle Velocity [m/s]")
    axes[1].plot(f_input / 1e6, np.squeeze(as_input) / np.max(as_input), "k-")
    axes[1].plot([tone_burst_freq / 1e6, tone_burst_freq / 1e6], [0, 1], "k--")
    axes[1].set_xlabel("Frequency [MHz]")
    axes[1].set_ylabel("Amplitude Spectrum [au]")
    axes[1].set_xlim(0, f_max / 1e6)
    fig.tight_layout()

    # plot the recorded time series and their spectra
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t_us = np.asarray(kgrid.t_array).squeeze() * 1e6
    offset = -3e6
    labels = [f"Sensor Position {i + 1}" for i in range(p.shape[0])]
    for i in range(p.shape[0]):
        axes[0].plot(t_us, p[i, :] + offset * i, "k-")
    axes[0].set_yticks([offset * i for i in range(p.shape[0])], labels)
    axes[0].set_xlabel(r"Time [$\mu$s]")
    axes[0].set_title("Recorded Pressure")

    for i, style in enumerate(["k-", "b-", "r-"]):
        axes[1].plot(f / 1e6, as_sensor[i, :] / np.max(as_sensor[0, :]), style, label=labels[i])
    axes[1].legend()
    axes[1].set_xlabel("Frequency [MHz]")
    axes[1].set_ylabel("Normalised Amplitude Spectrum [au]")
    axes[1].set_xlim(0, f_max / 1e6)
    fig.tight_layout()

    plt.show()
