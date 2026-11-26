"""
Simulating Ultrasound Beam Patterns Example

Ported from: k-Wave/examples/example_us_beam_patterns.m

Shows how the nonlinear beam pattern from an ultrasound transducer can be
modelled.  It builds on the Defining An Ultrasound Transducer example.  A
32-element linear transducer focused at 20 mm transmits a tone burst into a
nonlinear, absorbing medium, and the pressure is recorded over a plane
through the centre of the transducer (x-y or x-z).

Two kinds of beam pattern are computed from a single simulation:

* the RMS and peak pressure over time ("statistics" beam patterns);
* the amplitude spectrum at the fundamental and second harmonic, and the
  integral of the amplitude spectrum ("harmonic" beam patterns), together
  with the directivity at the focal depth.

kspaceFirstOrder() does not take a transducer object as its source, so the
transducer is converted to an equivalent velocity source (see the Defining
An Ultrasound Transducer example).
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
from kwave.utils.filters import spect
from kwave.utils.math import find_closest
from kwave.utils.signals import tone_burst

# size of the PML in grid points (added outside the grid)
PML_size = (20, 10, 10)

# input signal properties
source_strength = 1e6  # [Pa]
tone_burst_freq = 0.5e6  # [Hz]
tone_burst_cycles = 5
focus_distance = 20e-3  # [m]


# %%
def transducer_source(transducer):
    """Convert a NotATransducer into an equivalent kSource velocity source.

    Each active grid point is driven by the (zero-padded) transducer input
    signal delayed by its beamforming delay and scaled by the transmit
    apodization, mirroring how k-Wave applies transducer sources.
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
        tuple: (kgrid, medium, source)
    """
    # create the computational grid
    Nx = 128 - 2 * PML_size[0]  # [grid points]
    Ny = 64 - 2 * PML_size[1]  # [grid points]
    Nz = 64 - 2 * PML_size[2]  # [grid points]
    dx = 40e-3 / Nx  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    # create the time array
    kgrid.makeTime(medium.sound_speed, t_end=45e-6)

    # define the input signal (converted from pressure to particle velocity)
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
    input_signal = (source_strength / (medium.sound_speed * medium.density)) * input_signal

    # physical properties of the transducer
    number_elements = 32  # total number of transducer elements
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
        radius=np.inf,
        position=np.round([1, Ny / 2 - transducer_width / 2, Nz / 2 - element_length / 2]),
    )

    not_transducer = NotATransducer(
        transducer,
        kgrid,
        sound_speed=float(medium.sound_speed),  # sound speed [m/s]
        focus_distance=focus_distance,  # focus distance [m]
        elevation_focus_distance=19e-3,  # focus distance in the elevation plane [m]
        steering_angle=0,  # steering angle [degrees]
        transmit_apodization="Rectangular",
        receive_apodization="Rectangular",
        active_elements=np.ones((number_elements, 1)),
        input_signal=input_signal,
    )

    return kgrid, medium, transducer_source(not_transducer)


# %%
def run(backend="python", device="cpu", quiet=True, mask_plane="xy"):
    """Run the simulation recording the pressure over a plane.

    Args:
        mask_plane: 'xy' or 'xz', the plane in which to compute the beam pattern.

    Returns:
        dict with keys (all beam patterns are (Nx, Nj), where Nj = Ny or Nz):
            'p_rms', 'p_max': statistics beam patterns [Pa]
            'beam_pattern_f1', 'beam_pattern_f2': amplitude at the fundamental
                and second harmonic
            'beam_pattern_total': integral of the amplitude spectrum
            'directivity_f1', 'directivity_f2': normalised directivity at the focus
    """
    kgrid, medium, source = setup()
    Nx, Ny, Nz = (int(n) for n in kgrid.N)

    # define the sensor plane
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    if mask_plane == "xy":
        sensor_mask[:, :, Nz // 2] = True
        Nj, dj = Ny, kgrid.dy
    elif mask_plane == "xz":
        sensor_mask[:, Ny // 2, :] = True
        Nj, dj = Nz, kgrid.dz
    else:
        raise ValueError(f"mask_plane must be 'xy' or 'xz', got {mask_plane!r}")
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p", "p_rms", "p_max"]

    result = kspaceFirstOrder(
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

    # sensor points are in C-order, i.e. (x, j) with j varying fastest
    p = np.asarray(result["p"]).reshape(Nx, Nj, -1)
    p_rms = np.asarray(result["p_rms"]).reshape(Nx, Nj)
    p_max = np.asarray(result["p_max"]).reshape(Nx, Nj)

    # compute the amplitude spectrum
    freq, amp_spect, _ = spect(p, 1 / kgrid.dt, dim=2)

    # compute the index at which the source frequency and its harmonics occur
    _, f1_index = find_closest(freq, tone_burst_freq)
    _, f2_index = find_closest(freq, 2 * tone_burst_freq)

    # extract the amplitude at the source frequency and second harmonic, and
    # the integral of the total amplitude spectrum
    beam_pattern_f1 = amp_spect[:, :, f1_index]
    beam_pattern_f2 = amp_spect[:, :, f2_index]
    beam_pattern_total = np.sum(amp_spect, axis=2)

    # compute the directivity at each of the harmonics at the focal depth
    focus_index = round(focus_distance / kgrid.dx)
    directivity_f1 = beam_pattern_f1[focus_index, :] / np.max(beam_pattern_f1[focus_index, :])
    directivity_f2 = beam_pattern_f2[focus_index, :] / np.max(beam_pattern_f2[focus_index, :])

    # compute relative angles from the transducer
    horz_axis = (np.arange(1, Nj + 1) - Nj / 2) * dj
    angles = 180 * np.arctan2(horz_axis, focus_distance) / np.pi

    return {
        "p_rms": p_rms,
        "p_max": p_max,
        "beam_pattern_f1": beam_pattern_f1,
        "beam_pattern_f2": beam_pattern_f2,
        "beam_pattern_total": beam_pattern_total,
        "directivity_f1": directivity_f1,
        "directivity_f2": directivity_f2,
        "angles": angles,
    }


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mask_plane = "xy"  # set to 'xy' or 'xz' to generate the beam pattern in different planes
    result = run(quiet=False, mask_plane=mask_plane)
    kgrid, _, _ = setup()

    j_vec = kgrid.y_vec if mask_plane == "xy" else kgrid.z_vec
    j_label = mask_plane[1]
    extent = [
        j_vec.min() * 1e3,
        j_vec.max() * 1e3,
        (kgrid.x_vec.max() - kgrid.x_vec.min()) * 1e3,
        0,
    ]

    # plot the statistics beam patterns
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, key, name in zip(axes, ["p_rms", "p_max"], ["RMS", "MAX"]):
        im = ax.imshow(result[key] * 1e-6, extent=extent, aspect="auto", cmap="jet")
        ax.set_xlabel(f"{j_label}-position [mm]")
        ax.set_ylabel("x-position [mm]")
        ax.set_title(f"Total Beam Pattern Using {name}\nOf Recorded Pressure")
        fig.colorbar(im, ax=ax, label="Pressure [MPa]")
    fig.tight_layout()

    # plot the harmonic beam patterns
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    titles = [
        "Beam Pattern\nAt Source Fundamental",
        "Beam Pattern\nAt Second Harmonic",
        "Total Beam Pattern\nUsing Integral Of Recorded Pressure",
    ]
    for ax, key, title in zip(axes, ["beam_pattern_f1", "beam_pattern_f2", "beam_pattern_total"], titles):
        im = ax.imshow(result[key] * 1e-6, extent=extent, aspect="auto", cmap="jet")
        ax.set_xlabel(f"{j_label}-position [mm]")
        ax.set_ylabel("x-position [mm]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()

    # plot the directivity at the focal depth
    fig, ax = plt.subplots()
    ax.plot(result["angles"], result["directivity_f1"], "k-", label="Fundamental")
    ax.plot(result["angles"], result["directivity_f2"], "k--", label="Second Harmonic")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Normalised Amplitude")
    ax.set_xlim(-25, 25)
    ax.legend(loc=2)

    plt.show()
