"""
Simulating B-mode Ultrasound Images Example

Ported from: k-Wave/examples/example_us_bmode_linear_transducer.m

Illustrates how k-Wave can be used to simulate B-mode ultrasound images
(including tissue harmonic imaging) analogous to those produced by a modern
diagnostic ultrasound scanner.  A 32-element linear transducer is used both
as the source and the receiver.  The phantom is larger than the computational
grid: for each of the 96 scan lines the medium is shifted by one element
width under the transducer, a focused pulse is transmitted, and the received
signals are beamformed into a single scan line.  The scan lines are then
processed (input-signal removal, time gain compensation, frequency filtering
at the fundamental and second harmonic, envelope detection and log
compression) to form fundamental and harmonic images.

kspaceFirstOrder() does not take a transducer object as its source or
sensor, so:

* the transducer is converted to an equivalent velocity source (see the
  Defining An Ultrasound Transducer example);
* the sensor mask is the active-element mask, and the recorded time series
  are combined per element (with the elevation focusing delays) before
  being beamformed with NotATransducer.scan_line().

The full simulation runs 96 3D simulations of 216 x 108 x 108 grid points
and is slow on a CPU; use device="gpu" (CuPy) or backend="cpp" if available,
or reduce number_scan_lines for a quick test.
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
from kwave.reconstruction.beamform import envelope_detection
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.filters import gaussian_filter
from kwave.utils.mapgen import make_ball
from kwave.utils.signals import get_win, tone_burst

# size of the PML in grid points (added outside the grid)
PML_size = (20, 10, 10)

# medium parameters
c0 = 1540  # sound speed [m/s]
rho0 = 1000  # density [kg/m^3]
alpha_coeff = 0.75  # [dB/(MHz^y cm)]
alpha_power = 1.5
BonA = 6

# input signal properties
source_strength = 1e6  # [Pa]
tone_burst_freq = 1.5e6  # [Hz]
tone_burst_cycles = 4

# transducer properties
number_elements = 32  # total number of transducer elements
element_width = 2  # width of each element [grid points]
element_length = 24  # length of each element [grid points]
element_spacing = 0  # spacing (kerf width) between the elements [grid points]

# number of scan lines in the image
number_scan_lines = 96


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
def combine_transducer_data(transducer, sensor_data):
    """Combine the time series recorded over the active-element mask per element.

    The time series of each grid point are shifted by the elevation focusing
    delay and averaged over the grid points of each element, as in
    kWaveTransducer.combine_sensor_data.

    Args:
        sensor_data: (num_active_grid_points, Nt) recorded pressure, rows in
            C-order of the active-element mask.

    Returns:
        (number_active_elements, Nt) element signals.
    """
    mask = transducer.active_elements_mask != 0
    element_index = np.asarray(transducer.indexed_active_elements_mask)[mask].astype(int) - 1
    elevation_delay = np.asarray(transducer.delay_mask(2))[mask].astype(int)

    Nt = sensor_data.shape[1]
    element_data = np.zeros((transducer.number_active_elements, Nt))
    for row, (element, delay) in enumerate(zip(element_index, elevation_delay)):
        element_data[element, : Nt - delay] += sensor_data[row, delay:]

    return element_data / (transducer.transducer.element_width * transducer.transducer.element_length)


# %%
def make_phantom(kgrid, seed=0):
    """Create a scattering phantom wider than the grid by one element width per scan line.

    Returns:
        tuple: (sound_speed_map, density_map), each (Nx, Ny_tot, Nz).
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = (int(n) for n in kgrid.N)
    dx = kgrid.dx

    # define a large image size to move across
    Ny_tot = Ny + number_scan_lines * element_width
    shape = (Nx, Ny_tot, Nz)

    # define a random distribution of scatterers for the medium
    background_map = 1 + 0.008 * rng.standard_normal(shape)

    # define a random distribution of scatterers for the highly scattering regions
    scattering_c0 = np.clip(c0 + 25 + 75 * rng.standard_normal(shape), 1400, 1600)
    scattering_rho0 = scattering_c0 / 1.5

    sound_speed_map = c0 * background_map
    density_map = rho0 * background_map

    # define spheres for highly scattering regions: (radius, x_pos, y_pos) [m]
    for radius, x_pos, y_pos in [(6e-3, 27.5e-3, 20.5e-3), (5e-3, 30.5e-3, 37e-3), (4.5e-3, 15.5e-3, 30.5e-3)]:
        ball = make_ball(
            Vector([Nx, Ny_tot, Nz]),
            Vector([round(x_pos / dx), round(y_pos / dx), Nz // 2]),
            round(radius / dx),
        ).astype(bool)
        sound_speed_map[ball] = scattering_c0[ball]
        density_map[ball] = scattering_rho0[ball]

    return sound_speed_map, density_map


# %%
def setup():
    """Set up the simulation physics (grid, medium, transducer).

    The medium sound speed and density are replaced by a section of the
    phantom for each scan line in run().

    Returns:
        tuple: (kgrid, medium, source, not_transducer)
    """
    # create the computational grid
    Nx = 256 - 2 * PML_size[0]  # [grid points]
    Ny = 128 - 2 * PML_size[1]  # [grid points]
    Nz = 128 - 2 * PML_size[2]  # [grid points]
    dx = 40e-3 / Nx  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # create the time array
    t_end = (Nx * dx) * 2.2 / c0  # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # define the properties of the propagation medium (homogeneous placeholder)
    medium = kWaveMedium(sound_speed=c0, density=rho0, alpha_coeff=alpha_coeff, alpha_power=alpha_power, BonA=BonA)

    # define the input signal (converted from pressure to particle velocity)
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
    input_signal = (source_strength / (c0 * rho0)) * input_signal

    # position the transducer in the middle of the computational grid
    transducer_width = number_elements * element_width + (number_elements - 1) * element_spacing
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
        sound_speed=c0,  # sound speed [m/s]
        focus_distance=20e-3,  # focus distance [m]
        elevation_focus_distance=19e-3,  # focus distance in the elevation plane [m]
        steering_angle=0,  # steering angle [degrees]
        transmit_apodization="Hanning",
        receive_apodization="Rectangular",
        active_elements=np.ones((number_elements, 1)),
        input_signal=input_signal,
    )

    return kgrid, medium, transducer_source(not_transducer), not_transducer


# %%
def run(backend="python", device="cpu", quiet=True, num_lines=number_scan_lines, seed=0):
    """Simulate the scan lines by moving the phantom under the transducer.

    Args:
        num_lines: number of scan lines to simulate (<= number_scan_lines).
        seed: random seed for the phantom.

    Returns:
        dict with keys:
            'scan_lines': (num_lines, Nt) beamformed scan lines
            'sound_speed_map': (Nx, Ny_tot, Nz) phantom sound speed [m/s]
    """
    kgrid, medium, source, not_transducer = setup()
    Ny = int(kgrid.Ny)
    sound_speed_map, density_map = make_phantom(kgrid, seed=seed)

    # the transducer is used as the sensor
    sensor = kSensor(mask=not_transducer.active_elements_mask != 0)
    sensor.record = ["p"]

    scan_lines = np.zeros((num_lines, int(kgrid.Nt)))
    medium_position = 0
    for scan_line_index in range(num_lines):
        if not quiet:
            print(f"Computing scan line {scan_line_index + 1} of {num_lines}")

        # load the current section of the medium
        medium.sound_speed = sound_speed_map[:, medium_position : medium_position + Ny, :]
        medium.density = density_map[:, medium_position : medium_position + Ny, :]

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

        # combine the data per element and beamform it into a scan line
        element_data = combine_transducer_data(not_transducer, np.asarray(result["p"]))
        scan_lines[scan_line_index, :] = not_transducer.scan_line(element_data)

        # update medium position
        medium_position += element_width

    return {"scan_lines": scan_lines, "sound_speed_map": sound_speed_map}


# %%
def process_scan_lines(scan_lines, kgrid, input_signal_length):
    """Form fundamental and harmonic images from the beamformed scan lines.

    Returns:
        dict with the processed images ('fund', 'harm') and the intermediate
        processing steps of the central scan line.
    """
    Nt = int(kgrid.Nt)
    scan_lines = scan_lines.copy()
    centre = scan_lines.shape[0] // 2
    steps = {}

    # remove the input signal: window the first part of each scan line to zero
    tukey_win, _ = get_win(Nt * 2, "Tukey", False, 0.05)
    scan_line_win = np.concatenate((np.zeros(input_signal_length * 2), np.squeeze(tukey_win)[: Nt - input_signal_length * 2]))
    scan_lines *= scan_line_win
    steps["1. Beamformed Signal"] = scan_lines[centre].copy()

    # time gain compensation (t0 corresponds to the middle of the input signal)
    t0 = input_signal_length * kgrid.dt / 2
    r = c0 * (np.arange(1, Nt + 1) * kgrid.dt - t0) / 2  # [m]
    tgc_alpha_db_cm = alpha_coeff * (tone_burst_freq * 1e-6) ** alpha_power
    tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100
    scan_lines *= np.exp(tgc_alpha_np_m * 2 * r)
    steps["2. Time Gain Compensation"] = scan_lines[centre].copy()

    # frequency filtering at the fundamental and second harmonic
    scan_lines_fund = gaussian_filter(scan_lines, 1 / kgrid.dt, tone_burst_freq, 100)
    scan_lines_harm = gaussian_filter(scan_lines, 1 / kgrid.dt, 2 * tone_burst_freq, 30)
    steps["3. Frequency Filtering"] = scan_lines_fund[centre].copy()

    # envelope detection
    scan_lines_fund = envelope_detection(scan_lines_fund)
    scan_lines_harm = envelope_detection(scan_lines_harm)
    steps["4. Envelope Detection"] = scan_lines_fund[centre].copy()

    # log compression
    compression_ratio = 3
    scan_lines_fund = log_compression(scan_lines_fund, compression_ratio, True)
    scan_lines_harm = log_compression(scan_lines_harm, compression_ratio, True)
    steps["5. Log Compression"] = scan_lines_fund[centre].copy()

    return {"fund": scan_lines_fund, "harm": scan_lines_harm, "steps": steps}


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    kgrid, _, _, not_transducer = setup()
    input_signal_length = np.asarray(not_transducer.stored_input_signal).size
    images = process_scan_lines(result["scan_lines"], kgrid, input_signal_length)

    Nx, Ny, Nz = (int(n) for n in kgrid.N)
    sound_speed_map = result["sound_speed_map"]

    # create the axis variables
    depth = Nx * kgrid.dx * 1.1 * 1e3  # [mm] (the scan lines record 2.2x the grid depth)
    width = number_scan_lines * element_width * kgrid.dy * 1e3  # [mm]
    extent = [-width / 2, width / 2, depth, 0]

    # plot the phantom and the fundamental and harmonic images
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(
        sound_speed_map[:, Ny // 2 : Ny // 2 + number_scan_lines * element_width, Nz // 2],
        aspect="auto",
        extent=[-width / 2, width / 2, Nx * kgrid.dx * 1e3, 0],
    )
    axes[0].set_title("Sound Speed")
    axes[0].set_ylabel("Depth [mm]")
    axes[1].imshow(images["fund"].T, cmap="gray", aspect="auto", extent=extent)
    axes[1].set_title("Fundamental")
    axes[2].imshow(images["harm"].T, cmap="gray", aspect="auto", extent=extent)
    axes[2].set_title("Harmonic")
    for ax in axes:
        ax.set_xlabel("Width [mm]")
        ax.set_ylim(40, 5)
    fig.tight_layout()

    # plot the processing steps for the central scan line
    fig, ax = plt.subplots(figsize=(14, 4))
    t_us = np.asarray(kgrid.t_array).squeeze() * 1e6
    offset = -6e5
    for i, (label, data) in enumerate(images["steps"].items()):
        ax.plot(t_us, data / np.max(np.abs(data)) * abs(offset) / 2 + offset * i, label=label)
    ax.set_yticks([offset * i for i in range(len(images["steps"]))], list(images["steps"].keys()))
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_title("Processing Steps (Normalised)")
    fig.tight_layout()

    plt.show()
