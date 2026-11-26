"""
Simulating B-mode Images Using A Phased Array Example

Ported from: k-Wave/examples/example_us_bmode_phased_array.m

Illustrates how k-Wave can be used to simulate B-mode ultrasound images
(including tissue harmonic imaging) using a phased array transducer.  Unlike
the linear transducer example, the entire phantom fits inside the
computational grid: for each scan line the beam is electronically steered
to a different angle (-32 to 32 degrees), and the received signals are
beamformed with the same steering.  The scan lines are then processed and
scan-converted to form fundamental and harmonic images.

kspaceFirstOrder() does not take a transducer object as its source or
sensor, so:

* the transducer is converted to an equivalent velocity source (see the
  Defining An Ultrasound Transducer example), rebuilt for each steering
  angle;
* the sensor mask is the active-element mask, and the recorded time series
  are combined per element (with the elevation focusing delays) before
  being beamformed with NotATransducer.scan_line().

The full simulation runs 33 3D simulations of 226 x 236 x 108 grid points
and is slow on a CPU; use device="gpu" (CuPy) or backend="cpp" if available,
or pass fewer steering angles for a quick test.
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
from kwave.reconstruction.beamform import envelope_detection, scan_conversion
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.filters import gaussian_filter
from kwave.utils.mapgen import make_ball
from kwave.utils.signals import get_win, tone_burst

# size of the PML in grid points (added outside the grid)
PML_size = (15, 10, 10)

# medium parameters
c0 = 1540  # sound speed [m/s]
rho0 = 1000  # density [kg/m^3]
alpha_coeff = 0.75  # [dB/(MHz^y cm)]
alpha_power = 1.5
BonA = 6

# input signal properties
source_strength = 1e6  # [Pa]
tone_burst_freq = 1e6  # [Hz]
tone_burst_cycles = 4

# transducer properties
number_elements = 64  # total number of transducer elements
element_width = 1  # width of each element [grid points]
element_length = 40  # length of each element [grid points]
element_spacing = 0  # spacing (kerf width) between the elements [grid points]

# range of steering angles to test [deg]
steering_angles = np.arange(-32, 33, 2)


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
    """Create a random scattering medium containing a highly scattering sphere.

    Returns:
        tuple: (sound_speed_map, density_map), each (Nx, Ny, Nz).
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = (int(n) for n in kgrid.N)

    # define a random distribution of scatterers for the medium
    background_map = 1 + 0.008 * rng.standard_normal((Nx, Ny, Nz))
    sound_speed_map = c0 * background_map
    density_map = rho0 * background_map

    # define a random distribution of scatterers for the highly scattering region
    scattering_c0 = np.clip(c0 + 25 + 75 * rng.standard_normal((Nx, Ny, Nz)), 1400, 1600)
    scattering_rho0 = scattering_c0 / 1.5

    # define a sphere for a highly scattering region
    radius = 8e-3  # [m]
    x_pos = 32e-3  # [m]
    y_pos = kgrid.dy * Ny / 2  # [m]
    z_pos = kgrid.dz * Nz / 2  # [m]
    ball_center = np.round(Vector([x_pos, y_pos, z_pos]) / kgrid.dx)
    ball = make_ball(Vector([Nx, Ny, Nz]), ball_center, round(radius / kgrid.dx)).astype(bool)

    sound_speed_map[ball] = scattering_c0[ball]
    density_map[ball] = scattering_rho0[ball]

    return sound_speed_map, density_map


# %%
def setup(seed=0):
    """Set up the simulation physics (grid, medium, transducer).

    Returns:
        tuple: (kgrid, medium, not_transducer)
    """
    # create the computational grid
    Nx = 256 - 2 * PML_size[0]  # [grid points]
    Ny = 256 - 2 * PML_size[1]  # [grid points]
    Nz = 128 - 2 * PML_size[2]  # [grid points]
    dx = 50e-3 / Nx  # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # create the time array
    t_end = (Nx * dx) * 2.2 / c0  # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # define the properties of the propagation medium
    sound_speed_map, density_map = make_phantom(kgrid, seed=seed)
    medium = kWaveMedium(
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        BonA=BonA,
    )

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

    # steering_angle_max fixes the number of zeros appended to the input
    # signal, so all scan lines share the same time offset
    not_transducer = NotATransducer(
        transducer,
        kgrid,
        sound_speed=c0,  # sound speed [m/s]
        focus_distance=30e-3,  # focus distance [m]
        elevation_focus_distance=30e-3,  # focus distance in the elevation plane [m]
        steering_angle=0,  # steering angle [degrees]
        steering_angle_max=32,  # maximum steering angle [degrees]
        transmit_apodization="Hanning",
        receive_apodization="Rectangular",
        active_elements=np.ones((number_elements, 1)),
        input_signal=input_signal,
    )

    return kgrid, medium, not_transducer


# %%
def run(backend="python", device="cpu", quiet=True, angles=steering_angles, seed=0):
    """Simulate one scan line per steering angle.

    Args:
        angles: steering angles [deg] (each within +/- 32 degrees).
        seed: random seed for the phantom.

    Returns:
        dict with keys:
            'scan_lines': (len(angles), Nt) beamformed scan lines
            'sound_speed_map': (Nx, Ny, Nz) phantom sound speed [m/s]
    """
    kgrid, medium, not_transducer = setup(seed=seed)

    # the transducer is used as the sensor
    sensor = kSensor(mask=not_transducer.active_elements_mask != 0)
    sensor.record = ["p"]

    scan_lines = np.zeros((len(angles), int(kgrid.Nt)))
    for angle_index, angle in enumerate(angles):
        if not quiet:
            print(f"Computing scan line {angle_index + 1} of {len(angles)} (steering angle {angle} deg)")

        # update the current steering angle and the corresponding source
        not_transducer.steering_angle = angle
        source = transducer_source(not_transducer)

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
        scan_lines[angle_index, :] = not_transducer.scan_line(element_data)

    return {"scan_lines": scan_lines, "sound_speed_map": np.asarray(medium.sound_speed)}


# %%
def process_scan_lines(scan_lines, kgrid, not_transducer):
    """Process the beamformed scan lines into fundamental and harmonic scan lines.

    Returns:
        dict with keys 'raw', 'fund' and 'harm' (all (num_lines, Nt - t0_offset)).
    """
    # trim the delay offset from the scan line data
    input_signal_length = np.asarray(not_transducer.stored_input_signal).size
    t0_offset = int(round(input_signal_length / 2) + (not_transducer.appended_zeros - not_transducer.beamforming_delays_offset))
    scan_lines = scan_lines[:, t0_offset:].copy()
    Nt = scan_lines.shape[1]

    # window the first part of each scan line to remove the input signal
    tukey_win, _ = get_win(Nt * 2, "Tukey", False, 0.05)
    tukey_win = np.squeeze(tukey_win)
    scan_line_win = np.concatenate((np.zeros(t0_offset * 2), tukey_win[: Nt - t0_offset * 2]))
    scan_lines *= scan_line_win
    raw = scan_lines.copy()

    # time gain compensation
    r = c0 * np.arange(1, Nt + 1) * kgrid.dt / 2  # [m]
    tgc_alpha_db_cm = alpha_coeff * (tone_burst_freq * 1e-6) ** alpha_power
    tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100
    scan_lines *= np.exp(tgc_alpha_np_m * 2 * r)

    # frequency filtering at the fundamental and second harmonic
    scan_lines_fund = gaussian_filter(scan_lines, 1 / kgrid.dt, tone_burst_freq, 100)
    scan_lines_harm = gaussian_filter(scan_lines, 1 / kgrid.dt, 2 * tone_burst_freq, 30)

    # envelope detection
    scan_lines_fund = envelope_detection(scan_lines_fund)
    scan_lines_harm = envelope_detection(scan_lines_harm)

    # log compression
    compression_ratio = 3
    scan_lines_fund = log_compression(scan_lines_fund, compression_ratio, True)
    scan_lines_harm = log_compression(scan_lines_harm, compression_ratio, True)

    return {"raw": raw, "fund": scan_lines_fund, "harm": scan_lines_harm}


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    kgrid, _, not_transducer = setup()
    lines = process_scan_lines(result["scan_lines"], kgrid, not_transducer)

    # scan convert the processed scan lines
    image_size = [kgrid.Nx * kgrid.dx, kgrid.Ny * kgrid.dy]
    image_res = [256, 256]
    b_mode_fund = scan_conversion(lines["fund"], steering_angles, image_size, c0, kgrid.dt, image_res)
    b_mode_harm = scan_conversion(lines["harm"], steering_angles, image_size, c0, kgrid.dt, image_res)

    # create the axis variables
    x_axis = [0, image_size[0] * 1e3]  # [mm]
    y_axis = [0, image_size[1] * 1e3]  # [mm]
    line_extent = [steering_angles[0], steering_angles[-1], x_axis[1], x_axis[0]]
    image_extent = [y_axis[0], y_axis[1], x_axis[1], x_axis[0]]

    # plot the scan lines before and after processing, and the B-mode image
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(lines["raw"].T, aspect="auto", extent=line_extent, interpolation="none", cmap="gray")
    axes[0].set(xlabel="Steering angle [deg]", ylabel="Depth [mm]", title="Raw Scan-Line Data")
    axes[1].imshow(lines["fund"].T, aspect="auto", extent=line_extent, interpolation="none", cmap="bone")
    axes[1].set(xlabel="Steering angle [deg]", ylabel="Depth [mm]", title="Processed Scan-Line Data")
    axes[2].imshow(b_mode_fund, cmap="bone", aspect="auto", extent=image_extent, interpolation="none")
    axes[2].set(xlabel="Horizontal Position [mm]", ylabel="Depth [mm]", title="B-Mode Image")
    fig.tight_layout()

    # plot the phantom and the fundamental and harmonic images
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(result["sound_speed_map"][:, :, int(kgrid.Nz) // 2], aspect="auto", extent=image_extent)
    axes[0].set(xlabel="Horizontal Position [mm]", ylabel="Depth [mm]", title="Scattering Phantom")
    axes[1].imshow(b_mode_fund, cmap="bone", aspect="auto", extent=image_extent, interpolation="none")
    axes[1].set(xlabel="Horizontal Position [mm]", ylabel="Depth [mm]", title="B-Mode Image")
    axes[2].imshow(b_mode_harm, cmap="bone", aspect="auto", extent=image_extent, interpolation="none")
    axes[2].set(xlabel="Horizontal Position [mm]", ylabel="Depth [mm]", title="Harmonic Image")
    fig.tight_layout()

    plt.show()
