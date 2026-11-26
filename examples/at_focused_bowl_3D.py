"""
Modelling A Focused Bowl Transducer Example

Ported from: k-Wave/examples/example_at_piston_and_bowl_transducers.m
(section: focused bowl in 3D)

Models a focused bowl transducer in 3D, driven by a continuous-wave signal.
The bowl is defined as an off-grid element using kWaveArray.  The on-axis
amplitude is extracted from the steady-state pressure and compared with
O'Neil's analytical solution, computed with focused_bowl_oneil.

The PML is placed outside the user grid with an automatically chosen size.
"""

# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import extract_amp_phase
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import focused_bowl_oneil
from kwave.utils.math import round_even
from kwave.utils.signals import create_cw_signals

# medium parameters
c0 = 1500.0  # sound speed [m/s]
rho0 = 1000.0  # density [kg/m^3]

# source parameters
source_f0 = 1.0e6  # source frequency [Hz]
source_roc = 30e-3  # bowl radius of curvature [m]
source_diameter = 30e-3  # bowl aperture diameter [m]
source_amp = 1.0e6  # source pressure [Pa]
source_phase = 0.0  # source phase [radians]

# grid parameters
axial_size = 50.0e-3  # total grid size in the axial dimension [m]
lateral_size = 40.0e-3  # total grid size in the lateral dimension [m]

# computational parameters
ppw = 3  # number of points per wavelength
t_end = 40e-6  # total compute time [s] (this must be long enough to reach steady state)
record_periods = 1  # number of periods to record
cfl = 0.5  # CFL number
source_x_offset = 20  # grid points to offset the source
bli_tolerance = 0.01  # tolerance for truncation of the off-grid source points
upsampling_rate = 10  # density of integration points relative to grid

# points per temporal period
ppp = round(ppw / cfl)


# %%
def make_array(kgrid):
    """Define the bowl element (position depends on the grid).

    Returns:
        kWaveArray: array with a single bowl element.
    """
    # set bowl position and orientation
    bowl_pos = [kgrid.x_vec[0].item() + source_x_offset * kgrid.dx, 0.0, 0.0]
    focus_pos = [kgrid.x_vec[-1].item(), 0.0, 0.0]

    karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate, single_precision=True)
    karray.add_bowl_element(bowl_pos, source_roc, source_diameter, focus_pos)
    return karray


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """
    # calculate the grid spacing based on the PPW and F0
    dx = c0 / (ppw * source_f0)  # [m]

    # compute the size of the grid
    Nx = round_even(axial_size / dx) + source_x_offset
    Ny = round_even(lateral_size / dx)
    Nz = Ny
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # create the time array using an integer number of points per period
    dt = 1.0 / (ppp * source_f0)
    Nt = int(np.round(t_end / dt))
    kgrid.setTime(Nt, dt)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=c0, density=rho0)

    # create time varying continuous wave source
    source_sig = create_cw_signals(np.squeeze(kgrid.t_array), source_f0, np.array([source_amp]), np.array([source_phase]))

    karray = make_array(kgrid)
    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, source_sig, order="C")

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run the simulation, recording the central x-y plane in steady state.

    Returns:
        dict with keys:
            'amp': (Nx - source_x_offset - 2, Ny) steady-state pressure amplitude [Pa]
            'p': raw recorded pressure (num_sensor_points x num_recorded_time_points)
    """
    kgrid, medium, source = setup()
    Nx, Ny, Nz = (int(n) for n in kgrid.N)

    # record the central plane in front of the bowl
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[(source_x_offset + 1) : -1, :, Nz // 2] = True
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]

    # record only the final few periods when the field is in steady state
    num_recorded = record_periods * ppp
    sensor.record_start_index = int(kgrid.Nt) - num_recorded + 1

    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device=device,
        quiet=quiet,
        pml_inside=False,
        pml_size="auto",
    )

    # keep only the steady-state periods
    p = np.asarray(result["p"])[:, -num_recorded:]

    # extract the amplitude at the source frequency
    amp, _, _ = extract_amp_phase(p, 1.0 / kgrid.dt, source_f0, dim=1, fft_padding=1, window="Rectangular")
    amp = np.reshape(amp, (Nx - (source_x_offset + 2), Ny))

    return {"amp": amp, "p": p}


# %%
def analytical_on_axis(x):
    """O'Neil's solution for the on-axis pressure amplitude of the bowl."""
    velocity = source_amp / (c0 * rho0)
    p_axial, _, _ = focused_bowl_oneil(source_roc, source_diameter, velocity, source_f0, c0, rho0, axial_positions=x)
    return p_axial


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    amp = result["amp"]
    kgrid, _, source = setup()
    Nx, Ny, Nz = (int(n) for n in kgrid.N)

    # extract pressure on axis and define axis vectors for plotting
    amp_on_axis = amp[:, Ny // 2]
    x_vec = np.squeeze(kgrid.x_vec[(source_x_offset + 1) : -1] - kgrid.x_vec[source_x_offset])
    y_vec = np.squeeze(kgrid.y_vec)

    # analytical solution on a fine grid and at the k-Wave grid points
    x_ref = np.linspace(0.0, Nx * kgrid.dx, 10001)
    p_ref = analytical_on_axis(x_ref)
    p_ref_kw = analytical_on_axis(x_vec)
    L2_error = 100 * np.linalg.norm(p_ref_kw - amp_on_axis, ord=2) / np.linalg.norm(p_ref_kw, ord=2)
    Linf_error = 100 * np.max(np.abs(p_ref_kw - amp_on_axis)) / np.max(np.abs(p_ref_kw))
    print(f"L2 error = {L2_error:.2f} %, Linf error = {Linf_error:.2f} %")

    # plot the pressure along the focal axis of the bowl
    fig, ax = plt.subplots()
    ax.plot(1e3 * x_ref, 1e-6 * p_ref, "k-", label="Exact")
    ax.plot(1e3 * x_vec, 1e-6 * amp_on_axis, "b.", label="k-Wave")
    ax.legend()
    ax.set(xlabel="Axial Position [mm]", ylabel="Pressure [MPa]", title="Axial Pressure")
    ax.set_xlim(0.0, 1e3 * axial_size)
    ax.grid()

    # plot the source mask and the off-grid source weights (PML is outside the grid)
    grid_weights = make_array(kgrid).get_array_grid_weights(kgrid)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_a.pcolormesh(1e3 * y_vec, 1e3 * np.squeeze(kgrid.x_vec), source.p_mask[:, :, Nz // 2], shading="nearest")
    ax_a.set(xlabel="y [mm]", ylabel="x [mm]", title="Source Mask")
    ax_a.invert_yaxis()
    ax_b.pcolormesh(1e3 * y_vec, 1e3 * np.squeeze(kgrid.x_vec), grid_weights[:, :, Nz // 2], shading="nearest")
    ax_b.set(xlabel="y [mm]", ylabel="x [mm]", title="Off-Grid Source Weights")
    ax_b.invert_yaxis()
    fig.tight_layout()

    # plot the pressure field
    fig, ax = plt.subplots()
    im = ax.pcolormesh(1e3 * y_vec, 1e3 * x_vec, 1e-6 * amp, shading="gouraud")
    ax.set(xlabel="Lateral Position [mm]", ylabel="Axial Position [mm]", title="Pressure Field")
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax, label="[MPa]")

    plt.show()
