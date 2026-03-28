from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import extract_amp_phase
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.math import round_even
from kwave.utils.signals import create_cw_signals

# material values
c0: float = 1500.0  # sound speed [m/s]
rho0: float = 1000.0  # density [kg/m^3]

# source parameters
source_f0: float = 1e6  # source frequency [Hz]
source_diam: float = 10e-3  # piston diameter [m]
source_amp: float = 1e6  # source pressure [Pa]
source_phase: float = 0.0  # source pressure [Pa]

# grid parameters
axial_size: float = 32e-3  # total grid size in the axial dimension [m]
lateral_size: float = 23e-3  # total grid size in the lateral dimension [m]

# computational parameters
ppw = 3  # number of points per wavelength
t_end: float = 40e-6  # total compute time [s] (this must be long enough to reach steady state)
record_periods: int = 1  # number of periods to record
cfl: float = 0.5  # CFL number
bli_tolerance: float = 0.03  # tolerance for truncation of the off-grid source points
upsampling_rate: int = 10  # density of integration points relative to grid

verbose: bool = False

# =========================================================================
# RUN SIMULATION
# =========================================================================

# --------------------
# GRID
# --------------------

kgrid = kWaveGrid.from_domain(
    dimensions=np.array([axial_size, lateral_size, lateral_size]), frequency=source_f0, sound_speed_min=c0, points_per_wavelength=ppw
)

# compute points per temporal period
ppp: int = round(ppw / cfl)

# compute corresponding time spacing
dt: float = 1.0 / (ppp * source_f0)

# create the time array using an integer number of points per period
Nt: int = int(np.round(t_end / dt))
kgrid.setTime(Nt, dt)

# calculate the actual CFL and PPW
if verbose:
    print("PPW = " + str(c0 / (dx * source_f0)))
    print("CFL = " + str(c0 * dt / dx))

# --------------------
# SOURCE
# --------------------

source = kSource()

# create time varying source
source_sig = create_cw_signals(np.squeeze(kgrid.t_array), source_f0, np.array([source_amp]), np.array([source_phase]))

# create empty kWaveArray
karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate)

# add disc shaped element at one end of the grid
karray.add_disc_element([kgrid.x_vec[0].item(), 0.0, 0.0], source_diam, [0.0, 0.0, 0.0])

# assign binary mask
source.p_mask = karray.get_array_binary_mask(kgrid)

# assign source signals
source.p = karray.get_distributed_source_signal(kgrid, source_sig)

# --------------------
# MEDIUM
# --------------------

# assign medium properties
medium = kWaveMedium(sound_speed=c0, density=rho0)

# --------------------
# SENSOR
# --------------------

sensor = kSensor()

# set sensor mask to record central plane, not including the source point
sensor.mask = np.zeros(kgrid.N, dtype=bool)
sensor.mask[1:, :, kgrid.Nz // 2] = True

# record the pressure
sensor.record = ["p"]

# record only the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - (record_periods * ppp) + 1

# --------------------
# SIMULATION
# --------------------

simulation_options = SimulationOptions(pml_auto=True, data_recast=True, save_to_disk=True, save_to_disk_exit=False, pml_inside=False)

execution_options = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False, verbose_level=2)

sensor_data = kspaceFirstOrder3D(
    medium=deepcopy(medium),
    kgrid=deepcopy(kgrid),
    source=deepcopy(source),
    sensor=deepcopy(sensor),
    simulation_options=simulation_options,
    execution_options=execution_options,
)

# extract amplitude from the sensor data
amp, _, _ = extract_amp_phase(sensor_data["p"].T, 1.0 / kgrid.dt, source_f0, dim=1, fft_padding=1, window="Rectangular")

# reshape data
amp = np.reshape(amp, (kgrid.Nx - 1, kgrid.Ny), order="F")

# extract pressure on axis
amp_on_axis = amp[:, kgrid.Ny // 2]

# define axis vectors for plotting
x_vec = kgrid.x_vec[1:, :] - kgrid.x_vec[0]
y_vec = kgrid.y_vec

# =========================================================================
# ANALYTICAL SOLUTION
# =========================================================================

# calculate the wavenumber
k: float = 2.0 * np.pi * source_f0 / c0

# define radius and axis
a: float = source_diam / 2.0
x_max: float = (kgrid.Nx - 1) * kgrid.dx
delta_x: float = x_max / 10000.0
x_ref: float = np.arange(0.0, x_max + delta_x, delta_x, dtype=float)

# calculate the analytical solution for a piston in an infinite baffle
# for comparison (Eq 5-7.3 in Pierce)
r_ref = np.sqrt(x_ref**2 + a**2)
p_ref = source_amp * np.abs(2.0 * np.sin((k * r_ref - k * x_ref) / 2.0))

# get analytical solution at exactly the same points as k-Wave
r = np.sqrt(x_vec**2 + a**2)
p_ref_kw = source_amp * np.abs(2.0 * np.sin((k * r - k * x_vec) / 2.0))

# calculate error
L2_error = 100 * np.linalg.norm(p_ref_kw - amp_on_axis, ord=2)
Linf_error = 100 * np.linalg.norm(p_ref_kw - amp_on_axis, ord=np.inf)
# =========================================================================
# VISUALISATION
# =========================================================================

# plot the pressure along the focal axis of the piston
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(1e3 * x_ref, 1e-6 * p_ref, "k-", label="Exact")
ax1.plot(1e3 * x_vec, 1e-6 * amp_on_axis, "b.", label="k-Wave")
ax1.legend()
ax1.set(xlabel="Axial Position [mm]", ylabel="Pressure [MPa]", title="Axial Pressure")
ax1.set_xlim(0.0, 1e3 * axial_size)
ax1.set_ylim(0.0, 1.1 * source_amp * 2e-6)
ax1.grid()

# plot the source mask (pml is outside the grid in this example)
fig2, ax2 = plt.subplots(1, 1)
ax2.pcolormesh(
    1e3 * np.squeeze(kgrid.y_vec), 1e3 * np.squeeze(kgrid.x_vec), np.flip(source.p_mask[:, :, kgrid.Nz // 2], axis=0), shading="nearest"
)
ax2.set(xlabel="y [mm]", ylabel="x [mm]", title="Source Mask")

# plot the pressure field
fig3, ax3 = plt.subplots(1, 1)
ax3.pcolormesh(1e3 * np.squeeze(y_vec), 1e3 * np.squeeze(x_vec), np.flip(amp, axis=1), shading="gouraud")
ax3.set(xlabel="Lateral Position [mm]", ylabel="Axial Position [mm]", title="Pressure Field")
ax3.set_ylim(1e3 * x_vec[-1], 1e3 * x_vec[0])

plt.show()
