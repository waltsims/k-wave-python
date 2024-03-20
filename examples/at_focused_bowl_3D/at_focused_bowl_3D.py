import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from kwave.data import Vector

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.utils.filters import extract_amp_phase
from kwave.utils.mapgen import focused_bowl_oneil
from kwave.utils.math import round_even
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import create_cw_signals

from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D

from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# Modelling A Focused Bowl Transducer In 3D Example

# This example models a focused bowl transducer in 3D. The on-axis pressure
# is compared with the exact solution calculated using focused_bowl_oneil.

verbose: bool = False

# medium parameters
c0: float = 1500.0  # sound speed [m/s]
rho0: float = 1000.0  # density [kg/m^3]

# source parameters
source_f0 = 1.0e6  # source frequency [Hz]
source_roc = 30e-3  # bowl radius of curvature [m]
source_diameter = 30e-3  # bowl aperture diameter [m]
source_amp = 1.0e6  # source pressure [Pa]
source_phase = 0.0  # source phase [radians]

# grid parameters
axial_size: float = 50.0e-3  # total grid size in the axial dimension [m]
lateral_size: float = 40.0e-3  # total grid size in the lateral dimension [m]

# computational parameters
ppw: int = 3  # number of points per wavelength
t_end: float = 40e-6  # total compute time [s] (this must be long enough to reach steady state)
record_periods: int = 1  # number of periods to record
cfl: float = 0.5  # CFL number
source_x_offset: int = 20  # grid points to offset the source
bli_tolerance: float = 0.01  # tolerance for truncation of the off-grid source points
upsampling_rate: int = 10  # density of integration points relative to grid

# =========================================================================
# RUN SIMULATION
# =========================================================================

# --------------------
# GRID
# --------------------

# calculate the grid spacing based on the PPW and F0
dx: float = c0 / (ppw * source_f0)  # [m]

# compute the size of the grid
Nx: int = round_even(axial_size / dx) + source_x_offset
Ny: int = round_even(lateral_size / dx)
Nz: int = Ny

grid_size_points = Vector([Nx, Ny, Nz])
grid_spacing_meters = Vector([dx, dx, dx])

# create the k-space grid
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

# compute points per temporal period
ppp: float = np.round(ppw / cfl)

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

# set bowl position and orientation
bowl_pos = [kgrid.x_vec[0].item() + source_x_offset * kgrid.dx, 0.0, 0.0]
focus_pos = [kgrid.x_vec[-1].item(), 0.0, 0.0]

# create empty kWaveArray
karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate, single_precision=True)

# add bowl shaped element
karray.add_bowl_element(bowl_pos, source_roc, source_diameter, focus_pos)

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
sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
sensor.mask[(source_x_offset + 1) : -1, :, Nz // 2] = True

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
amp = np.reshape(amp, (Nx - (source_x_offset + 2), Ny), order="F")

# extract pressure on axis
amp_on_axis = amp[:, Ny // 2]

# define axis vectors for plotting
x_vec = kgrid.x_vec[(source_x_offset + 1) : -1, :] - kgrid.x_vec[source_x_offset]
y_vec = kgrid.y_vec

# =========================================================================
# ANALYTICAL SOLUTION
# =========================================================================

# calculate the wavenumber
knumber = 2.0 * np.pi * source_f0 / c0

# define axis
x_max = Nx * dx
delta_x = x_max / 10000.0
x_ref = np.arange(0.0, x_max + delta_x, delta_x)

Z = source_amp / (c0 * rho0)

# calculate analytical solution
p_ref_axial, _, _ = focused_bowl_oneil(source_roc, source_diameter, Z, source_f0, c0, 
                                       rho0, axial_positions=x_ref)

# calculate analytical solution at exactly the same points as the simulation
p_ref_axial_kw, _, _ = focused_bowl_oneil(source_roc, source_diameter, Z, source_f0, c0, 
                                          rho0, axial_positions=np.squeeze(x_vec))

L2_error = 100 * np.linalg.norm(p_ref_axial_kw - amp_on_axis, ord=2)
Linf_error = 100 * np.linalg.norm(p_ref_axial_kw - amp_on_axis, ord=np.inf)

# =========================================================================
# VISUALISATION
# =========================================================================

# plot the pressure along the focal axis of the piston
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(1e3 * x_ref, 1e-6 * p_ref_axial, "k-", label="Exact")
ax1.plot(1e3 * x_vec, 1e-6 * amp_on_axis, "b.", label="k-Wave")
ax1.legend()
ax1.set(xlabel="Axial Position [mm]", ylabel="Pressure [MPa]", title="Axial Pressure")
ax1.set_xlim(0.0, 1e3 * axial_size)
ax1.set_ylim(0.0, 20)
ax1.grid()

# plot the source mask (pml is outside the grid in this example)

# get grid weights
grid_weights = karray.get_array_grid_weights(kgrid)

fig2, (ax2a, ax2b) = plt.subplots(1, 2)
ax2a.pcolormesh(
    1e3 * np.squeeze(kgrid.y_vec),
    1e3 * np.squeeze(kgrid.x_vec),
    np.flip(source.p_mask[:, :, Nz // 2], axis=0),
    shading="nearest",
)
ax2a.set(xlabel="y [mm]", ylabel="x [mm]", title="Source Mask")
ax2b.pcolormesh(
    1e3 * np.squeeze(kgrid.y_vec),
    1e3 * np.squeeze(kgrid.x_vec),
    np.flip(grid_weights[:, :, Nz // 2], axis=0),
    shading="nearest",
)
ax2b.set(xlabel="y [mm]", ylabel="x [mm]", title="Off-Grid Source Weights")
plt.tight_layout(pad=1.2)

# plot the pressure field
fig3, ax3 = plt.subplots(1, 1)
ax3.pcolormesh(1e3 * np.squeeze(y_vec), 1e3 * np.squeeze(x_vec), np.flip(amp, axis=1), shading="gouraud")
ax3.set(xlabel="Lateral Position [mm]", ylabel="Axial Position [mm]", title="Pressure Field")
ax3.set_ylim(1e3 * x_vec[-1], 1e3 * x_vec[0])

# show figures
plt.show()
