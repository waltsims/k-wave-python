"""
Modelling A Circular Plane Piston Transducer Assuming Axisymmetry Example

This example models a circular piston transducer assuming axisymmetry.
The on-axis pressure is compared with the analytical expression from [1].
Compared to the 3D simulation, a lower CFL (which gives a smaller time
step) is used, as the k-space correction for the axisymmetric code is not
exact in the radial direction.
[1] A. D. Pierce, Acoustics: An Introduction to its Physical Principles
and Applications. New York: Acoustical Society of America, 1989.
"""

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from kwave.data import Vector
from kwave.utils.math import round_even
from kwave.utils.kwave_array import kWaveArray
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.signals import create_cw_signals
from kwave.utils.filters import extract_amp_phase
from kwave.kspaceFirstOrderAS import kspaceFirstOrderASC
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# medium parameters
c0 = 1500.0  # sound speed [m/s]
rho0 = 1000.0  # density [kg/m^3]

# piston diameter [m]
source_diam = 10e-3

# source frequency [Hz]
source_f0 = 1e6

# source pressure [Pa]
source_mag = np.array([1e6])

# phase [rad]
source_phase = np.array([0.0])

# grid parameters
axial_size = 32e-3  # total grid size in the axial dimension [m]
lateral_size = 8e-3  # total grid size in the lateral dimension [m]

# computational parameters
ppw = 4  # number of points per wavelength
t_end = 40e-6  # total compute time [s] (this must be long enough to reach steady state)
record_periods = 1  # number of periods to record
cfl = 0.05  # CFL number
bli_tolerance = 0.05  # tolerance for truncation of the off-grid source points
upsampling_rate = 10  # density of integration points relative to grid

# =========================================================================
# RUN SIMULATION
# =========================================================================

# --------------------
# GRID
# --------------------

# grid resolution
dx = c0 / (ppw * source_f0)  # [m]

# compute the size of the grid
Nx = round_even(axial_size / dx)
Ny = round_even(lateral_size / dx)

grid_size_points = Vector([Nx, Ny])
grid_spacing_meters = Vector([dx, dx])

# create the k-space grid
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

# compute points per period
ppp = round(ppw / cfl)

# compute time step
dt = 1.0 / (ppp * source_f0)

# create the time array using an integer number of points per period
Nt = round(t_end / dt)
kgrid.setTime(Nt, dt)


# --------------------
# SOURCE
# --------------------

# create time varying continuous wave source
source_sig = create_cw_signals(np.squeeze(kgrid.t_array), source_f0, source_mag, source_phase)

# create empty kWaveArray this specfies the transducer properties in
# axisymmetric coordinate system
karray = kWaveArray(axisymmetric=True, bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate, single_precision=True)

# add line shaped element for transducer
karray.add_line_element([kgrid.x_vec[0].item(), -source_diam / 2.0], [kgrid.x_vec[0].item(), source_diam / 2.0])


# make a source object
source = kSource()

# assign binary mask using the karray
source.p_mask = karray.get_array_binary_mask(kgrid)

# assign source pressure output in time
source.p = karray.get_distributed_source_signal(kgrid, source_sig)


# --------------------
# MEDIUM
# --------------------

# water
medium = kWaveMedium(sound_speed=c0, density=rho0)

# --------------------
# SENSOR
# --------------------

sensor = kSensor()

# set sensor mask to record central plane, not including the source point
# sensor.mask = np.zeros((Nx, Ny), dtype=bool)
# sensor.mask[1:, :] = True

sensor.mask = np.ones((Nx, Ny), dtype=bool)

# set the record type: record the pressure waveform
sensor.record = ["p"]

# record only the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - record_periods * ppp + 1

# =========================================================================
# DEFINE THE SIMULATION PARAMETERS
# =========================================================================


# options for writing to file, but not doing simulations
simulation_options = SimulationOptions(
    simulation_type=SimulationType.AXISYMMETRIC, pml_auto=True, save_to_disk=True, save_to_disk_exit=False, pml_inside=False
)

execution_options = SimulationExecutionOptions(is_gpu_simulation=False, delete_data=False, verbose_level=2)

# =========================================================================
# RUN THE SIMULATION
# =========================================================================

sensor_data = kspaceFirstOrderASC(
    medium=medium,
    kgrid=kgrid,
    source=deepcopy(source),
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options,
)

# extract amplitude from the sensor data
amp, _, _ = extract_amp_phase(sensor_data["p"].T, 1.0 / kgrid.dt, source_f0, dim=1, fft_padding=1, window="Rectangular")

# reshape data
amp = np.reshape(amp, (Nx, Ny), order="F")

# extract pressure on axis
amp_on_axis = amp[:, 0]

# define axis vectors for plotting
yvec = np.squeeze(kgrid.y_vec) - kgrid.y_vec[0].item()
y_vec = 1e3 * np.hstack((-np.flip(yvec)[:-1], yvec))
x_vec = 1e3 * np.squeeze(kgrid.x_vec[:, :] - kgrid.x_vec[0])

# =========================================================================
# ANALYTICAL SOLUTION
# =========================================================================

# calculate the wavenumber
k: float = 2.0 * np.pi * source_f0 / c0

# define radius and axis
a: float = source_diam / 2.0
x_max: float = (Nx - 1) * dx
delta_x: float = x_max / 10000.0
x_ref: float = np.arange(0.0, x_max + delta_x, delta_x, dtype=float)

# calculate the analytical solution for a piston in an infinite baffle
# for comparison (Eq 5-7.3 in Pierce)
r_ref = np.sqrt(x_ref**2 + a**2)
p_ref = source_mag[0] * np.abs(2.0 * np.sin((k * r_ref - k * x_ref) / 2.0))

# # get analytical solution at exactly the same points as k-Wave
# r = np.sqrt(x_vec**2 + a**2)
# p_ref_kw = source_mag[0] * np.abs(2.0 * np.sin((k * r - k * x_vec) / 2.0))

# # calculate error
# L2_error = 100 * np.linalg.norm(p_ref_kw - amp_on_axis, ord=2)
# Linf_error = 100 * np.linalg.norm(p_ref_kw -  amp_on_axis, ord=np.inf)

# =========================================================================
# VISUALISATION
# =========================================================================

data = np.hstack((np.fliplr(amp[:, :-1]), amp)) / 1e6
sp = np.hstack((np.fliplr(source.p_mask[:, :])[:, :-1], source.p_mask))

# plot the pressure along the focal axis of the piston
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(1e3 * x_ref, 1e-6 * p_ref, "k-", label="Exact")
ax1.plot(x_vec, 1e-6 * amp_on_axis, "b.", label="k-Wave")
ax1.legend()
ax1.set(xlabel="Axial Position [mm]", ylabel="Pressure [MPa]", title="Axial Pressure")
ax1.grid()

# plot the source mask
fig2, ax2 = plt.subplots(1, 1)
ax2.pcolormesh(y_vec, x_vec, sp[:, :], shading="nearest")
ax2.set(xlabel="y [mm]", ylabel="x [mm]", title="Source Mask")
ax2.invert_yaxis()

fig3, ax3 = plt.subplots(1, 1)
im3 = ax3.pcolormesh(y_vec, x_vec, data, shading="gouraud")
cbar3 = fig3.colorbar(im3, ax=ax3)
_ = cbar3.ax.set_title("[MPa]", fontsize="small")
ax3.invert_yaxis()

plt.show()
