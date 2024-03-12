import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from cycler import cycler

import h5py

from skimage import measure
from skimage.segmentation import find_boundaries
from scipy.interpolate import interpn

from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.checks import check_stability
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.signals import create_cw_signals
from kwave.utils.filters import extract_amp_phase
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu

from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

verbose: bool = True
savePlotting: bool = False
useMaxTimeStep: bool = True


Nx: int = 141
Nz: int = 241

dx: float = 0.5e-3
dz: float = dx

focus = 128

focus_coords = [(Nx - 1) // 2, focus]

bowl_coords = [(Nx - 1) // 2, 0]

# =========================================================================
# DEFINE THE MATERIAL PROPERTIES
# =========================================================================

# water
sound_speed = 1500.0 * np.ones((Nx,Nz))
density = 1000.0 * np.ones((Nx,Nz))
alpha_coeff = np.zeros((Nx,Nz))

# non-dispersive
alpha_power = 2.0

# cortical bone
sound_speed[:, 60:74] = 2800.0
density[:, 60:74] = 1850.0
alpha_coeff[:, 60:74] = 4.0

c0_min = np.min(np.ravel(sound_speed))
c0_max = np.max(np.ravel(sound_speed))

medium = kWaveMedium(sound_speed=sound_speed,
                     density=density,
                     alpha_coeff=alpha_coeff,
                     alpha_power=alpha_power)

# =========================================================================
# DEFINE THE TRANSDUCER SETUP
# =========================================================================

# bowl radius of curvature [m]
source_roc: float = 64.0e-3

# as we will use the bowl element this has to be a int or float
diameters: float = 64.0e-3

# frequency [Hz]
freq = 500e3

# source pressure [Pa]
source_amp = np.array([60e3])

# phase [rad]
source_phase = np.array([0.0])


# =========================================================================
# DEFINE COMPUTATIONAL PARAMETERS
# =========================================================================

# wavelength
k_min: float = c0_min / freq

# points per wavelength
ppw: float = k_min / dx

# number of periods to record
record_periods: int = 3

# compute points per period
ppp: int = 30

# CFL number determines time step
cfl: float = (ppw / ppp)


# =========================================================================
# DEFINE THE KGRID
# =========================================================================

grid_size_points = Vector([Nx, Nz])

grid_spacing_meters = Vector([dx, dz])

# create the k-space grid
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)


# =========================================================================
# DEFINE THE TIME VECTOR
# =========================================================================

# compute corresponding time stepping
dt = 1.0 / (ppp * freq)

# compute corresponding time stepping
dt = (c0_min / c0_max) / (float(ppp) * freq)

dt_stability_limit = check_stability(kgrid, medium)

if (useMaxTimeStep and (not np.isfinite(dt_stability_limit)) and (dt_stability_limit < dt)):
    dt_old = dt
    ppp = np.ceil(1.0 / (dt_stability_limit * freq))
    dt = 1.0 / (ppp * freq)

# calculate the number of time steps to reach steady state
t_end = np.sqrt(kgrid.x_size**2 + kgrid.y_size**2) / c0_min

# create the time array using an integer number of points per period
Nt = round(t_end / dt)

# make time array
kgrid.setTime(Nt, dt)

# calculate the actual CFL after adjusting for dt
cfl_actual = 1.0 / (dt * freq)


# =========================================================================
# DEFINE THE SOURCE PARAMETERS
# =========================================================================

# create empty kWaveArray this specfies the transducer properties
karray = kWaveArray(bli_tolerance=0.01,
                    upsampling_rate=16,
                    single_precision=True)

# set bowl position and orientation
bowl_pos = [kgrid.x_vec[bowl_coords[0]].item(),
            kgrid.y_vec[bowl_coords[1]].item()]

focus_pos = [kgrid.x_vec[focus_coords[0]].item(),
             kgrid.y_vec[focus_coords[1]].item()]

# add planar array
karray.add_line_element(start_point=[-10E-3, kgrid.y_vec[0].item()], end_point=[10e-3 + dx, kgrid.y_vec[0].item()])

# create time varying source
source_sig = create_cw_signals(np.squeeze(kgrid.t_array),
                               freq,
                               source_amp,
                               source_phase)

# make a source object.
source = kSource()

# assign binary mask using the karray
source.p_mask = karray.get_array_binary_mask(kgrid)

# assign source pressure output in time
source.p = karray.get_distributed_source_signal(kgrid, source_sig)


# =========================================================================
# DEFINE THE SENSOR PARAMETERS
# =========================================================================

sensor = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor.mask = np.ones((Nx, Nz), dtype=bool)

# set the record type: record the pressure waveform
sensor.record = ['p']

# record the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - (record_periods * ppp) + 1


# =========================================================================
# DEFINE THE SIMULATION PARAMETERS
# =========================================================================

DATA_CAST = 'single'
DATA_PATH = 'data/'

input_filename = 'ph1_bm3_sc2_input.h5'
output_filename = 'ph1_bm3_sc2_output.h5'

# options for writing to file, but not doing simulations
simulation_options = SimulationOptions(
    data_cast=DATA_CAST,
    data_recast=True,
    save_to_disk=True,
    input_filename=input_filename,
    output_filename=output_filename,
    save_to_disk_exit=False,
    data_path=DATA_PATH,
    pml_inside=False)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=True,
    delete_data=False,
    verbose_level=2)


# =========================================================================
# RUN THE SIMULATION
# =========================================================================

sensor_data = kspace_first_order_2d_gpu(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options)

# sampling frequency
fs = 1.0 / kgrid.dt

# get Fourier coefficients
amp, _, _ = extract_amp_phase(sensor_data['p'].T, fs, freq, dim=1, fft_padding=1, window='Rectangular')

# reshape to array
p = np.reshape(amp, (Nx, Nz), order='F')

# axes for plotting
x_vec = kgrid.x_vec
y_vec = kgrid.y_vec[0] - kgrid.y_vec

fig1, ax1 = plt.subplots(1, 1)
p1 = ax1.pcolormesh(1e3 * np.squeeze(x_vec),
                    1e3 * np.squeeze(y_vec),
                    np.flip(p.T, axis=1) / 1e3,
                    shading='gouraud', cmap='viridis')
ax1.set(xlabel='Lateral Position [mm]',
        ylabel='Axial Position [mm]',
        title='PH1-BM3-SC2')
ax1.set_aspect('equal')
cbar1 = fig1.colorbar(p1, ax=ax1)
_ = cbar1.ax.set_title('[kPa]', fontsize='small')

plt.show()