import numpy as np

from copy import deepcopy
import requests
import shutil

import logging
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG

from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console and file handlers and set level to debug
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(filename='runner.log')
fh.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
# add formatter to ch, fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add ch, fh to logger
logger.addHandler(ch)
logger.addHandler(fh)
# propagate
ch.propagate = True
fh.propagate = True
logger.propagate = True

verbose: bool = True
savePlotting: bool = True
useMaxTimeStep: bool = True

tag = 'bm8'
res = '1mm'

url = 'https://raw.githubusercontent.com/djps/k-wave-python/benchmarks/examples/benchmarks/9/skull_mask_bm9_dx_1mm.mat'
mask_filename = requests.get(url, stream=True)
mask_filename.raw.decode_content = True
with open("temp.h5", "wb") as _fh:
    shutil.copyfileobj(mask_filename.raw, _fh)

data = h5py.File("temp.h5", 'r')

# is given in millimetres
dx = data['dx'][:].item()

# scale to metres
dx = dx / 1000.0
dy = dx
dz = dx

xi = np.squeeze(np.asarray(data['xi'][:]))
yi = np.squeeze(np.asarray(data['yi'][:]))
zi = np.squeeze(np.asarray(data['zi'][:]))

matlab_shape = np.shape(xi)[0], np.shape(yi)[0], np.shape(zi)[0]

skull_mask = np.squeeze(data['skull_mask'][:]).astype(bool)
brain_mask = np.squeeze(data['brain_mask'][:]).astype(bool)

skull_mask = np.reshape(skull_mask.flatten(), matlab_shape, order='F')
brain_mask = np.reshape(brain_mask.flatten(), matlab_shape, order='F')

water_mask = np.ones(skull_mask.shape, dtype=int) - (skull_mask.astype(int) + brain_mask.astype(int))
water_mask = water_mask.astype(bool)

skull_mask = np.swapaxes(skull_mask, 0, 2)
brain_mask = np.swapaxes(brain_mask, 0, 2)
water_mask = np.swapaxes(water_mask, 0, 2)

Nx, Ny, Nz = skull_mask.shape

focus = int(64 / data['dx'][:].item())

focus_coords = [(Nx - 1) // 2, (Ny - 1) // 2, focus]

bowl_coords = [(Nx - 1) // 2, (Ny - 1) // 2, 0]

disc_coords = [(Nx-1) // 2, (Ny-1) // 2, 0]

# =========================================================================
# DEFINE THE MATERIAL PROPERTIES
# =========================================================================

# water
sound_speed = 1500.0 * np.ones(skull_mask.shape)
density = 1000.0 * np.ones(skull_mask.shape)
alpha_coeff = np.zeros(skull_mask.shape)

# non-dispersive
alpha_power = 2.0

# skull
sound_speed[skull_mask] = 2800.0
density[skull_mask] = 1850.0
alpha_coeff[skull_mask] = 4.0

# brain
sound_speed[brain_mask] = 1560.0
density[brain_mask] = 1040.0
alpha_coeff[brain_mask] = 0.3

c0_min = np.min(sound_speed.flatten())
c0_max = np.min(sound_speed.flatten())

medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power
)

# bowl radius of curvature [m]
source_roc = 64.0e-3

# as we will use the bowl element this has to be a int or float
diameters = 64.0e-3

# frequency [Hz]
freq = 500e3

# source pressure [Pa]
source_amp = np.array([60e3])

# phase [rad]
source_phase = np.array([0.0])

# wavelength
k_min = c0_min / freq

# points per wavelength
ppw = k_min / dx

# number of periods to record
record_periods: int = 3

# compute points per period
ppp: int = 20

# CFL number determines time step
cfl = (ppw / ppp)

grid_size_points = Vector([Nx, Ny, Nz])

grid_spacing_meters = Vector([dx, dy, dz])

# create the k-space grid
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

# compute corresponding time stepping
dt = 1.0 / (ppp * freq)

# compute corresponding time stepping
dt = (c0_min / c0_max) / (float(ppp) * freq)

dt_stability_limit = check_stability(kgrid, medium)

if (useMaxTimeStep and (not np.isfinite(dt_stability_limit)) and
    (dt_stability_limit < dt)):
    dt_old = dt
    ppp = np.ceil( 1.0 / (dt_stability_limit * freq) )
    dt = 1.0 / (ppp * freq)
    if verbose:
        logger.info("updated dt")
else:
    pass

# calculate the number of time steps to reach steady state
t_end = np.sqrt(kgrid.x_size**2 + kgrid.y_size**2) / c0_min

# create the time array using an integer number of points per period
Nt = round(t_end / dt)

# make time array
kgrid.setTime(Nt, dt)

# calculate the actual CFL after adjusting for dt
cfl_actual = 1.0 / (dt * freq)

# create empty kWaveArray this specfies the transducer properties
karray = kWaveArray(bli_tolerance=0.01,
                    upsampling_rate=16,
                    single_precision=True)

# set bowl position and orientation
bowl_pos = [kgrid.x_vec[bowl_coords[0]].item(),
            kgrid.y_vec[bowl_coords[1]].item(),
            kgrid.z_vec[bowl_coords[2]].item()]

focus_pos = [kgrid.x_vec[focus_coords[0]].item(),
             kgrid.y_vec[focus_coords[1]].item(),
             kgrid.z_vec[focus_coords[2]].item()]

# add bowl shaped element
karray.add_bowl_element(bowl_pos, source_roc, diameters, focus_pos)

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

sensor = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor.mask = np.ones((Nx, Ny, Nz), dtype=bool)

# set the record type: record the pressure waveform
sensor.record = ['p_max', 'p_min']

# record the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - record_periods * ppp + 1

DATA_CAST = 'single'
DATA_PATH = './'

input_filename = tag + '_' + '_' + res + '_input.h5'
output_filename = tag + '_' + '_' + res + '_output.h5'

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

sensor_data = kspaceFirstOrder3DG(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options)

# sampling frequency
fs = 1.0 / kgrid.dt

# get Fourier coefficients
#amp, _, _ = extract_amp_phase(sensor_data['p'].T, fs, freq, dim=1, fft_padding=1, window='Rectangular')

# reshape data: matlab uses Fortran ordering
p = np.reshape(sensor_data['p_max'].T, (Nx, Ny, Nz), order='F')

# # get maximum pressure
# pmax = np.nanmax(p)

# # get location of maximum pressure
# max_loc = np.unravel_index(np.nanargmax(p), p.shape, order='F')

p_brain = np.empty_like(p)
p_brain.fill(np.nan)
p_brain[brain_mask] = p[brain_mask]
pmax_brain = np.nanmax(p_brain)
max_loc_brain = np.unravel_index(np.nanargmax(p_brain), p.shape, order='F')

fig3, (ax3a, ax3b) = plt.subplots(1, 2)
im3a = ax3a.pcolormesh(np.squeeze(kgrid.y_vec), np.squeeze(kgrid.z_vec), p[max_loc_brain[0], :, :].T / 1e6,
                       shading='gouraud', cmap='viridis')
ax3a.grid(False)
ax3a.axes.get_yaxis().set_visible(False)
ax3a.axes.get_xaxis().set_visible(False)
divider3a = make_axes_locatable(ax3a)
cax3a = divider3a.append_axes("right", size="5%", pad=0.05)
cbar_3a = fig3.colorbar(im3a, cax=cax3a)
cbar_3a.ax.set_title('[MPa]', fontsize='small')
ax3a.invert_yaxis()
im3b = ax3b.pcolormesh(np.squeeze(kgrid.x_vec), np.squeeze(kgrid.z_vec), p[:, max_loc_brain[1], :].T / 1e6,
                       shading='gouraud', cmap='viridis')
ax3b.grid(False)
ax3b.axes.get_yaxis().set_visible(False)
ax3b.axes.get_xaxis().set_visible(False)
divider3b = make_axes_locatable(ax3b)
cax3b = divider3b.append_axes("right", size="5%", pad=0.05)
cbar_3b = fig3.colorbar(im3a, cax=cax3b)
cbar_3b.ax.set_title('[MPa]', fontdict={'fontsize': 8})
ax3b.invert_yaxis()
plt.tight_layout(pad=1.2)

plt.show
