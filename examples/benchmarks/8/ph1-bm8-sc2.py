import numpy as np

import logging
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

import h5py

from skimage import measure
from skimage.segmentation import find_boundaries
from scipy.interpolate import interpn
from scipy.interpolate import RegularGridInterpolator

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

import pyvista as pv


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
transducer = 'sc2'

mask_folder = 'C:/Users/dsinden/GitHub/k-wave-python/data/'

mask_filename = mask_folder + 'skull_mask_' + tag + '_dx_' + res + '.mat'

if verbose:
    logger.info(mask_filename)

data = h5py.File(mask_filename, 'r')

if verbose:
    logger.info( list(data.keys()) )

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

# convert to Fortran-ordered arrays
skull_mask = np.reshape(skull_mask.flatten(), matlab_shape, order='F')
brain_mask = np.reshape(brain_mask.flatten(), matlab_shape, order='F')

# create water mask
water_mask = np.ones(skull_mask.shape, dtype=int) - (skull_mask.astype(int) +
                                                     brain_mask.astype(int))
water_mask = water_mask.astype(bool)

# orientation of axes
skull_mask = np.swapaxes(skull_mask, 0, 2)
brain_mask = np.swapaxes(brain_mask, 0, 2)
water_mask = np.swapaxes(water_mask, 0, 2)

# cropping settings - was 10
skull_mask = skull_mask[48:145, 48:145, 16:]
brain_mask = brain_mask[48:145, 48:145, 16:]
water_mask = water_mask[48:145, 48:145, 16:]

Nx, Ny, Nz = skull_mask.shape

msg = "new shape=" + str(skull_mask.shape)
if verbose:
    logger.info(msg)

if (transducer == 'sc1'):
    # curved element with focal depth of 64 mm, so is scaled by resolution to give value in grid point
    # bowl radius of curvature [m]
    msg = "transducer is focused"
    focus = int(64 / data['dx'][:].item())
    focus_coords = [(Nx - 1) // 2, (Ny - 1) // 2, focus]
    bowl_coords = [(Nx - 1) // 2, (Ny - 1) // 2, 0]

if (transducer == 'sc2'):
    # planar element
    msg = "transducer is planar"
    focus_coords = [(Nx - 1) // 2, (Ny - 1) // 2, (Nz - 1) // 2]
    disc_coords = [(Nx - 1) // 2, (Ny - 1) // 2, 0]

if verbose:
    logger.info(msg)

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

# =========================================================================
# DEFINE THE TRANSDUCER SETUP
# =========================================================================

# single spherical transducer
if (transducer == 'sc1'):

    # bowl radius of curvature [m]
    source_roc = 64.0e-3

    # as we will use the bowl element this has to be a int or float
    diameters = 64.0e-3

elif (transducer == 'sc2'):

    # diameter of the disc
    diameter = 10e-3

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
k_min = c0_min / freq

# points per wavelength
ppw = k_min / dx

# number of periods to record
record_periods: int = 3

# compute points per period
ppp: int = 20

# CFL number determines time step
cfl = (ppw / ppp)


# =========================================================================
# DEFINE THE KGRID
# =========================================================================

grid_size_points = Vector([Nx, Ny, Nz])

grid_spacing_meters = Vector([dx, dy, dz])

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
msg = "dt_stability_limit=" + str(dt_stability_limit) + ", dt=" + str(dt)
if verbose:
    logger.info(msg)

if (useMaxTimeStep and (not np.isfinite(dt_stability_limit)) and
    (dt_stability_limit < dt)):
    dt_old = dt
    ppp = np.ceil( 1.0 / (dt_stability_limit * freq) )
    dt = 1.0 / (ppp * freq)
    if verbose:
        logger.info("updated dt")
else:
    if verbose:
        logger.info("not updated dt")


# calculate the number of time steps to reach steady state
t_end = np.sqrt(kgrid.x_size**2 + kgrid.y_size**2) / c0_min

# create the time array using an integer number of points per period
Nt = round(t_end / dt)

# make time array
kgrid.setTime(Nt, dt)

# calculate the actual CFL after adjusting for dt
cfl_actual = 1.0 / (dt * freq)

if verbose:
    logger.info('PPW = ' + str(ppw))
    logger.info('CFL = ' + str(cfl_actual))
    logger.info('PPP = ' + str(ppp))


# =========================================================================
# DEFINE THE SOURCE PARAMETERS
# =========================================================================

if verbose:
    logger.info("kSource")

# create empty kWaveArray this specfies the transducer properties
karray = kWaveArray(bli_tolerance=0.01,
                    upsampling_rate=16,
                    single_precision=True)

if (transducer == 'sc1'):

    # set bowl position and orientation
    bowl_pos = [kgrid.x_vec[bowl_coords[0]].item(),
                kgrid.y_vec[bowl_coords[1]].item(),
                kgrid.z_vec[bowl_coords[2]].item()]

    focus_pos = [kgrid.x_vec[focus_coords[0]].item(),
                 kgrid.y_vec[focus_coords[1]].item(),
                 kgrid.z_vec[focus_coords[2]].item()]

    # add bowl shaped element
    karray.add_bowl_element(bowl_pos, source_roc, diameters, focus_pos)

elif (transducer == 'sc2'):

    # set disc position
    position = [kgrid.x_vec[disc_coords[0]].item(),
                kgrid.y_vec[disc_coords[1]].item(),
                kgrid.z_vec[disc_coords[2]].item()]

    # arbitrary position
    focus_pos = [kgrid.x_vec[focus_coords[0]].item(),
                 kgrid.y_vec[focus_coords[1]].item(),
                 kgrid.z_vec[focus_coords[2]].item()]

    # add disc-shaped planar element
    karray.add_disc_element(position, diameter, focus_pos)

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

if verbose:
    logger.info("kSensor")

sensor = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor.mask = np.ones((Nx, Ny, Nz), dtype=bool)

# set the record type: record the pressure waveform
sensor.record = ['p']

# record the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - record_periods * ppp + 1


# =========================================================================
# DEFINE THE SIMULATION PARAMETERS
# =========================================================================

DATA_CAST = 'single'
DATA_PATH = './'

input_filename = tag + '_' + transducer + '_' + res + '_input.h5'
output_filename = tag + '_' + transducer + '_' + res + '_output.h5'

# set input options
if verbose:
    logger.info("simulation_options")

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

if verbose:
    logger.info("execution_options")

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=True,
    delete_data=False,
    verbose_level=2)



# =========================================================================
# RUN THE SIMULATION
# =========================================================================

if verbose:
    logger.info("kspaceFirstOrder3DG")

sensor_data = kspaceFirstOrder3DG(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options)


# =========================================================================
# POST-PROCESS
# =========================================================================

if verbose:
    logger.info("post processing")

# sampling frequency
fs = 1.0 / kgrid.dt

if verbose:
    logger.info("extract_amp_phase")

# get Fourier coefficients
amp, _, _ = extract_amp_phase(sensor_data['p'].T, fs, freq, dim=1,
                              fft_padding=1, window='Rectangular')

# reshape data: matlab uses Fortran ordering
p = np.reshape(amp, (Nx, Ny, Nz), order='F')

x = np.linspace(-Nx // 2, Nx // 2 - 1, Nx)
y = np.linspace(-Ny // 2, Ny // 2 - 1, Ny)
z = np.linspace(-Nz // 2, Nz // 2 - 1, Nz)
x, y, z = np.meshgrid(x, y, z, indexing='ij')

pmax = np.nanmax(p)
max_loc = np.unravel_index(np.nanargmax(p), p.shape, order='C')

p_water = np.empty_like(p)
p_water.fill(np.nan)
p_water[water_mask] = p[water_mask]
pmax_water = np.nanmax(p_water)
max_loc_water = np.unravel_index(np.nanargmax(p_water), p.shape, order='C')

p_skull = np.empty_like(p)
p_skull.fill(np.nan)
p_skull[skull_mask] = p[skull_mask]
pmax_skull = np.nanmax(p_skull)
max_loc_skull = np.unravel_index(np.nanargmax(p_skull), p.shape, order='C')

p_brain = np.empty_like(p)
p_brain.fill(np.nan)
p_brain[brain_mask] = p[brain_mask]
pmax_brain = np.nanmax(p_brain)
max_loc_brain = np.unravel_index(np.nanargmax(p_brain), p.shape, order='C')

# domain axes
x_vec = np.linspace(kgrid.x_vec[0].item(), kgrid.x_vec[-1].item(), kgrid.Nx)
y_vec = np.linspace(kgrid.y_vec[0].item(), kgrid.y_vec[-1].item(), kgrid.Ny)
z_vec = np.linspace(kgrid.z_vec[0].item(), kgrid.z_vec[-1].item(), kgrid.Nz)

# colours
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# brain axes
# x
if (transducer == 'sc1'):
    indx = bowl_coords[2]
    indy = bowl_coords[0]
elif (transducer == 'sc2'):
    indx = disc_coords[2]
    indy = disc_coords[0]
x_x = [kgrid.z_vec[indx].item(), kgrid.z_vec[max_loc_brain[2]].item()]
y_x = [kgrid.x_vec[indy].item(), kgrid.x_vec[max_loc_brain[0]].item()]
coefficients_x = np.polyfit(x_x, y_x, 1)
polynomial_x = np.poly1d(coefficients_x)
axis = np.linspace(kgrid.z_vec[0].item(), kgrid.z_vec[-1].item(), kgrid.Nz)
beam_axis_x = polynomial_x(z_vec)
# y
if (transducer == 'sc1'):
    indx = bowl_coords[2]
    indy = bowl_coords[1]
elif (transducer == 'sc2'):
    indx = disc_coords[2]
    indy = disc_coords[1]
x_y = [kgrid.z_vec[indx].item(), kgrid.z_vec[max_loc_brain[2]].item()]
y_y = [kgrid.y_vec[indy].item(), kgrid.y_vec[max_loc_brain[1]].item()]
coefficients_y = np.polyfit(x_y, y_y, 1)
polynomial_y = np.poly1d(coefficients_y)
beam_axis_y = polynomial_y(z_vec)
# beam axis
beam_axis = np.vstack((beam_axis_x, beam_axis_y, z_vec)).T
# interpolate for pressure on brain axis
beam_pressure_brain = interpn((x_vec, y_vec, z_vec) , p, beam_axis,
                    method='linear', bounds_error=False, fill_value=np.nan)

# skull axes
# x
if (transducer == 'sc1'):
    indx = bowl_coords[2]
    indy = bowl_coords[0]
elif (transducer == 'sc2'):
    indx = disc_coords[2]
    indy = disc_coords[0]
x_x = [kgrid.z_vec[indx].item(), kgrid.z_vec[max_loc_skull[2]].item()]
y_x = [kgrid.x_vec[indy].item(), kgrid.x_vec[max_loc_skull[0]].item()]
coefficients_x = np.polyfit(x_x, y_x, 1)
polynomial_x = np.poly1d(coefficients_x)
axis = np.linspace(kgrid.z_vec[0].item(), kgrid.z_vec[-1].item(), kgrid.Nz)
beam_axis_x = polynomial_x(z_vec)
# y
if (transducer == 'sc1'):
    indx = bowl_coords[2]
    indy = bowl_coords[1]
elif (transducer == 'sc2'):
    indx = disc_coords[2]
    indy = disc_coords[1]
x_y = [kgrid.z_vec[indx].item(), kgrid.z_vec[max_loc_skull[2]].item()]
y_y = [kgrid.y_vec[indy].item(), kgrid.y_vec[max_loc_skull[1]].item()]
coefficients_y = np.polyfit(x_y, y_y, 1)
polynomial_y = np.poly1d(coefficients_y)
beam_axis_y = polynomial_y(z_vec)

# beam axis
beam_axis = np.vstack((beam_axis_x, beam_axis_y, z_vec)).T

# interpolate for pressure
beam_pressure_skull = interpn((x_vec, y_vec, z_vec) , p, beam_axis,
                    method='linear', bounds_error=False, fill_value=np.nan)



# plot pressure on through centre lines
fig1, ax1 = plt.subplots()
ax1.plot(p[(Nx-1)//2, (Nx-1)//2, :] / 1e6, label='geometric')
ax1.plot(beam_pressure_brain / 1e6, label='focal')
ax1.plot(beam_pressure_skull / 1e6, label='skull')
ax1.hlines(pmax_brain / 1e6, 0, len(z_vec), color=cycle[1], linestyle='dashed', lw=0.5)
ax1.hlines(pmax_skull / 1e6, 0, len(z_vec), color=cycle[2], linestyle='dashed', lw=0.5)
ax1.set(xlabel='Axial Position [mm]',
        ylabel='Pressure [MPa]',
        title='Centreline Pressures')
ax1.legend()
ax1.grid(True)



def get_edges(mask, fill_with_nan=True):
    """returns the mask as a float array and Np.NaN"""
    edges = find_boundaries(mask, mode='thin').astype(np.float32)
    if fill_with_nan:
        edges[edges == 0] = np.nan
    return edges

# contouring block

edges_x = get_edges(np.transpose(skull_mask[max_loc_brain[0], :, :]).astype(int), fill_with_nan=False)
edges_y = get_edges(np.transpose(skull_mask[:, max_loc_brain[1], :]).astype(int), fill_with_nan=False)
edges_z = get_edges(np.transpose(skull_mask[:, :, max_loc_brain[2]]).astype(int), fill_with_nan=False)

contour_x, num_x = measure.label(edges_x, background=0, return_num=True, connectivity=2)
contour_y, num_y = measure.label(edges_y, background=0, return_num=True, connectivity=2)
contour_z, num_z = measure.label(edges_z, background=0, return_num=True, connectivity=2)

if verbose:
    msg = "size of contours:" + str(np.shape(contour_x)) + ", " + str(np.shape(contour_y)) + ", " + str(np.shape(contour_z)) + "."
    logger.info(msg)
    msg = "number of contours: (" + str(num_x) + ", " + str(num_y) + ", " + str(num_z) + ")."
    logger.info(msg)

jmax = 0
jmin = Ny
i_inner = None
i_outer = None
# for a number of contours
for i in range(num_x):
    idx = int(np.shape(contour_x)[1] // 2)
    j = np.argmax(np.where(contour_x[:, idx]==(i+1), 1, 0))
    if (j > jmax):
        jmax = j
        i_outer = i + 1
    k = np.argmin(np.where(contour_x[:, idx]==(i+1), 0, 1))
    if (k < jmin):
        jmin = k
        i_inner = i + 1
contours_x_inner = measure.find_contours(np.where(contour_x==i_inner, 1, 0))
if not contours_x_inner:
    logger.warning("size of contours_x_inner is zero")
contours_x_outer = measure.find_contours(np.where(contour_x==i_outer, 1, 0))
if not contours_x_outer:
    logger.warning("size of contours_x_outer is zero")
inner_index_x = float(Ny)
outer_index_x = float(0)
for i in range(len(contours_x_inner)):
    x_min = np.min(contours_x_inner[i][:, 1])
    if (x_min < inner_index_x):
        inner_index_x = i
for i in range( len(contours_x_outer) ):
    x_max = np.max(contours_x_outer[i][:, 1])
    if (x_max > outer_index_x):
        outer_index_x = i

jmax = 0
jmin = Nx
i_inner = None
i_outer = None
for i in range(num_y):
    idy: int = int(np.shape(contour_y)[1] // 2)
    j = np.argmax(np.where(contour_y[:, idy]==(i+1), 1, 0))
    if (j > jmax):
        jmax = j
        i_outer = i + 1
    k = np.argmin(np.where(contour_y[:, idy]==(i+1), 0, 1))
    if (k < jmin):
        jmin = k
        i_inner = i + 1
contours_y_inner = measure.find_contours(np.where(contour_y==i_inner, 1, 0))
if not contours_y_inner:
    logger.warning("size of contours_y_inner is zero")
contours_y_outer = measure.find_contours(np.where(contour_y==i_outer, 1, 0))
if not contours_y_outer:
    logger.warning("size of contours_y_outer is zero")
inner_index_y = float(Nx)
outer_index_y = float(0)
for i in range( len(contours_y_inner) ):
    y_min = np.min(contours_y_inner[i][:, 1])
    if (y_min < inner_index_y):
        inner_index_y = i
for i in range( len(contours_y_outer) ):
    y_max = np.max(contours_y_outer[i][:, 1])
    if (y_max > outer_index_y):
        outer_index_y = i

jmax = 0
jmin = Ny
i_inner = None
i_outer = None
for i in range(num_z):
    idz: int = int(np.shape(contour_z)[1] // 2)
    j = np.argmax(np.where(contour_z[:, idz]==(i+1), 1, 0))
    if (j > jmax):
        jmax = j
        i_outer = i+1
    k = np.argmin(np.where(contour_z[:, idz]==(i+1), 0, 1))
    if (k < jmin):
        jmin = k
        i_inner = i+1

contours_z_inner = measure.find_contours(np.where(contour_z==i_inner, 1, 0))
if not contours_z_inner:
    logger.warning("size of contours_z_inner is zero")
else:
    inner_index_z = float(Nx)
    for i in range( len(contours_z_inner) ):
        z_min = np.min(contours_z_inner[i][:, 1])
        if (z_min < inner_index_z):
            inner_index_z = i

contours_z_outer = measure.find_contours(np.where(contour_z==i_outer, 1, 0))
if not contours_z_outer:
    logger.warning("size of contours_z_outer is zero")
else:
    outer_index_z = float(0)
    for i in range( len(contours_z_outer) ):
        z_max = np.max(contours_z_outer[i][:, 1])
        if (z_max > outer_index_z):
            outer_index_z = i

# end of contouring block

edges_x = get_edges(np.transpose(skull_mask[max_loc_brain[0], :, :]).astype(int))
edges_y = get_edges(np.transpose(skull_mask[:, max_loc_brain[1], :]).astype(int))
edges_z = get_edges(np.transpose(skull_mask[:, :, max_loc_brain[2]]).astype(int), fill_with_nan=True)

# plot the pressure field at mid point along z axis
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(p[:, :, max_loc_brain[2]] / 1e6,
            aspect='auto',
            interpolation='none',
            origin='lower',
            cmap='viridis')

if not contours_z_inner:
    ax2.imshow(edges_z, aspect='auto', interpolation='none',
               cmap='Greys', origin='upper')
else:
      ax2.plot(contours_z_inner[inner_index_z][:, 1],
               contours_z_inner[inner_index_z][:, 0], 'w', linewidth=0.5)
if not contours_z_outer:
    pass
else:
    ax2.plot(contours_z_outer[outer_index_z][:, 1],
             contours_z_outer[outer_index_z][:, 0], 'w', linewidth=0.5)

ax2.set(xlabel=r'$x$ [mm]',
        ylabel=r'$y$ [mm]',
        title='Pressure Field')
ax2.grid(False)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar_2 = fig2.colorbar(im2, cax=cax2)
cbar_2.ax.set_title('[MPa]', fontsize='small')

pwater_max_x = np.nanmax(p_water[max_loc_brain[0], :, :].flatten())
pskull_max_x = np.nanmax(p_skull[max_loc_brain[0], :, :].flatten())
pbrain_max_x = np.nanmax(p_brain[max_loc_brain[0], :, :].flatten())

pwater_max_y = np.nanmax(p_water[:, max_loc_brain[1], :].flatten())
pskull_max_y = np.nanmax(p_skull[:, max_loc_brain[1], :].flatten())
pbrain_max_y = np.nanmax(p_brain[:, max_loc_brain[1], :].flatten())

fig3, (ax3a, ax3b) = plt.subplots(1,2)
im3a_water = ax3a.imshow(p_water[max_loc_brain[0], :, :].T,
            aspect='auto',
            interpolation='none',
            origin='upper',
            cmap='cool')
im3a_skull = ax3a.imshow(p_skull[max_loc_brain[0], :, :].T,
            aspect='auto',
            interpolation='none',
            origin='upper',
            cmap='turbo')
im3a_brain = ax3a.imshow(p_brain[max_loc_brain[0], :, :].T,
            aspect='auto',
            interpolation='none',
            origin='upper',
            cmap='viridis')

ax3a.plot(contours_x_inner[inner_index_x][:, 1],
          contours_x_inner[inner_index_x][:, 0], 'k', linewidth=0.5)
ax3a.plot(contours_x_outer[outer_index_x][:, 1],
          contours_x_outer[outer_index_x][:, 0], 'k', linewidth=0.5)

ax3a.grid(False)
ax3a.axes.get_yaxis().set_visible(False)
ax3a.axes.get_xaxis().set_visible(False)
divider3a = make_axes_locatable(ax3a)
cax3a = divider3a.append_axes("right", size="5%", pad=0.05)
cbar_3a = fig3.colorbar(im3a_brain, cax=cax3a)
cbar_3a.ax.set_title('[kPa]', fontsize='small')
ax3b.imshow(p_water[:, max_loc_brain[1], :].T,
            aspect='auto',
            interpolation='none',
            origin='upper',
            cmap='cool')
ax3b.imshow(p_skull[:, max_loc_brain[1], :].T,
            aspect='auto',
            interpolation='none',
            origin='upper',
            cmap='turbo')
im3b_brain = ax3b.imshow(p_brain[:, max_loc_brain[1], :].T,
            aspect='auto',
            interpolation='none',
            origin='upper',
            cmap='viridis')

ax3b.grid(False)
ax3b.axes.get_yaxis().set_visible(False)
ax3b.axes.get_xaxis().set_visible(False)
divider3b = make_axes_locatable(ax3b)
cax3b = divider3b.append_axes("right", size="5%", pad=0.05)
cbar_3b = fig3.colorbar(im3b_brain, cax=cax3b)
cbar_3b.ax.set_title('[Pa]', fontdict={'fontsize':8})


fig4, ax4 = plt.subplots()
if not contours_z_inner:
    pass
else:
    ax4.plot(contours_z_inner[inner_index_z][:, 1],
             contours_z_inner[inner_index_z][:, 0], 'w', linewidth=0.5)
if not contours_z_outer:
    pass
else:
    ax4.plot(contours_z_outer[outer_index_z][:, 1],
             contours_z_outer[outer_index_z][:, 0], 'w', linewidth=0.5)


fig5, (ax5a, ax5b) = plt.subplots(1,2)
im5a = ax5a.imshow(p[max_loc_brain[0], :, :].T / 1e6,
                   vmin=0, vmax=pmax / 1e6,
                   aspect='auto',
                   interpolation='none',
                   origin='upper',
                   cmap='viridis')
im5a_boundary = ax5a.imshow(edges_x, aspect='auto', interpolation='none',
                            cmap='Greys', origin='upper', alpha=0.75)
ax5a.grid(False)
ax5a.axes.get_yaxis().set_visible(False)
ax5a.axes.get_xaxis().set_visible(False)
divider5a = make_axes_locatable(ax5a)
cax5a = divider5a.append_axes("right", size="5%", pad=0.05)
cbar_5a = fig5.colorbar(im5a, cax=cax5a)
cbar_5a.ax.set_title('[MPa]', fontsize='small')
im5b = ax5b.imshow(p[:, max_loc_brain[1], :].T / 1e6,
                   vmin=0, vmax=pmax / 1e6,
                   aspect='auto',
                   interpolation='none',
                   origin='upper',
                   cmap='viridis')
im5b_boundary = ax5b.imshow(edges_y, aspect='auto', interpolation='none',
                            cmap='Greys',origin='upper', alpha=0.75)
ax5b.grid(False)
ax5b.axes.get_yaxis().set_visible(False)
ax5b.axes.get_xaxis().set_visible(False)
divider5b = make_axes_locatable(ax5b)
cax5b = divider5b.append_axes("right", size="5%", pad=0.05)
cbar_5b = fig5.colorbar(im5b, cax=cax5b)
cbar_5b.ax.set_title('[MPa]', fontsize='small')

all_contours_x = []
for i in range(num_x):
    all_contours_x.append(measure.find_contours(np.where(contour_x==(i+1), 1, 0)))

all_contours_y = []
for i in range(num_y):
    all_contours_y.append(measure.find_contours(np.where(contour_y==(i+1), 1, 0)))

fig6, (ax6a, ax6b) = plt.subplots(1,2)
im6a = ax6a.imshow(p[max_loc_brain[0], :, :].T / 1e6,
                   vmin=0, vmax=pmax / 1e6,
                   aspect='auto',
                   interpolation='none',
                   origin='upper',
                   cmap='viridis')
for contour in all_contours_x:
    # logger.info(contour  dir(contour))
    for i in range( len(contour) ):
        ax6a.plot(contour[i][:, 1], contour[i][:, 0], 'w', linewidth=0.5)

ax6a.grid(False)
ax6a.axes.get_yaxis().set_visible(False)
ax6a.axes.get_xaxis().set_visible(False)
divider6a = make_axes_locatable(ax5a)
cax6a = divider6a.append_axes("right", size="5%", pad=0.05)
cbar_6a = fig6.colorbar(im6a, cax=cax6a)
cbar_6a.ax.set_title('[MPa]', fontsize='small')
im6b = ax6b.imshow(p[:, max_loc_brain[1], :].T / 1e6,
                   vmin=0, vmax=pmax / 1e6,
                   aspect='auto',
                   interpolation='none',
                   origin='upper',
                   cmap='viridis')

custom_cycler = cycler(ls=['-', '--', ':', '-.'])

ax6b.set_prop_cycle(custom_cycler)

for idx, contour in enumerate(all_contours_y):
    for i in range( len(contour) ):
        ax6b.plot(contour[i][:, 1], contour[i][:, 0], c=cycle[idx],
                  linewidth=0.5, label=str(idx))
ax6b.legend()
ax6b.grid(False)
ax6b.axes.get_yaxis().set_visible(False)
ax6b.axes.get_xaxis().set_visible(False)
divider6b = make_axes_locatable(ax6b)
cax6b = divider6b.append_axes("right", size="5%", pad=0.05)
cbar_6b = fig6.colorbar(im6b, cax=cax6b)
cbar_6b.ax.set_title('[MPa]', fontsize='small')

# plt.show()

plotter = pv.Plotter()

pmax = np.nanmax(p)
pmin = np.nanmin(p)

grid = pv.ImageData()
grid.dimensions = np.array(p.shape) + 1
grid.spacing = (1, 1, 1)
grid.cell_data['pressure'] = np.ravel(p, order="F")

xslice_depth = max_loc_brain[0]
yslice_depth = max_loc_brain[1]
zslice_depth = max_loc_brain[2]



slice_x_focus = grid.slice(normal='x', origin=[xslice_depth, yslice_depth, zslice_depth],
                      generate_triangles=False, contour=False, progress_bar=False)
slice_y_focus = grid.slice(normal='y', origin=[xslice_depth, yslice_depth, zslice_depth],
                      generate_triangles=False, contour=False, progress_bar=False)
slice_z_focus = grid.slice(normal='z', origin=[xslice_depth, yslice_depth, zslice_depth],
                      generate_triangles=False, contour=False, progress_bar=False)

# slice_array = slice_z_focus.cell_data['pressure'].reshape(grid.dimensions[0]-1, grid.dimensions[1]-1)

slice_z_tx = grid.slice(normal='-z', origin=disc_coords,
                      generate_triangles=False, contour=False, progress_bar=False)

# slice_array = slice_z_tx.cell_data['pressure'].reshape(grid.dimensions[0]-1, grid.dimensions[1]-1)

slice_z_rx = grid.slice(normal='z', origin=[(Nx-1) // 2, (Ny - 1) // 2, Nz-1],
                      generate_triangles=False, contour=False, progress_bar=False)

slice_array = slice_z_rx.cell_data['pressure'].reshape(grid.dimensions[0]-1, grid.dimensions[1]-1)

# now get points on skull surfaces
verts, faces, normals, _ = measure.marching_cubes(skull_mask, 0)

vfaces = np.column_stack((np.ones(len(faces),) * 3, faces)).astype(int)

x = np.arange(p.shape[0])  # X-coordinates
y = np.arange(p.shape[1])  # Y-coordinates
z = np.arange(p.shape[2])  # Z-coordinates

# set up a interpolator
interpolator = RegularGridInterpolator((x, y, z), p)
# get the pressure values on the vertices
interpolated_values = interpolator(verts)

# set up mesh for skull surface
mesh = pv.PolyData(verts, vfaces)
mesh['Normals'] = normals

# Assign interpolated data to mesh
mesh.point_data['abs pressure'] = interpolated_values
# clip data
mesh.point_data['abs pressure'] = np.where(mesh.point_data['abs pressure'] > pmax_brain, pmax_brain, mesh.point_data['abs pressure'] )

if verbose:
    msg = 'focus in brain: ' + str(max_loc_brain) + ', mid point: ' + str(disc_coords) + ' last plane: ' + str(np.unravel_index(np.argmax(slice_array), slice_array.shape))
    logger.info(msg)

# Choose a colormap
plotter.add_mesh(mesh, scalars='abs pressure', opacity=0.25, show_edges=False, cmap='viridis', clim=[pmin, pmax_brain], show_scalar_bar=True)
plotter.add_mesh(slice_x_focus, opacity=0.95, cmap='viridis', clim=[pmin, pmax_brain], show_scalar_bar=False)
plotter.add_mesh(slice_y_focus, opacity=0.95, cmap='viridis', clim=[pmin, pmax_brain], show_scalar_bar=False)
plotter.add_mesh(slice_z_focus, opacity=0.95, cmap='viridis', clim=[pmin, pmax_brain], show_scalar_bar=False)
plotter.add_mesh(slice_z_tx, opacity=0.75, cmap='viridis', clim=[pmin, pmax_brain], show_scalar_bar=False)
plotter.add_mesh(slice_z_rx, opacity=0.75, cmap='viridis', clim=[pmin, pmax_brain], show_scalar_bar=False)
plotter.show_axes()
plotter.show_bounds()

plotter.show()
