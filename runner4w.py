import numpy as np
import os
import logging
import sys
import h5py
import matplotlib.pyplot as plt

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

from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.signals import create_cw_signals
from kwave.utils.filters import extract_amp_phase
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG

from kwave.options import SimulationOptions, SimulationExecutionOptions

"""
This is based on the water example from
https://github.com/sitiny/BRIC_TUS_Simulation_Tools
"""

verbose: bool = False
doPlotting: bool = True
savePlotting: bool = True

# =========================================================================
# DEFINE THE TRANSDUCER SETUP
# =========================================================================

# single spherical transducer with four concentric elements, equal pressure,
# but different phases in order to get a coherent focus

# name of transducer
transducer = 'CTX500'

# pressure [Pa]
pressure = 51590.0

# phase offsets [degrees]
phase = np.array([0, 319, 278, 237])

# bowl radius of curvature [m]
source_roc = 63.2e-3

# this has to be a list of lists with each list in the main list being the
# aperture diameters of the elements given an inner, outer pairs
diameters = [[0, 1.28],
             [1.3, 1.802],
             [1.82, 2.19],
             [2.208, 2.52]]

# number of elements
n: int = np.size(phase)

# the data was provided in inches, so is scaled to metres. 
# Have to scale a list of list
scale = 0.0254
diameters = [[ scale * i for i in inner ] for inner in diameters]

# frequency [Hz]
freq = 500e3

# source pressure [Pa]
source_amp = np.squeeze(np.tile(pressure, [1, n]))

# phase [rad]
source_phase = np.squeeze(np.array([np.deg2rad(phase), ]))


# =========================================================================
# DEFINE THE MEDIUM PARAMETERS
# =========================================================================

# sound speed [m/s]
c0 = 1500.0

# density [kg/m^3]
rho0 = 1000.0

# Robertson et al., PMB 2017
alpha_power = 1.43

# [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017),
# see also 0.05 Fomenko et al., 2020
alpha_coeff = 0.0

medium = kWaveMedium(
    sound_speed=c0,
    density=rho0,
    alpha_coeff=alpha_coeff,
    alpha_power=alpha_power
)


# =========================================================================
# DEFINE COMPUTATIONAL PARAMETERS
# =========================================================================

ppw = 3                    # number of points per wavelength
record_periods = 3         # number of periods to record
cfl = 0.3                  # CFL number
ppp = np.round(ppw / cfl)  # compute points per period


# =========================================================================
# DEFINE THE KGRID
# =========================================================================

# grid parameters
axial_size = 128e-3    # total grid size in the axial dimension [m]
lateral_size = 128e-3  # total grid size in the lateral dimension [m]

# calculate the grid spacing based on the PPW and frequency.
# Note equal grid spacing
dx = c0 / (ppw * freq)  # [m]
dy = dx
dz = dx

# compute the number of points in the grid
Nx: int = int(2 * np.round((axial_size / dx) / 2) )
Ny: int = int(2 * np.round((lateral_size / dx) / 2) )
Nz: int = int(Ny)

# the matlab pml auto sets these to 17 as it is the nearest power of two
pml_inside: bool = False

if (pml_inside is True):
    pml_x_size: int = 34
    pml_y_size: int = 34
    pml_z_size: int = 34
    pml_size = np.array([pml_x_size, pml_y_size, pml_z_size])
else:
    pml_x_size: int = 0
    pml_y_size: int = 0
    pml_z_size: int = 0
    pml_size = None

Nx: int = int(Nx + pml_x_size)
Ny: int = int(Ny + pml_y_size)
Nz: int = int(Nz + pml_z_size)

grid_size_points = Vector([Nx, Ny, Nz])
grid_spacing_meters = Vector([dx, dy, dz])

# create the k-space grid
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)


# =========================================================================
# DEFINE THE TIME VECTOR
# =========================================================================

# compute corresponding time stepping
dt = 1.0 / (ppp * freq)

# calculate the number of time steps to reach steady state
t_end = np.sqrt(kgrid.x_size**2 + kgrid.y_size**2) / c0

# create the time array using an integer number of points per period
Nt: int = int(np.round(t_end / dt))
kgrid.setTime(Nt, dt)

# calculate the actual CFL and PPW
if verbose:
    msg = 'PPW = ' + str(c0 / (dx * freq))
    logger.info(str(msg))
    msg = 'CFL = ' + str(c0 * dt / dx)
    logger.info(str(msg))
    msg = 'source_amp.dim = ' + str(source_amp)
    logger.info(str(msg))
    msg = 'source_phase.dim = ' + str(source_phase)
    logger.info(str(msg))


# =========================================================================
# DEFINE THE SENSOR PARAMETERS
# =========================================================================

# create instance of a sensor
sensor = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor.mask = np.ones((Nx, Ny, Nz), dtype=bool)

# set the record type: record the pressure waveform
sensor.record = ['p']

# record the final few periods when the field is in steady state
sensor.record_start_index = kgrid.Nt - record_periods * ppp


# =========================================================================
# DEFINE THE SOURCE PARAMETERS
# =========================================================================

# source (location) parameters: index of bowl position and focal position
# in x-axis. Due to differs in matlab/python indexing, these are one less than
# in the matlab version, but correspond to the same poistion in space
bx: int = 9
fx: int = 53

# set bowl position - offset by pml
bowl_pos = [float(kgrid.x_vec[bx + pml_x_size]),
            0.0,
            0.0]

# set focus position - offset by pml
focus_pos = [float(kgrid.x_vec[fx + pml_x_size]),
             0.0,
             0.0]

# create empty kWaveArray in order to specify the transducer properties
karray = kWaveArray(bli_tolerance=0.01,
                    upsampling_rate=16,
                    single_precision=True)

# add a bowl shaped transducer
karray.add_annular_array(bowl_pos, source_roc, diameters, focus_pos)

# create time varying source
source_sig = create_cw_signals(np.squeeze(kgrid.t_array),
                               freq,
                               source_amp,
                               source_phase)

# make a source object
source = kSource()

# assign a binary mask using the karray class and the kgrid
source.p_mask = karray.get_array_binary_mask(kgrid)

# assign source pressure output in time, using karray class, the kgrid and the
# signal
source.p = karray.get_distributed_source_signal(kgrid, source_sig)


# =========================================================================
# DEFINE THE SIMULATION PARAMETERS
# =========================================================================

input_filename = 'brics_water_input.h5'
output_filename = 'brics_water_output.h5'
DATA_CAST = 'single'
DATA_PATH = './data/water/'
BINARY_PATH = './kwave/bin/windows/'

# set input options
if verbose:
    logger.info("simulation_options")

# options for writing to file, but not doing simulations
simulation_options = SimulationOptions(
    data_cast=DATA_CAST,
    data_recast=True,
    pml_size=pml_size,
    save_to_disk=True,
    input_filename=input_filename,
    output_filename=output_filename,
    pml_auto=True,
    save_to_disk_exit=False,
    data_path=DATA_PATH,
    pml_inside=False)

if verbose:
    logger.info("execution_options")

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=True,
    delete_data=False,
    verbose_level=2,
    binary_path=BINARY_PATH)


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

# This is how the data is actually output from the executable
with h5py.File(os.path.join(DATA_PATH, output_filename), 'r') as hf:
    sensor_data0 = np.array(hf['p'])[0].T 
    print(np.shape(sensor_data), np.shape(sensor_data0))
    a = np.reshape(sensor_data, (128, 128, 128, 31))
    b = np.reshape(sensor_data, (128, 128, 128, 31))
    print(a[int(Nx / 2), int(Ny / 2), :, 0:10] )
    print(b[int(Nx / 2), int(Ny / 2), :, 0:10] )


# =========================================================================
# POST-PROCESS
# =========================================================================

if verbose:
    logger.info("extract_amp_phase")

# sampling frequency
fs = 1.0 / kgrid.dt

# get Fourier coefficients
amp, phi, f = extract_amp_phase(sensor_data, fs, freq, dim=1,
                                fft_padding=1, window='Rectangular')

# reshape data
p = np.reshape(amp, (Nx, Ny, Nz), order='C')

# extract pressure on beam axis
amp_on_axis = p[int(Nx / 2), int(Ny / 2), :]

# define axis vectors for plotting: set from (0, Nx*dx) by shifting
x_vec = kgrid.x_vec - kgrid.x_vec[0]
y_vec = kgrid.y_vec

# scale axes to mm
x_vec = np.squeeze(x_vec) * 1e3
y_vec = np.squeeze(y_vec) * 1e3

# scale pressure to MPa
amp_on_axis = amp_on_axis * 1e-6

# location of maximum pressure
max_pressure = np.max(p.flatten())
idx = np.argmax(p.flatten())
mx, my, mz = np.unravel_index(idx, np.shape(p))

# pulse length [s]
pulse_length = 20e-3

# pulse repetition frequency [Hz]
pulse_rep_freq = 5.0

# spatial peak-pulse average of plane wave intensity
Isppa = max_pressure**2 / (2 * medium.density * medium.sound_speed)  # [W/m2]
Isppa = np.squeeze(Isppa) * 1e-4  # [W/cm2]

# spatial peak-temporal average of plane wave intensity
Ispta = Isppa * pulse_length * pulse_rep_freq  # [W/cm2]

# Mechanical Index (MI):  max_pressure [MPa] / sqrt freq [MHz]
MI = max_pressure * 1e-6 / np.sqrt(freq * 1e-6)

# distance to computational to theoretical focus.
ix = int( np.squeeze( np.where(kgrid.x_vec == 0)[0] ) )

# indices of computed focus
comp_focus_index = np.array([bx, ix, ix], dtype=int)
# indices of where focus should be
ideal_focus_index = np.array([mx, my, mz], dtype=int)
# distance between them
dist = np.linalg.norm(comp_focus_index - ideal_focus_index) * dx * 1e3

px = mx * dx * 1e3
py = mx * dy * 1e3
pz = mz * dz * 1e3
if verbose:
    msg = 'Max Pressure = ' + str(max_pressure * 1e-6) + ' MPa'
    logger.info(str(msg))
    msg = 'MI = ' + str(MI)
    logger.info(str(msg))
    msg = 'Coordinates of max pressure: (' + str(mx) + ', ' + str(my) + ', ' + str(mz) + ').'
    logger.info(str(msg))
    msg = 'Location of max pressure: (' + str(px) + ', ' + str(py) + ', ' + str(pz) + ').'
    logger.info(str(msg))
    msg = 'Isppa = ' + str(Isppa) + ' W/cm2'
    logger.info(str(msg))
    msg = 'Ispta = ' + str(Ispta * 1e3) + ' mW/cm2'
    logger.info(str(msg))

if doPlotting:
    # plot pressure on beam axis
    fig1, ax1 = plt.subplots()
    ax1.plot(x_vec, amp_on_axis, color='blue', marker='.', linestyle='-', linewidth=2, markersize=12)

    ax1.set(xlabel=r'Axial Position [mm]',
            ylabel=r'Pressure [MPa]',
            title=r'Axial Pressure')
    ax1.grid(True)
    if savePlotting:
        fig1.savefig("axial.png")

    # plot the pressure field at mid point along z axis
    fig2, ax2 = plt.subplots()
    extent = [np.min(x_vec), np.max(x_vec),
              np.min(y_vec), np.max(y_vec)]
    ax2.imshow(np.rot90(p[:, :, int(Nz / 2)]),
               aspect='auto',
               interpolation='none',
               extent=extent,
               origin='lower',
               cmap='turbo')
    ax2.set(xlabel=r'$x$ [mm]',
            ylabel=r'$y$ [mm]',
            title=r'Pressure Field')
    ax2.grid(False)

    fig3, ax3 = plt.subplots()
    extent = [np.min(x_vec), np.max(x_vec),
              np.min(y_vec), np.max(y_vec)]
    ax3.imshow(p[:, int(Ny / 2), :],
               aspect='auto',
               interpolation='none',
               extent=extent,
               origin='lower',
               cmap='turbo')
    ax3.set(xlabel=r'Axial Position [mm]',
            ylabel=r'Lateral Position [mm]',
            title=r'Pressure Field')
    ax3.grid(False)

    plt.show()
