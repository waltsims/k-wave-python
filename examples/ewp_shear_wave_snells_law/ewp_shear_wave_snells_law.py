import os
import numpy as np
import matplotlib.pyplot as plt
from operator import not_
from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.pstdElastic2D import pstd_elastic_2d

from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# change scale to 2 to reproduce higher resolution figures in help file
scale: int = 1

# create the computational grid
PML_size: int = 10                    # [grid points]
Nx: int = 128 * scale - 2 * PML_size  # [grid points]
Ny: int = 192 * scale - 2 * PML_size  # [grid points]
dx: float = 0.5e-3 / float(scale)     # [m]
dy: float = 0.5e-3 / float(scale)     # [m]

kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

# define the medium properties for the top layer
cp1 = 1540.0      # compressional wave speed [m/s]
cs1 = 0.0         # shear wave speed [m/s]
rho1 = 1000.0     # density [kg/m^3]
alpha0_p1 = 0.1   # compressional absorption [dB/(MHz^2 cm)]
alpha0_s1  = 0.1  # shear absorption [dB/(MHz^2 cm)]

# define the medium properties for the bottom layer
cp2 = 3000.0     # compressional wave speed [m/s]
cs2 = 1400.0     # shear wave speed [m/s]
rho2 = 1850.0    # density [kg/m^3]
alpha0_p2 = 1.0  # compressional absorption [dB/(MHz^2 cm)]
alpha0_s2 = 1.0  # shear absorption [dB/(MHz^2 cm)]

# create the time array
cfl = 0.1
t_end = 60e-6
kgrid.makeTime(cp1, cfl, t_end)

# define position of heterogeneous slab
slab = np.zeros((Nx, Ny), dtype=bool)
slab[Nx // 2 : -1, :] = True

# define the source geometry in SI units (where 0, 0 is the grid center)
arc_pos = [-15e-3, -25e-3]  # [m]
focus_pos = [5e-3, 5e-3]    # [m]
radius = 25e-3              # [m]
diameter = 20e-3            # [m]

# define the driving signal
source_freq = 500e3    # [Hz]
source_strength = 1e6  # [Pa]
source_cycles = 3      # number of tone burst cycles

# convert the source parameters to grid points
arc_pos = np.rint(np.asarray(arc_pos) / dx) + np.asarray([Nx / 2, Ny / 2]).astype(int)
focus_pos = np.rint(np.asarray(focus_pos) / dx) + np.asarray([Nx / 2, Ny / 2]).astype(int)
radius_pos = int(round(radius / dx))
diameter_pos = int(round(diameter / dx))

# force the diameter to be odd
if (np.isclose(rem(diameter_pos, 2), 0.0) ):
    diameter_pos = diameter_pos + 1

# generate the source geometry
source_mask = make_arc(Vector([Nx, Ny]), np.asarray(arc_pos), radius_pos, diameter_pos, Vector(focus_pos))

fs = 1.0 / kgrid.dt
signal = tone_burst(fs, source_freq, source_cycles, envelope="Gaussian", plot_signal=False, signal_length=0, signal_offset=0)

# =========================================================================
# FLUID SIMULATION
# =========================================================================

# assign the medium properties
sound_speed = cp1 * np.ones((Nx, Ny))
density = rho1 * np.ones((Nx, Ny))
alpha_coeff = alpha0_p1 * np.ones((Nx, Ny))
alpha_power = 2.0

sound_speed[slab] = cp2
density[slab] = rho2
alpha_coeff[slab] = alpha0_p2

medium = kWaveMedium(sound_speed,
                        density=density,
                        alpha_coeff=alpha_coeff,
                        alpha_power=alpha_power)

# define the sensor to record the maximum particle velocity everywhere
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)
sensor.record = ['u_max_all']

# assign the source
source = kSource()
source.p_mask = source_mask
source.p = source_strength * signal

# set the input settings
input_filename_p = './data_p_input.h5'
output_filename_p = './data_p_output.h5'

DATA_CAST: str = 'single'

DATA_PATH = 'data' + os.sep

RUN_SIMULATION = True

# options for writing to file, but not doing simulations
simulation_options = SimulationOptions(data_cast=DATA_CAST,
                                       data_recast=True,
                                       save_to_disk=True,
                                       input_filename=input_filename_p,
                                       output_filename=output_filename_p,
                                       save_to_disk_exit=not_(RUN_SIMULATION),
                                       data_path=DATA_PATH,
                                       pml_inside=False,
                                       pml_size=PML_size,
                                       hdf_compression_level='lzf')

execution_options = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False)

# run the fluid simulation
sensor_data_fluid = kspace_first_order_2d_gpu(medium=deepcopy(medium),
                                              kgrid=deepcopy(kgrid),
                                              source=deepcopy(source),
                                              sensor=deepcopy(sensor),
                                              simulation_options=deepcopy(simulation_options),
                                              execution_options=deepcopy(execution_options))

# =========================================================================
# ELASTIC SIMULATION
# =========================================================================

# set the input settings
input_filename_e = './data_e_input.h5'
output_filename_e = './data_e_output.h5'

# define the medium properties
sound_speed_compression = cp1 * np.ones((Nx, Ny))
sound_speed_shear = cs1 * np.ones((Nx, Ny))
density = rho1 * np.ones((Nx, Ny))
alpha_coeff_compression = alpha0_p1 * np.ones((Nx, Ny))
alpha_coeff_shear  = alpha0_s1 * np.ones((Nx, Ny))

sound_speed_compression[slab] = cp2
sound_speed_shear[slab] = cs2
density[slab] = rho2
alpha_coeff_compression[slab] = alpha0_p2
alpha_coeff_shear[slab] = alpha0_s2

medium_e = kWaveMedium(sound_speed_compression,
                       sound_speed_compression=sound_speed_compression,
                       density=density,
                       alpha_coeff=alpha_coeff_compression,
                       alpha_power=2.0,
                       sound_speed_shear=sound_speed_shear,
                       alpha_coeff_shear=alpha_coeff_shear)

# assign the source
source_e = kSource()
source_e.s_mask = source_mask
source_e.sxx = -source_strength * signal
source_e.syy = source.sxx

simulation_options_e = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                         data_cast=DATA_CAST,
                                         data_recast=True,
                                         save_to_disk=True,
                                         input_filename=input_filename_e,
                                         output_filename=output_filename_e,
                                         save_to_disk_exit=not_(RUN_SIMULATION),
                                         data_path=DATA_PATH,
                                         pml_inside=False,
                                         pml_size=PML_size,
                                         hdf_compression_level='lzf')

# run the elastic simulation
sensor_data_elastic = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                      source=deepcopy(source_e),
                                      sensor=deepcopy(sensor),
                                      medium=deepcopy(medium_e),
                                      simulation_options=deepcopy(simulation_options_e))

# =========================================================================
# VISUALISATION
# =========================================================================

# define plot vector- convert to cm
x_vec = kgrid.x_vec * 1e3
y_vec = kgrid.y_vec * 1e3

# calculate square of velocity magnitude for fluid and elastic simulations
u_f = sensor_data_fluid.ux_max_all**2 + sensor_data_fluid.uy_max_all**2
log_f = 20.0 * np.log10(u_f / np.max(u_f))
u_e = sensor_data_elastic.ux_max_all**2 + sensor_data_elastic.uy_max_all**2
log_e = 20.0 * np.log10(u_e / np.max(u_e))

# plot layout
fig1, ax1 = plt.subplots(nrows=1, ncols=1)
_ = ax1.pcolormesh(kgrid.y.T, kgrid.x.T, np.logical_or(slab, source_mask), cmap='gray_r', shading='gouraud', alpha=0.5)

# plot velocities
fig2, (ax2a, ax2b) = plt.subplots(nrows=2, ncols=1)
_ = ax2a.pcolormesh(kgrid.y.T, kgrid.x.T, log_f, cmap='viridis', shading='gouraud', alpha=0.5)
_ = ax2b.pcolormesh(kgrid.y.T, kgrid.x.T, log_e, cmap='viridis', shading='gouraud', alpha=0.5)