import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import cv2 
import sys
import os
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from copy import deepcopy

from kwave.data import Vector
from kwave.utils.conversion import cart2grid
from kwave.utils.io import load_image
from kwave.utils.filters import smooth
from kwave.utils.interp import interp_cart_data
from kwave.utils.conversion import grid2cart
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.mapgen import make_cart_circle, make_circle
from kwave.utils.signals import add_noise, reorder_binary_sensor_data
from kwave.utils.colormap import get_color_map
from kwave.utils.matrix import resize, sort_rows
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D, kspaceFirstOrder2DC
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.reconstruction.time_reversal import TimeReversal

pml_size: int = 20 # size of the PML in grid points
Nx: int = 256 - 2 * pml_size # number of grid points in the x direction
Ny: int = 256 - 2 * pml_size # number of grid points in the y direction

x: float = 10e-3 # total grid size [m]
y: float = 10e-3 # total grid size [m]

dx: float = x / float(Nx) # grid point spacing in the x direction [m]
dy: float = y / float(Ny) # grid point spacing in the y direction [m]

kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

medium = kWaveMedium(sound_speed=1500.0)

kgrid.makeTime(medium.sound_speed)

p0_magnitude = 2.0
p0 = p0_magnitude * load_image('tests/EXAMPLE_source_two.bmp', is_gray=True)

p0 = resize(p0, [Nx, Ny])
p0 = smooth(p0, True)

source = kSource()
source.p0 = p0


# =========================================================================
# DEFINE THE MATERIAL PROPERTIES
# =========================================================================

sensor = kSensor()

# define a centered Cartesian circular sensor
sensor_radius: float = 4.5e-3            # [m]
sensor_angle: float = 3.0 * np.pi / 2.0  # [rad]
sensor_pos = Vector([0, 0])              # [m]
num_sensor_points: int = 70
cart_sensor_mask = make_cart_circle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle)

# put the cartesian points into the sensor.mask
sensor.mask = cart_sensor_mask

# set the record type: record the pressure waveform
sensor.record = ['p']

# =========================================================================
# DEFINE THE SIMULATION PARAMETERS
# =========================================================================

DATA_CAST = 'single'
DATA_PATH = './'

input_filename = 'input.h5'
output_filename = 'output.h5'

# set input options
# options for writing to file, but not doing simulations
simulation_options = SimulationOptions(
    data_cast=DATA_CAST,
    data_recast=True,
    save_to_disk=True,
    smooth_p0=False,
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
# RUN THE FIRST SIMULATION
# =========================================================================

sensor_data_original = kspaceFirstOrder2D(medium=deepcopy(medium),
                                          kgrid=deepcopy(kgrid),
                                          source=deepcopy(source),
                                          sensor=deepcopy(sensor),
                                          simulation_options=deepcopy(simulation_options),
                                          execution_options=deepcopy(execution_options))

cmap = get_color_map()

grid, _, reorder_index = cart2grid(kgrid, cart_sensor_mask)

cart_sensor_mask_reordered = cart_sensor_mask[:, np.squeeze(reorder_index.T).astype(int)-1]

sensor_data_reordered = reorder_binary_sensor_data(sensor_data_original['p'].T, reorder_index=reorder_index)

fig, ax = plt.subplots(1, 1)
im = ax.pcolormesh(np.squeeze(kgrid.x_vec)* 1e3, 
                   np.squeeze(kgrid.y_vec)* 1e3, p0, shading='gouraud', cmap=cmap, vmin=-1, vmax=1)
ax.scatter(cart_sensor_mask[1, :] * 1e3, cart_sensor_mask[0, :] * 1e3, c='k', marker='o', s=8)
ax.yaxis.set_inverted(True)

# plot the simulated sensor data
fig, ax = plt.subplots(1, 1)
im = ax.pcolormesh(sensor_data_reordered,
                   shading='gouraud', cmap=cmap, vmin=-1, vmax=1)
ax.set_aspect('auto', adjustable='box')
ax.set_ylabel('Sensor')
ax.set_xlabel('Time')
ax.yaxis.set_inverted(True)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="3%", pad="2%")
cbar = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
cbar.ax.tick_params(labelsize=8)




# =========================================================================
# SECOND
# =========================================================================

# add noise to the recorded sensor data
signal_to_noise_ratio: float = 40.0	# [dB]

sensor_data = add_noise(sensor_data_original['p'].T, signal_to_noise_ratio, 'peak')


# create a second computation grid for the reconstruction to avoid the
# inverse crime
Nx: int = 300              # number of grid points in the x direction
Ny: int = 300              # number of grid points in the y direction
dx: float = x / float(Nx)  # grid point spacing in the x direction [m]
dy: float = y / float(Ny)  # grid point spacing in the y direction [m]
kgrid_recon = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

# use the same time array for the reconstruction
kgrid_recon.setTime(kgrid.Nt, kgrid.dt)


#######
source = kSource()

del sensor

mask, _, _ = cart2grid(kgrid_recon, cart_sensor_mask)

sensor = kSensor()

sensor.mask = mask
sensor.recorded_pressure = sensor_data

# set the record type: record the pressure waveform
sensor.record = ['p']

tr = TimeReversal(kgrid_recon, medium, sensor, compensation_factor=1.0)
p0_recon = tr(kspaceFirstOrder2D, simulation_options, execution_options)

fig, ax = plt.subplots()
im = plt.pcolormesh(np.squeeze(kgrid_recon.x_vec), 
                    np.squeeze(kgrid_recon.y_vec), p0_recon, 
                    cmap=cmap, vmin=-1, vmax=1)
ax.yaxis.set_inverted(True)
ax.set_title('Reconstructed Pressure Distribution')


# =========================================================================
# THIRD
# =========================================================================

# create a binary sensor mask of an equivalent continuous circle, 
# this has the enlarged number of sensor points
sensor_radius_grid_points: int = round(sensor_radius / kgrid_recon.dx)
binary_sensor_mask = make_circle(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]),
                                 sensor_radius_grid_points, sensor_angle)
binary_sensor_mask = binary_sensor_mask.astype(bool)

del sensor
sensor = kSensor()
sensor.mask = binary_sensor_mask
sensor.record = ['p']
sensor.recorded_pressure = interp_cart_data(kgrid=kgrid_recon,
                                            cart_sensor_data=sensor_data_reordered,
                                            cart_sensor_mask=cart_sensor_mask,
                                            binary_sensor_mask=binary_sensor_mask)

# sensor defines the source
tr = TimeReversal(kgrid_recon, medium, sensor, compensation_factor=1.0)
p0_recon_interp = tr(kspaceFirstOrder2D, simulation_options, execution_options)

fig, ax = plt.subplots()
im = plt.pcolormesh(np.squeeze(kgrid_recon.x_vec), 
                    np.squeeze(kgrid_recon.y_vec), p0_recon_interp, 
                    cmap=cmap, vmin=-1, vmax=1)
ax.yaxis.set_inverted(True)
ax.set_title('Reconstructed Pressure Distribution with Interpolation')

# plot a profile for comparison
slice_pos = 4.5e-3;  # [m] location of the slice from top of grid [m]
i = int(round(slice_pos / kgrid.dx))
j = int(round(slice_pos / kgrid_recon.dx)) 
fig, ax = plt.subplots()
ax.plot(np.squeeze(kgrid.y_vec) * 1e3, p0[i,: ], 'k--', label='Initial Pressure')
ax.plot(np.squeeze(kgrid_recon.y_vec) * 1e3, np.transpose(p0_recon)[:, j], 'r-', label='Point Reconstruction')
ax.plot(np.squeeze(kgrid_recon.y_vec) * 1e3, np.transpose(p0_recon_interp)[:, j], 'b-', label='Interpolated Reconstruction')
ax.set_xlabel('y-position [mm]')
ax.set_ylabel('Pressure')
ax.set_ylim(0, 2.1)
ax.legend()