import numpy as np

# noinspection PyUnresolvedReferences
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.io import load_image
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matrix import resize

# assign the grid size and create the computational grid
pml_size = Vector([20, 20])  # [grid points]
grid_size_points = Vector([256, 256]) - 2 * pml_size  # [grid points]
grid_size_meters = Vector([10e-3, 10e-3])  # [m]
grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

# load the initial pressure distribution from an image and scale
p0_magnitude = 2
p0 = p0_magnitude * load_image('EXAMPLE_source_two.bmp', is_gray=True)

# resize the input image to the desired number of grid points
p0 = resize(p0, grid_size_points)

# smooth the initial pressure distribution and restore the magnitude
p0 = smooth(p0, True)

# assign to the source structure
source = kSource()
source.p0 = p0

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# define a centered Cartesian circular sensor
sensor_radius = 4.5e-3              # [m]
sensor_angle = 3 * np.pi / 2        # [rad]
sensor_pos = Vector([0, 0])         # [m]
num_sensor_points = 70
cart_sensor_mask = make_cart_circle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle)

# assign to sensor structure
sensor_mask = cart_sensor_mask
sensor = kSensor(sensor_mask)

# create the time array
kgrid.makeTime(medium.sound_speed)

# set the input settings
simulation_options = SimulationOptions(
    pml_inside=False,
    smooth_p0=False,
    save_to_disk=True,
)

# run the simulation
simulation_output = kspaceFirstOrder2D(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=SimulationExecutionOptions(is_gpu_simulation=True)
)


sensor_data = simulation_output['p']

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from kwave.utils.interp import interp_cart_data

from kwave.utils.mapgen import make_circle
from kwave.utils.signals import add_noise



# %%

signal_to_noise_ratio = 40  # [dB]

sensor_data_with_noise = add_noise(sensor_data, signal_to_noise_ratio)


# Creating a second computation grid for the reconstruction
N_recon = Vector([300, 300])  # number of grid points in the x direction

grid_spacing_recon_meters = grid_size_meters / N_recon  # grid point spacing [m]

kgrid_recon = kWaveGrid(grid_size_points, grid_spacing_meters)


# Use the same time array for the reconstruction

# Assuming setTime is a method to set the time array, and kgrid has attributes Nt and dt

kgrid_recon.setTime(kgrid.Nt, kgrid.dt)



# Reset the initial pressure
source.p0 = np.array([])

# Assign the time reversal data
sensor.time_reversal_boundary_data = sensor_data_with_noise

# Run the time-reversal reconstruction
p0_recon = kspaceFirstOrder2D(kgrid=kgrid_recon, medium=medium, source=source, sensor=sensor,
    simulation_options=simulation_options,
    execution_options=SimulationExecutionOptions(is_gpu_simulation=True))

# Creating a binary sensor mask of an equivalent continuous circle
sensor_radius_grid_points = round(sensor_radius / kgrid_recon.dx)

binary_sensor_mask = make_circle(kgrid_recon.Nx, kgrid_recon.Ny, kgrid_recon.Nx//2 + 1, kgrid_recon.Ny//2 + 1, sensor_radius_grid_points, sensor_angle)

# Assign to sensor structure

sensor['mask'] = binary_sensor_mask

# Interpolate data to remove the gaps and assign to sensor structure
# Assuming interpCartData is a function you have defined for interpolation
sensor.time_reversal_boundary_data = interp_cart_data(kgrid_recon, sensor_data_with_noise, cart_sensor_mask, binary_sensor_mask)

# Run the time-reversal reconstruction again with interpolated data
p0_recon_interp = kspaceFirstOrder2D(kgrid=kgrid_recon, medium=medium, source=source, sensor=sensor, simulation_options=simulation_options, execution_options=SimulationExecutionOptions(is_gpu_simulation=True))

import numpy as np
import matplotlib.pyplot as plt
from kwave.utils.colormap import get_color_map
from kwave.utils.conversion import cart2grid

(grid_data, _, _) = cart2grid(kgrid, cart_sensor_mask)

# Plot the initial pressure and sensor distribution
plt.figure()
plt.imshow(p0 + grid_data, extent=(kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.min() * 1e3, kgrid.x_vec.max() * 1e3), cmap=get_color_map(), vmin=-1, vmax=1)
plt.colorbar(label='Pressure')
plt.ylabel('x-position [mm]')
plt.xlabel('y-position [mm]')
plt.title('Initial Pressure and Sensor Distribution')
plt.axis('image')

# Plot the simulated sensor data
plt.figure()
plt.imshow(sensor_data, cmap=get_color_map(), aspect='auto', extent=[0, sensor_data.shape[1], 0, sensor_data.shape[0]], vmin=-1, vmax=1)
plt.colorbar(label='Pressure')
plt.ylabel('Sensor Position')
plt.xlabel('Time Step')
plt.title('Simulated Sensor Data')

# ## Time reveral is not yet supported

# # Plot the reconstructed initial pressure
# plt.figure()
# plt.imshow(p0_recon, extent=(kgrid_recon.y_vec.min() * 1e3, kgrid_recon.y_vec.max() * 1e3, kgrid_recon.x_vec.min() * 1e3, kgrid_recon.x_vec.max() * 1e3), cmap=get_colormap(), vmin=-1, vmax=1)
# plt.colorbar(label='Pressure')
# plt.ylabel('x-position [mm]')
# plt.xlabel('y-position [mm]')
# plt.title('Reconstructed Initial Pressure')
# plt.axis('image')

# # Plot the reconstructed initial pressure using the interpolated data
# plt.figure()
# plt.imshow(p0_recon_interp, extent=(kgrid_recon.y_vec.min() * 1e3, kgrid_recon.y_vec.max() * 1e3, kgrid_recon.x_vec.min() * 1e3, kgrid_recon.x_vec.max() * 1e3), cmap=get_colormap(), vmin=-1, vmax=1)
# plt.colorbar(label='Pressure')
# plt.ylabel('x-position [mm]')
# plt.xlabel('y-position [mm]')
# plt.title('Reconstructed Initial Pressure with Interpolated Data')
# plt.axis('image')

# # Plot a profile for comparison
# slice_index = round(slice_pos / kgrid.dx)
# slice_index_recon = round(slice_pos / kgrid_recon.dx)
# plt.figure()
# plt.plot(kgrid.y_vec * 1e3, p0[slice_index, :], 'k--', label='Initial Pressure')
# plt.plot(kgrid_recon.y_vec * 1e3, p0_recon[slice_index_recon, :], 'r-', label='Point Reconstruction')
# plt.plot(kgrid_recon.y_vec * 1e3, p0_recon_interp[slice_index_recon, :], 'b-', label='Interpolated Reconstruction')
# plt.xlabel('y-position [mm]')
# plt.ylabel('Pressure')
# plt.legend()
# plt.axis('tight')
# plt.ylim([0, 2.1])
# plt.title('Pressure Profile Comparison')


