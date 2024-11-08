import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.utils.filters import smooth


# =========================================================================
# SET UP AND RUN THE SIMULATION
# =========================================================================

# define literals
NUMBER_OF_ITERATIONS = 3  # number of iterations
PML_SIZE = 20  # size of the perfectly matched layer in grid points

# create the computational grid
Nx = 128 - 2 * PML_SIZE  # number of grid points in the x direction
Ny = 256 - 2 * PML_SIZE  # number of grid points in the y direction
dx = 0.1e-3  # grid point spacing in the x direction [m]
dy = 0.1e-3  # grid point spacing in the y direction [m]
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)  # [m/s]

# create the time array
kgrid.makeTime(medium.sound_speed)

# load an image for the initial pressure distribution
p0_image = io.imread('EXAMPLE_k-Wave.png', as_gray=True)

# make it binary
p0_image = np.double(p0_image > 0)

# smooth and scale the initial pressure distribution
p0 = smooth(p0_image, restore_max=True)

# assign to the source structure
source = {'p0': p0}

# define an L-shaped sensor mask
sensor_mask = np.zeros((Nx, Ny))
sensor_mask[0, :] = 1
sensor_mask[:, 0] = 1
sensor = {'mask': sensor_mask}

# set the input arguments: force the PML to be outside the computational grid,
# switch off p0 smoothing within kspaceFirstOrder2D
input_args = {
    'PMLInside': False,
    'PMLSize': PML_SIZE,
    'Smooth': False,
    'PlotPML': False,
    'PlotSim': True
}

# run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, **input_args)

# =========================================================================
# RECONSTRUCT AN IMAGE USING TIME REVERSAL
# =========================================================================

# remove the initial pressure field used in the simulation
del source['p0']

# use the sensor points as sources in time reversal
source['p_mask'] = sensor['mask']

# time reverse and assign the data
source['p'] = np.fliplr(sensor_data)

# enforce, rather than add, the time-reversed pressure values
source['p_mode'] = 'dirichlet'

# set the simulation to record the final image (at t = 0)
sensor['record'] = ['p_final']

# run the time reversal reconstruction
p0_estimate = kspaceFirstOrder2D(kgrid, medium, source, sensor, **input_args)

# apply a positivity condition
p0_estimate['p_final'] = np.maximum(p0_estimate['p_final'], 0)

# store the latest image estimate
p0_1 = p0_estimate['p_final']

# =========================================================================
# ITERATE TO IMPROVE THE IMAGE
# =========================================================================

for loop in range(2, NUMBER_OF_ITERATIONS + 1):

    # remove the source used in the previous time reversal
    del source['p']

    # set the initial pressure to be the latest estimate of p0
    source['p0'] = p0_estimate['p_final']

    # set the simulation to record the time series
    if 'record' in sensor:
        del sensor['record']

    # calculate the time series using the latest estimate of p0
    sensor_data2 = kspaceFirstOrder2D(kgrid, medium, source, sensor, **input_args)

    # calculate the error in the estimated time series
    data_difference = sensor_data - sensor_data2

    # assign the data_difference as a time-reversal source
    source['p_mask'] = sensor['mask']
    source['p'] = np.fliplr(data_difference)
    del source['p0']
    source['p_mode'] = 'dirichlet'

    # set the simulation to record the final image (at t = 0)
    sensor['record'] = ['p_final']

    # run the time reversal reconstruction
    p0_update = kspaceFirstOrder2D(kgrid, medium, source, sensor, **input_args)

    # add the update to the latest image
    p0_estimate['p_final'] += p0_update['p_final']

    # apply a positivity condition
    p0_estimate['p_final'] = np.maximum(p0_estimate['p_final'], 0)

    # store the latest image estimate
    globals()[f'p0_{loop}'] = p0_estimate['p_final']

# =========================================================================
# VISUALISATION
# =========================================================================

# set the color scale
c_axis = [0, 1.1]

# plot the initial pressure
plt.figure()
plt.imshow(p0, cmap='gray', vmin=c_axis[0], vmax=c_axis[1])
plt.axis('image')
plt.xticks([])
plt.yticks([])
plt.title('Initial Acoustic Pressure')
plt.colorbar()
plt.show()

# plot the first iteration
plt.figure()
plt.imshow(p0_1, cmap='gray', vmin=c_axis[0], vmax=c_axis[1])
plt.axis('image')
plt.xlabel('y-position [mm]')
plt.ylabel('x-position [mm]')
plt.title('Time Reversal Reconstruction')
plt.colorbar()
plt.show()

# plot the 2nd iteration
plt.figure()
plt.imshow(p0_2, cmap='gray', vmin=c_axis[0], vmax=c_axis[1])
plt.axis('image')
plt.title('Time Reversal Reconstruction, 2 Iterations')
plt.colorbar()
plt.show()

# plot the 3rd iteration
plt.figure()
plt.imshow(p0_3, cmap='gray', vmin=c_axis[0], vmax=c_axis[1])
plt.axis('image')
plt.title('Time Reversal Reconstruction, 3 Iterations')
plt.colorbar()
plt.show()
