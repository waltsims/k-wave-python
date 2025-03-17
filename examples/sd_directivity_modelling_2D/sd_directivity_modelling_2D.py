import os
from copy import deepcopy
from tempfile import gettempdir

import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.colormap import get_color_map
from kwave.utils.conversion import cart2grid
from kwave.utils.data import scale_SI
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_cart_circle
from kwave.utils.matlab import ind2sub, matlab_find, unflatten_matlab_mask

grid_size_points = Vector([128, 128])  # [grid points]
grid_size_meters = Vector([50e-3, 50e-3])  # [m]
grid_spacing_meters = grid_size_meters / grid_size_points  # [m]
kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# define the array of time points [s]
Nt = 350
dt = 7e-8  # [s]
kgrid.setTime(Nt, dt)

sz = 20  # [grid points]
sensor_mask = np.zeros(grid_size_points)
sensor_mask[grid_size_points.x // 2, (grid_size_points.y // 2 - sz // 2) : (grid_size_points.y // 2 + sz // 2) + 1] = 1
sensor = kSensor(sensor_mask)


# define equally spaced point sources lying on a circle centred at the
# centre of the detector face
radius = 30  # [grid points]
points = 11
circle = make_cart_circle(radius * grid_spacing_meters.x, points, Vector([0, 0]), np.pi)

# find the binary sensor mask most closely corresponding to the Cartesian
# coordinates from makeCartCircle
circle, _, _ = cart2grid(kgrid, circle)

# find the indices of the sources in the binary source mask
source_positions = matlab_find(circle, val=1, mode="eq")

source = kSource()
source_freq = 0.25e6  # [Hz]
source_mag = 1  # [Pa]
source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

# filter the source to remove high frequencies not supported by the grid
source.p = filter_time_series(kgrid, medium, source.p)

# pre-allocate array for storing the output time series
single_element_data = np.zeros((Nt, points))  # noqa: F841

# run a simulation for each of these sources to see the effect that the
# angle from the detector has on the measured signal
for source_loop in range(points):
    # select a point source
    source.p_mask = np.zeros(grid_size_points)
    source.p_mask[unflatten_matlab_mask(source.p_mask, source_positions[source_loop] - 1)] = 1

    # run the simulation

    input_filename = f"example_input_{source_loop + 1}_input.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(save_to_disk=True, input_filename=input_filename, data_path=pathname)
    # run the simulation
    sensor_data = kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=deepcopy(sensor),
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )
    single_element_data[:, source_loop] = sensor_data["p"].sum(axis=1)

plt.figure()
plt.imshow(
    circle + sensor.mask,
    extent=[kgrid.y_vec.min() * 1e3, kgrid.y_vec.max() * 1e3, kgrid.x_vec.max() * 1e3, kgrid.x_vec.min() * 1e3],
    vmin=-1,
    vmax=1,
    cmap=get_color_map(),
)
plt.ylabel("x-position [mm]")
plt.xlabel("y-position [mm]")
plt.axis("image")

_, t_scale, t_prefix, _ = scale_SI(kgrid.t_array[-1])

# Plot the time series recorded for each of the sources
plt.figure()
# Loop through each source and plot individually
for i in range(single_element_data.shape[1]):
    plt.plot((kgrid.t_array * t_scale).squeeze(), single_element_data[:, i], label=f"Source {i+1}")

plt.xlabel(f"Time [{t_prefix}s]")
plt.ylabel("Pressure [au]")
plt.title("Time Series For Each Source Direction")


# Calculate angle between source and center of detector face
angles = []
for source_position in source_positions:
    x, y = ind2sub(kgrid.y.shape, source_position)
    angles.append(np.arctan(kgrid.y[x, y] / kgrid.x[x, y]))

# Plot the maximum amplitudes for each of the sources
plt.figure()
plt.plot(angles, np.max(single_element_data[200:350, :], axis=0), "o", mfc="none")
plt.xlabel("Angle Between Source and Centre of Detector Face [rad]")
plt.ylabel("Maximum Detected Pressure [au]")
plt.show()
