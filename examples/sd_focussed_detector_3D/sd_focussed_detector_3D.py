# Focussed Detector In 3D Example
# This example shows how k-Wave can be used to model the output of a focussed bowl detector where the directionality arises from spatially averaging across the detector surface.

import os
from copy import deepcopy
from tempfile import gettempdir

import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.ktransducer import kSensor
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.data import scale_SI
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_bowl

# create the computational grid
grid_size = Vector([64, 64, 64])  # [grid points]
grid_spacing_single = 100e-3 / grid_size.x
grid_spacing = Vector([grid_spacing_single, grid_spacing_single, grid_spacing_single])  # [m]
kgrid = kWaveGrid(grid_size, grid_spacing)

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# define the array of temporal points
_ = kgrid.makeTime(medium.sound_speed)

input_filename = "example_sd_focused_3d_input.h5"
pathname = gettempdir()
input_file_full_path = os.path.join(pathname, input_filename)
simulation_options = SimulationOptions(
    save_to_disk=True, input_filename=input_filename, data_path=pathname, pml_size=10, data_cast="single"
)

# create a concave sensor
sphere_offset = 10
diameter = grid_size.x / 2 + 1
radius = grid_size.x / 2
bowl_pos = Vector([1 + sphere_offset, grid_size.y / 2, grid_size.z / 2])
focus_pos = grid_size / 2

sensor_mask = make_bowl(grid_size, bowl_pos, radius, diameter, focus_pos)
sensor = kSensor(sensor_mask)

# define a time varying sinusoidal source
source = kSource()

source_freq = 0.25e6  # [Hz]
source_mag = 1  # [Pa]
source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)
source.p = filter_time_series(kgrid, medium, source.p)

# place the first point source near the focus of the detector
source1 = np.zeros(grid_size)
source1[int(sphere_offset + radius), grid_size.y // 2 - 1, grid_size.z // 2 - 1] = 1

# run the first simulation
source.p_mask = source1

sensor_data1 = kspaceFirstOrder3D(
    medium=medium,
    kgrid=kgrid,
    source=deepcopy(source),
    sensor=deepcopy(sensor),
    simulation_options=simulation_options,
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
)

# average the data recorded at each grid point to simulate the measured signal from a single element focused detector
sensor_data1 = np.sum(sensor_data1["p"], axis=1)

# place the second point source off axis
source2 = np.zeros(grid_size)
source2[int(1 + sphere_offset + radius), grid_size.y // 2 + 5, grid_size.z // 2 + 5] = 1


# run the second simulation
source.p_mask = source2
sensor_data2 = kspaceFirstOrder3D(
    medium=medium,
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
)

# average the data recorded at each grid point to simulate the measured signal from a single element focused detector
sensor_data2 = np.sum(sensor_data2["p"], axis=1)

# Combine arrays as in MATLAB: sensor.mask + source1 + source2
combined_array = sensor_mask + source1 + source2

# Find the indices of non-zero elements
x, y, z = np.nonzero(combined_array)

# Enable interactive mode
plt.ion()

# Create an interactive 3D plot with matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(x, y, z, c="blue", marker="s", s=100, depthshade=True)  # 's' for square-like markers

# Set the view angle to mimic MATLAB's `view([130, 40])`
ax.view_init(elev=40, azim=130)

# Customize the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Interactive 3D Voxel Plot")

t_sc, t_scale, t_prefix, _ = scale_SI(kgrid.t_array[-1])
t_array = kgrid.t_array.squeeze() * t_scale

plt.figure()
plt.plot(t_array, sensor_data1)
plt.plot(t_array, sensor_data2, "r")

plt.xlabel("Time [" + t_prefix + "s]")
plt.ylabel("Average Pressure Measured By Focussed Detector [Pa]")
plt.legend(["Source on axis", "Source off axis"])

plt.show()
