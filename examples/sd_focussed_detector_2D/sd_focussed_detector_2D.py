# # Focussed Detector In 2D Example
# This example shows how k-Wave-python can be used to model the output of a focused semicircular detector, where the directionality arises from spatially averaging across the detector surface. Unlike the original example in k-Wave, this example does not visualize the simulation, as this functionality is not intrinsically supported by the accelerated binaries.

import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.ktransducer import kSensor
from kwave.utils.data import scale_SI
from kwave.utils.mapgen import make_circle, make_disc

# In[3]:


# create the computational grid
grid_size = Vector([180, 180])  # [grid points]
grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
kgrid = kWaveGrid(grid_size, grid_spacing)

# define the properties of the propagation medium
medium = kWaveMedium(sound_speed=1500)

# define a sensor as part of a circle centred on the grid
sensor_radius = 65  # [grid points]
arc_angle = np.pi  # [rad]
sensor_mask = make_circle(grid_size, grid_size // 2 + 1, sensor_radius, arc_angle)
sensor = kSensor(sensor_mask)

# define the array of temporal points
t_end = 11e-6  # [s]
_ = kgrid.makeTime(medium.sound_speed, t_end=t_end)


## Run simulation with first source
# place a disc-shaped source near the focus of the detector
source = kSource()
source.p0 = 2 * make_disc(grid_size, grid_size / 2, 4)

# run the simulation
sensor_data1 = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    backend="cpp",
    device="cpu",
)


sensor_data1["p"].shape


## Run simulation with second source


# place a disc-shaped source horizontally shifted from the focus of the detector
source.p0 = 2 * make_disc(grid_size, grid_size / 2 + [0, 20], 4)

sensor_data2 = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    backend="cpp",
    device="cpu",
)


## Visualize recorded data


sensor_output1 = np.sum(sensor_data1["p"], axis=1) / np.sum(sensor.mask)
sensor_output2 = np.sum(sensor_data2["p"], axis=1) / np.sum(sensor.mask)

t_sc, t_scale, t_prefix, _ = scale_SI(t_end)
t_array = kgrid.t_array.squeeze() * t_scale

plt.plot(t_array, sensor_output1, "k")
plt.plot(t_array, sensor_output2, "r")

plt.xlabel("Time [" + t_prefix + "s]")
plt.ylabel("Average Pressure Measured Over Detector [au]")
plt.legend(
    [
        f"Source on focus, sum(output^2) = {round(np.sum(sensor_output1**2) * 100) / 100}",
        f"Source off focus, sum(output^2) = {round(np.sum(sensor_output2**2) * 100) / 100}",
    ]
)

plt.show()
