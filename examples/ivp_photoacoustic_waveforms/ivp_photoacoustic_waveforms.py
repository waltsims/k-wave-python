import numpy as np
import matplotlib.pyplot as plt

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.data import scale_SI
from kwave.utils.mapgen import make_disc, make_ball
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options import SimulationOptions, SimulationExecutionOptions

# number of grid points in the x (row) direction
Nx: int = 64

# size of the domain in the x direction [m]
x: float = 1e-3

# grid point spacing in the x direction [m]
dx: float = x / Nx

# sound speed [m/s]
sound_speed: float = 1500

# size of the initial pressure distribution
source_radius: int = 2  # [grid points]

# distance between the centre of the source and the sensor
source_sensor_distance: int = 10  # [grid points]

# time array
dt: float = 2e-9  # [s]
t_end: float = 300e-9  # [s]

# # create the medium
# medium1 = kWaveMedium(sound_speed=sound_speed)

# # create the computational grid
# kgrid1 = kWaveGrid([Nx], [dx])

# # create the time array
# kgrid1.setTime(np.round(t_end / dt), dt)

# # create initial pressure distribution
# source1 = kSource()
# source1.p0[Nx//2 - source_radius:Nx//2 + source_radius] = 1.0

# # define a single sensor point
# sensor1 = kSensor()
# sensor1.mask = np.zeros((Nx,), dtype=bool)
# sensor1.mask[Nx // 2 + source_sensor_distance] = True

# simulation_options1 = SimulationOptions(
#     data_cast='single',
#     save_to_disk=True)

# execution_options1 = SimulationExecutionOptions(
#     is_gpu_simulation=True,
#     delete_data=False,
#     verbose_level=2)

# # run the simulation
# sensor_data_1D = kspaceFirstOrder1D(
#     medium=medium1,
#     kgrid=kgrid1,
#     source=source1,
#     sensor=sensor1,
#     simulation_options=simulation_options1,
#     execution_options=execution_options1)

#######

# medium
medium2 = kWaveMedium(sound_speed=1500)
# create the k-space grid
kgrid2 = kWaveGrid([Nx, Nx], [dx, dx])

# create the time array using an integer number of points per period
Nt = int(np.round(t_end / dt))
kgrid2.setTime(Nt, dt)

# create instance of a sensor
sensor2 = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor2.mask = np.zeros((Nx, Nx), dtype=bool)

# define a single sensor point
sensor2.mask[Nx // 2 + source_sensor_distance, Nx // 2] = True

# set the record type: record the pressure waveform
sensor2.record = ["p"]

# make a source object
source2 = kSource()
source2.p0 = make_disc(Vector([Nx, Nx]), Vector([Nx // 2, Nx // 2]), source_radius, plot_disc=False)

simulation_options2 = SimulationOptions(data_cast="single", save_to_disk=True)

execution_options2 = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False, verbose_level=2)

# run the simulation
sensor_data_2D = kspaceFirstOrder2D(
    medium=medium2,
    kgrid=kgrid2,
    source=source2,
    sensor=sensor2,
    simulation_options=simulation_options2,
    execution_options=execution_options2,
)

############

# medium
medium3 = kWaveMedium(sound_speed=1500)

# create the k-space grid
kgrid3 = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])

# create the time array using an integer number of points per period
kgrid3.setTime(int(np.round(t_end / dt)), dt)

# create instance of a sensor
sensor3 = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor3.mask = np.zeros((Nx, Nx, Nx), dtype=bool)

# define a single sensor point
sensor3.mask[Nx // 2 + source_sensor_distance, Nx // 2, Nx // 2] = True

# set the record type: record the pressure waveform
sensor3.record = ["p"]

# make a source object
source3 = kSource()
source3.p0 = make_ball(Vector([Nx, Nx, Nx]), Vector([Nx // 2, Nx // 2, Nx // 2]), source_radius)

simulation_options3 = SimulationOptions(data_cast="single", save_to_disk=True)

execution_options3 = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False, verbose_level=2)

# run the simulation
sensor_data_3D = kspaceFirstOrder3D(
    medium=medium3,
    kgrid=kgrid3,
    source=source3,
    sensor=sensor3,
    simulation_options=simulation_options3,
    execution_options=execution_options3,
)

# plot the simulations
t_sc, t_scale, t_prefix, _ = scale_SI(t_end)
_, ax1 = plt.subplots()
ax1.plot(np.squeeze(kgrid2.t_array * t_scale), sensor_data_2D["p"] / np.max(np.abs(sensor_data_2D["p"])), "r-", label="2D")
ax1.plot(np.squeeze(kgrid3.t_array * t_scale), sensor_data_3D["p"] / np.max(np.abs(sensor_data_3D["p"])), "k-", label="3D")
ax1.set(xlabel=f"Time [{t_prefix}s]", ylabel="Recorded Pressure [au]")
ax1.grid(True)
ax1.legend(loc="upper right")
plt.show()
