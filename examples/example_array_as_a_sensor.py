import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import make_cart_circle, make_disc

# create empty array
karray = kWaveArray()

# define arc properties
radius = 100e-3  # [m]
diameter = 8e-3  # [m]
ring_radius = 50e-3  # [m]
num_elements = 20

# orient all elements towards the center of the grid
focus_pos = Vector([0, 0])  # [m]

element_pos = make_cart_circle(ring_radius, num_elements, focus_pos)

for idx in range(num_elements):
    karray.add_arc_element(element_pos[:, idx], radius, diameter, focus_pos)

# grid properties
Nx = 256
dx = 0.5e-3
Ny = 256
dy = 0.5e-3
kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

# medium properties
medium = kWaveMedium(sound_speed=1500)

# time array
kgrid.makeTime(medium.sound_speed)

source = kSource()
source.p0 = make_disc(Vector([Nx, Ny]), Vector([Nx / 4 + 20, Ny / 4]), 4)

sensor = kSensor()
sensor.mask = element_pos
simulation_options = SimulationOptions(
    save_to_disk=True,
    data_cast='single',
)

execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
sensor_data_point = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)

# assign binary mask from karray to the source mask
sensor.mask = karray.get_array_binary_mask(kgrid)

sensor_data = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)
combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data['p'])

# =========================================================================
# VISUALIZATION
# =========================================================================

# create pml mask (default size in 2D is 20 grid points)
pml_size = 20
pml_mask = np.zeros((Nx, Ny), dtype=bool)
pml_mask[:pml_size, :] = 1
pml_mask[:, :pml_size] = 1
pml_mask[-pml_size:, :] = 1
pml_mask[:, -pml_size:] = 1

# plot source and pml masks
plt.figure()
plt.imshow(source_p_mask | pml_mask, extent=[kgrid.x_vec[0], kgrid.x_vec[-1], kgrid.y_vec[0], kgrid.y_vec[-1]],
           aspect='auto', cmap='gray')
plt.xlabel('x-position [m]')
plt.ylabel('y-position [m]')
plt.title('Source and PML Masks')
plt.show()

# overlay the physical source positions
plt.figure()
karray.plot_array(show=True)
