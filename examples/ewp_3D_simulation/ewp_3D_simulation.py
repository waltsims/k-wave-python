
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib.animation import FuncAnimation
from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.pstdElastic3D import pstd_elastic_3d
from kwave.reconstruction.beamform import focus

# from kwave.utils.dotdictionary import dotdict
# from kwave.utils.mapgen import make_disc, make_circle
from kwave.utils.signals import tone_burst

# from kwave.utils.colormap import get_color_map

from kwave.options.simulation_options import SimulationOptions, SimulationType



"""
Simulations In Three Dimensions Example

This example provides a simple demonstration of using k-Wave to model
elastic waves in a three-dimensional heterogeneous propagation medium. It
builds on the Explosive Source In A Layered Medium and Simulations In
Three-Dimensions examples.

author: Bradley Treeby
date: 14th February 2014
last update: 29th May 2017

This function is part of the k-Wave Toolbox (http://www.k-wave.org)
Copyright (C) 2014-2017 Bradley Treeby

This file is part of k-Wave. k-Wave is free software: you can
redistribute it and/or modify it under the terms of the GNU Lesser
General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
"""


# =========================================================================
# SIMULATION
# =========================================================================

# create the computational grid
pml_size: int = 10

Nx: int = 64       # number of grid points in the x (row) direction
Ny: int = 64       # number of grid points in the y (column) direction
Nz: int = 64
dx: float = 0.1e-3  # grid point spacing in the x direction [m]
dy: float = 0.1e-3  # grid point spacing in the y direction [m]
dz: float = 0.1e-3
kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

# define the properties of the upper layer of the propagation medium
c0: float = 1500.0
sound_speed_compression = c0 * np.ones((Nx, Ny, Nz))  # [m/s]
sound_speed_shear = np.zeros((Nx, Ny, Nz))           # [m/s]
density = 1000.0 * np.ones((Nx, Ny, Nz))             # [kg/m^3]

# define the properties of the lower layer of the propagation medium
sound_speed_compression[Nx // 2:, :, :] = 2000.0  # [m/s]
sound_speed_shear[Nx // 2:, :, :] = 800.0         # [m/s]
density[Nx // 2:, :, :] = 1200.0                  # [kg/m^3]

# define the absorption properties
alpha_coeff_compression = 0.1  # [dB/(MHz^2 cm)]
alpha_coeff_shear = 0.5        # [dB/(MHz^2 cm)]

medium = kWaveMedium(sound_speed_compression,
                     sound_speed_compression=sound_speed_compression,
                     sound_speed_shear=sound_speed_shear,
                     density=density,
                     alpha_coeff_compression=alpha_coeff_compression,
                     alpha_coeff_shear=alpha_coeff_shear)

# create the time array
cfl: float = 0.1     # Courant-Friedrichs-Lewy number
t_end: float = 5e-6  # [s]
kgrid.makeTime(np.max(medium.sound_speed_compression.flatten()), cfl, t_end)

# define source mask to be a square piston
source = kSource()
source_x_pos: int = 10      # [grid points]
source_radius: int = 15     # [grid points]
source.u_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
source.u_mask[source_x_pos,
              Ny // 2 - source_radius:Ny // 2 + source_radius,
              Nz // 2 - source_radius:Nz // 2 + source_radius] = True

# define source to be a velocity source
source_freq = 2e6      # [Hz]
source_cycles = 3
source_mag = 1e-6      # [m/s]
fs = 1.0 / kgrid.dt
source.ux = source_mag * tone_burst(fs, source_freq, source_cycles)

# set source focus
source.ux = focus(kgrid, source.ux, source.u_mask, [0, 0, 0], c0)

# define sensor mask in x-y plane using cuboid corners, where a rectangular
# mask is defined using the xyz coordinates of two opposing corners in the
# form [x1, y1, z1, x2, y2, z2].'
sensor = kSensor()
sensor.mask = [[pml_size, pml_size, Nz // 2,
               Nx - pml_size, Ny - pml_size, Nz // 2]]

# record the maximum pressure in the plane
sensor.record = ['p_max']

# define input arguments
simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                       pml_inside=False)

# run the simulation
sensor_data = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                              source=deepcopy(source),
                              sensor=deepcopy(sensor),
                              medium=deepcopy(medium),
                              simulation_options=deepcopy(simulation_options))



# =========================================================================
# VISUALISATION
# =========================================================================

# # plot the sensor data
# figure;
# imagesc(sensor_data.p_max);
# colormap(getColorMap);
# ylabel('Sensor Position');
# xlabel('Time Step');
# colorbar;



# # plot velocities
# fig2, (ax2a, ax2b) = plt.subplots(nrows=2, ncols=1)
# pcm2a = ax2a.pcolormesh(kgrid.y.T, kgrid.x.T, log_f, shading='gouraud', cmap=plt.colormaps['jet'], clim=(-50.0, 0))
# ax2a.invert_yaxis()
# cb2a = fig2.colorbar(pcm2a, ax=ax2a)
# ax2a.set_xlabel('y [mm]')
# ax2a.set_ylabel('x [mm]')
# cb2a.ax.set_ylabel('[dB]', rotation=90)
# ax2a.set_title('Fluid Model')

# pcm2b = ax2b.pcolormesh(kgrid.y.T, kgrid.x.T, log_e, shading='gouraud', cmap=plt.colormaps['jet'], clim=(-50.0, 0))
# ax2b.invert_yaxis()
# cb2b = fig2.colorbar(pcm2b, ax=ax2b)
# ax2b.set_xlabel('y [mm]')
# ax2b.set_ylabel('x [mm]')
# cb2b.ax.set_ylabel('[dB]', rotation=90)
# ax2b.set_title('Elastic Model')

# fig3, ax3 = plt.subplots(nrows=1, ncols=1)
# pcm3 = ax3.pcolormesh(kgrid.y.T, kgrid.x.T, u_e, shading='gouraud', cmap=plt.colormaps['jet'])
# ax3.invert_yaxis()
# cb3 = fig3.colorbar(pcm3, ax=ax3)
# ax3.set_xlabel('y [mm]')
# ax3.set_ylabel('x [mm]')
# cb3.ax.set_ylabel('[dB]', rotation=90)
# ax3.set_title('Elastic Model')

# plt.show()