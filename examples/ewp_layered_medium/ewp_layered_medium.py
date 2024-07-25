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
from kwave.pstdElastic2D import pstd_elastic_2d

from kwave.utils.dotdictionary import dotdict
from kwave.utils.mapgen import make_disc, make_circle
from kwave.utils.signals import reorder_sensor_data
from kwave.utils.matlab import matlab_mask

from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.options.simulation_execution_options import SimulationExecutionOptions

"""
Explosive Source In A Layered Medium Example

This example provides a simple demonstration of using k-Wave for the
simulation and detection of compressional and shear waves in elastic and
viscoelastic media within a two-dimensional heterogeneous medium. It
builds on the Homogenous Propagation Medium and Heterogeneous Propagation
Medium examples.

author: Bradley Treeby
date: 11th February 2014
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
Nx: int = 128       # number of grid points in the x (row) direction
Ny: int = 128       # number of grid points in the y (column) direction
dx: float = 0.1e-3  # grid point spacing in the x direction [m]
dy: float = 0.1e-3  # grid point spacing in the y direction [m]
kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

# define the properties of the upper layer of the propagation medium
sound_speed_compression = 1500.0 * np.ones((Nx, Ny))  # [m/s]
sound_speed_shear = np.zeros((Nx, Ny))                # [m/s]
density = 1000.0 * np.ones((Nx, Ny))                  # [kg/m^3]

# define the properties of the lower layer of the propagation medium
sound_speed_compression[Nx // 2: -1, :] = 2000.0  # [m/s]
sound_speed_shear[Nx // 2: -1, :] = 800.0         # [m/s]
density[Nx // 2: -1, :] = 1200.0                  # [kg/m^3]

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
t_end: float = 8e-6  # [s]
kgrid.makeTime(np.max(medium.sound_speed_compression.flatten()), cfl, t_end)

# create initial pressure distribution using make_disc
disc_magnitude: float = 5.0  # [Pa]
disc_x_pos: int = 30         # [grid points]
disc_y_pos: int = 64         # [grid points]
disc_radius: int = 5         # [grid points]
source = kSource()
source.p0 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([disc_x_pos, disc_y_pos]), disc_radius)

# define a circular sensor or radius 20 grid points, centred at origin
sensor = kSensor()
sensor.mask = make_circle(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 20)

# prehaps this helps
sensor.record = ['p']

# define a custom display mask showing the position of the interface from
# the fluid side
display_mask = np.zeros((Nx, Ny), dtype=bool)
display_mask[Nx//2 - 1, :] = True

# run the simulation
simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                       pml_inside=False)

sensor_data = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                              source=deepcopy(source),
                              sensor=deepcopy(sensor),
                              medium=deepcopy(medium),
                              simulation_options=deepcopy(simulation_options))

# reorder the simulation data
sensor_data_reordered = dotdict()
sensor_data_reordered.p = reorder_sensor_data(kgrid, sensor, sensor_data.p)

# =========================================================================
# VISUALISATION
# =========================================================================

# plot layout of simulation
fig1, ax1 = plt.subplots(nrows=1, ncols=1)
_ = ax1.pcolormesh(kgrid.y.T, kgrid.x.T,
                   np.logical_or(np.logical_or(source.p0, sensor.mask), display_mask).T,
                   cmap='gray_r', shading='gouraud', alpha=1)
ax1.invert_yaxis()
ax1.set_xlabel('y [mm]')
ax1.set_ylabel('x [mm]')

# plot velocities
fig2, ax2 = plt.subplots(nrows=1, ncols=1)
pcm2 = ax2.pcolormesh(sensor_data.p, shading='gouraud', cmap=plt.colormaps['jet'])
cb2 = fig2.colorbar(pcm2, ax=ax2)
ax2.set_xlabel('Sensor Position')
ax2.set_ylabel('Time Step')

fig3, ax3 = plt.subplots(nrows=1, ncols=1)
pcm3 = ax3.imshow(sensor_data.p)
cb3 = fig3.colorbar(pcm3, ax=ax3)
ax3.set_xlabel('Sensor Position')
ax3.set_ylabel('Time Step')

fig3, ax3 = plt.subplots(nrows=1, ncols=1)
pcm3 = ax3.imshow(sensor_data_reordered.p)
cb3 = fig3.colorbar(pcm3, ax=ax3)
ax3.set_xlabel('Sensor Position')
ax3.set_ylabel('Time Step')


plt.show()