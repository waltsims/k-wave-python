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

from kwave.utils.filters import smooth
from kwave.utils.math import find_closest
from kwave.utils.signals import reorder_sensor_data

from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.options.simulation_execution_options import SimulationExecutionOptions

"""
Plane Wave Absorption Example
#
# This example illustrates the characteristics of the Kelvin-Voigt
# absorption model used in the k-Wave simulation functions pstdElastic2D,
# pstdElastic3D. It builds on the Explosive Source In A Layered Medium
# Example.
#
# author: Bradley Treeby
# date: 17th January 2014
# last update: 25th July 2019
#
# This function is part of the k-Wave Toolbox (http://www.k-wave.org)
# Copyright (C) 2014-2019 Bradley Treeby

# This file is part of k-Wave. k-Wave is free software: you can
# redistribute it and/or modify it under the terms of the GNU Lesser
# General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
"""


# =========================================================================
# SET GRID PARAMETERS
# =========================================================================

# create the computational grid
Nx: int = 128       # number of grid points in the x (row) direction
Ny: int = 32        # number of grid points in the y (column) direction
dx: float = 0.1e-3  # grid point spacing in the x direction [m]
dy: float = 0.1e-3  # grid point spacing in the y direction [m]
kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

# define the properties of the propagation medium
sound_speed_compression = 1800.0  # [m/s]
sound_speed_shear = 1200.0        # [m/s]
density = 1000.0                  # [kg/m^3]

# set the absorption properties
alpha_coeff_compression = 1.0  # [dB/(MHz^2 cm)]
alpha_coeff_shear = 1.0        # [dB/(MHz^2 cm)]

medium = kWaveMedium(sound_speed=sound_speed_compression,
                     sound_speed_compression=sound_speed_compression,
                     sound_speed_shear=sound_speed_shear,
                     density=density,
                     alpha_coeff_compression=alpha_coeff_compression,
                     alpha_coeff_shear=alpha_coeff_shear)

# define binary sensor mask with two sensor positions
sensor = kSensor()
sensor.mask = np.zeros((Nx, Ny), dtype=bool)
pos1: int = 45  # [grid points]
pos2: int = 65  # [grid points]
sensor.mask[pos1, Ny // 2] = True
sensor.mask[pos2, Ny // 2] = True

# set sensor to record to particle velocity
sensor.record = ["u"]

# calculate the distance between the sensor positions
d_cm: float = (pos2 - pos1) * dx * 100.0  # [cm]

# define source mask
source_mask = np.ones((Nx, Ny))
source_pos: int = 35  # [grid points]

# set the CFL
cfl: float = 0.05

# define the properties of the PML to allow plane wave propagation
pml_alpha: float = 0.0
pml_size = [int(30), int(2)]

# =========================================================================
# COMPRESSIONAL PLANE WAVE SIMULATION
# =========================================================================

# define source
source = kSource()
source.u_mask = source_mask
ux = np.zeros((Nx, Ny))
ux[source_pos, :] = 1.0
ux = smooth(ux, restore_max=True)
source.ux = 1e-6 * np.reshape(ux, (-1, 1))

# set end time
t_end = 3.5e-6

# create the time array
c_max = np.max([medium.sound_speed_compression, medium.sound_speed_shear])
kgrid.makeTime(c_max, cfl, t_end)

simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                       pml_inside=False,
                                       pml_size=pml_size,
                                       pml_alpha=pml_alpha)

# run the simulation
sensor_data_comp = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                   source=deepcopy(source),
                                   sensor=deepcopy(sensor),
                                   medium=deepcopy(medium),
                                   simulation_options=deepcopy(simulation_options))


# calculate the amplitude spectrum at the two sensor positions
fs = 1.0 / kgrid.dt
_, as1 = spect(sensor_data_comp.ux[0, :], fs)
f_comp, as2 = spect(sensor_data_comp.ux[1, :], fs)

# calculate the attenuation from the amplitude spectrums
attenuation_comp = -20.0 * np.log10(as2 / as1) / d_cm

# calculate the corresponding theoretical attenuation in dB/cm
attenuation_th_comp = medium.alpha_coeff_compression * (f_comp * 1e-6)**2

# calculate the maximum supported frequency
f_max_comp = medium.sound_speed_compression / (2.0 * dx)

# find the maximum frequency in the frequency vector
_, f_max_comp_index = find_closest(f_comp, f_max_comp)

# =========================================================================
# SHEAR PLANE WAVE SIMULATION
# =========================================================================

# define source
del source

source = kSource()
source.u_mask = source_mask
uy = np.zeros((Nx, Ny))
uy[source_pos, :] = 1.0
uy = smooth(uy, restore_max=True)
uy = 1e-6 * reshape(uy, [], 1)
source.uy = np.reshape(uy, (-1, 1))

# set end time
t_end: float = 4e-6

# create the time array
c_max = np.max([medium.sound_speed_compression.max(), medium.sound_speed_shear.max()])
kgrid.makeTime(c_max, cfl, t_end)

# run the simulation
sensor_data_shear = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                    source=deepcopy(source),
                                    sensor=deepcopy(sensor),
                                    medium=deepcopy(medium),
                                    simulation_options=deepcopy(simulation_options))

# calculate the amplitude at the two sensor positions
fs = 1.0 / kgrid.dt
_, as1 = spect(sensor_data_shear.uy[0, :], fs)
f_shear, as2 = spect(sensor_data_shear.uy[1, :], fs)

# calculate the attenuation from the amplitude spectrums
attenuation_shear = -20.0 * np.log10(as2 / as1) / d_cm

# calculate the corresponding theoretical attenuation in dB/cm
attenuation_th_shear = medium.alpha_coeff_shear * (f_shear * 1e-6)**2

# calculate the maximum supported frequency
f_max_shear = medium.sound_speed_shear / (2.0 * dx)

# find the maximum frequency in the frequency vector
_, f_max_shear_index = find_closest(f_shear, f_max_shear)

# =========================================================================
# VISUALISATION
# =========================================================================

# plot layout of simulation
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

# plot compressional wave traces
t_axis = np.arange(len(sensor_data_comp.ux) - 1) * kgrid.dt * 1e6
ax1.plt(t_axis, sensor_data_comp.ux, 'k-')
# axis tight;
ax1.set_xlabel('Time [$\mus$]')
ax1.set_ylabel('Particle Velocity')
ax1.set_title('Compressional Wave')

# plot compressional wave absorption

ax2.plt(f_comp * 1e-6, attenuation_comp, 'ko',
        f_comp * 1e-6, attenuation_th_comp, 'k-')
ax2.set_xlim(0, f_max_comp * 1e-6)
ax2.set_ylim(0, attenuation_th_comp[f_max_comp_index] * 1.1)
# box on;
ax2.set_xlabel('Frequency [MHz]')
ax2.set_ylabel('$\alpha$ [dB/cm]')

# plot shear wave traces

t_axis = np.arange(len(sensor_data_comp.ux) - 1) * kgrid.dt * 1e6

ax3.plt(t_axis, sensor_data_shear.uy, 'k-')
# axis tight;
ax3.xlabel('Time [$\mu$s]')
ax3.ylabel('Particle Velocity')
ax3.title('Shear Wave')

# plot shear wave absorption
ax4.plot(f_shear * 1e-6, attenuation_shear, 'ko',
         f_shear * 1e-6, attenuation_th_shear, 'k-')
ax4.set_xlim(0, f_max_shear * 1e-6)
ax4.set_ylim(0, attenuation_th_shear[f_max_shear_index] * 1.1)
# box on;
ax4.set_xlabel('Frequency [MHz]')
ax4.set_ylabel('$\alpha$ [dB/cm]')