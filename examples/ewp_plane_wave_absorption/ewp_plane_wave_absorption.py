
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.pstdElastic2D import pstd_elastic_2d

from kwave.utils.filters import smooth, spect
from kwave.utils.math import find_closest

from kwave.options.simulation_options import SimulationOptions, SimulationType

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
pos1: int = 44  # [grid points]
pos2: int = 64  # [grid points]
sensor.mask[pos1, Ny // 2 - 1] = True
sensor.mask[pos2, Ny // 2 - 1] = True

# set sensor to record to particle velocity
sensor.record = ["u"]

# calculate the distance between the sensor positions
d_cm: float = (pos2 - pos1) * dx * 100.0  # [cm]

# define source mask
source_mask = np.ones((Nx, Ny))
source_pos: int = 34  # [grid points]

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
source.ux = np.zeros((Nx, Ny))
source.ux[source_pos, :] = 1.0
source.ux = smooth(source.ux, restore_max=True)
# consistent shape: the source is of shape ((Nx*Ny, 1))
source.ux = 1e-6 * np.reshape(source.ux, (-1, 1), order='F')

# set end time
t_end = 3.5e-6

# create a time array
c_max = np.max([medium.sound_speed_compression, medium.sound_speed_shear])
kgrid.makeTime(c_max, cfl, t_end)

simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                       pml_inside=True,
                                       pml_size=pml_size,
                                       pml_alpha=pml_alpha,
                                       kelvin_voigt_model=True,
                                       binary_sensor_mask=True,
                                       use_sensor=True,
                                       nonuniform_grid=False,
                                       blank_sensor=False)

# run the simulation
sensor_data_comp = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                   source=deepcopy(source),
                                   sensor=deepcopy(sensor),
                                   medium=deepcopy(medium),
                                   simulation_options=deepcopy(simulation_options))

# calculate the amplitude spectrum at the two sensor positions as as1 and as2
fs = 1.0 / kgrid.dt
_, as1, _ = spect(np.expand_dims(sensor_data_comp.ux[0, :], axis=0), fs)
f_comp, as2, _ = spect(np.expand_dims(sensor_data_comp.ux[1, :], axis=0), fs)

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

# redefine source
del source

source = kSource()
source.u_mask = source_mask
source.uy = np.zeros((Nx, Ny))
source.uy[source_pos, :] = 1.0
source.uy = smooth(source.uy, restore_max=True)
# consistent shape: the source is of shape ((Nx*Ny, 1))
source.uy = 1e-6 * np.reshape(source.uy, (-1, 1), order='F')

# set end time
t_end: float = 4e-6

# create a time array
c_max = np.max([medium.sound_speed_compression, medium.sound_speed_shear])
kgrid.makeTime(c_max, cfl, t_end)

# run the simulation
sensor_data_shear = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                    source=deepcopy(source),
                                    sensor=deepcopy(sensor),
                                    medium=deepcopy(medium),
                                    simulation_options=deepcopy(simulation_options))

# calculate the amplitude at the two sensor positions
fs = 1.0 / kgrid.dt
_, as1, _ = spect(np.expand_dims(sensor_data_shear.uy[0, :], axis=0), fs)
f_shear, as2, _ = spect(np.expand_dims(sensor_data_shear.uy[1, :], axis=0), fs)

# calculate the attenuation from the amplitude spectrums
attenuation_shear = -20.0 * np.log10(as2 / as1) / d_cm

# calculate the corresponding theoretical attenuation in dB/cm
attenuation_th_shear = medium.alpha_coeff_shear * (f_shear * 1e-6)**2

# calculate the maximum supported frequency
f_max_shear = medium.sound_speed_shear / (2.0 * dx)

# find the maximum frequency in the frequency vector
_, f_max_shear_index = find_closest(f_shear, f_max_shear)

print(np.max(sensor_data_comp.ux[0, :]))
print(np.max(sensor_data_comp.ux[1, :]))
print(np.max(sensor_data_shear.uy[0, :]))
print(np.max(sensor_data_shear.uy[1, :]))

# =========================================================================
# VISUALISATION
# =========================================================================

# plot layout of simulation
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

# plot compressional wave traces
t_axis_comp = np.arange(np.shape(sensor_data_comp.ux)[1]) * kgrid.dt * 1e6
ax1.plot(t_axis_comp, sensor_data_comp.ux[1, :], 'r-')
ax1.plot(t_axis_comp, sensor_data_comp.ux[0, :], 'k--')
ax1.set_xlim(0, t_axis_comp[-1])
ax1.set_ylim(0, np.max(sensor_data_comp.ux[1, :]))
ax1.set_xlabel(r'Time [$\mu$s]')
ax1.set_ylabel('Particle Velocity')
ax1.set_title('Compressional Wave', fontweight='bold')

# plot compressional wave absorption
ax2.plot(f_comp * 1e-6, np.squeeze(attenuation_th_comp), 'k-', )
ax2.plot(f_comp * 1e-6, np.squeeze(attenuation_comp), 'o',
         markeredgecolor='k', markerfacecolor='None')
ax2.set_xlim(0, f_max_comp * 1e-6)
ax2.set_ylim(0, attenuation_th_comp[f_max_comp_index] * 1.1)
ax2.set_xlabel('Frequency [MHz]')
ax2.set_ylabel(r'$\alpha$ [dB/cm]')

# plot shear wave traces
t_axis_shear = np.arange(np.shape(sensor_data_shear.uy)[1]) * kgrid.dt * 1e6
ax3.plot(t_axis_shear, sensor_data_shear.uy[1, :], 'r-')
ax3.plot(t_axis_shear, sensor_data_shear.uy[0, :], 'k--')
ax3.set_xlim(0, t_axis_shear[-1])
ax3.set_ylim(0, np.max(sensor_data_shear.uy[1, :]))
ax3.set_xlabel(r'Time [$\mu$s]')
ax3.set_ylabel('Particle Velocity')
ax3.set_title('Shear Wave', fontweight='bold')

# plot shear wave absorption
ax4.plot(f_shear * 1e-6, np.squeeze(attenuation_th_shear), 'k-')
ax4.plot(f_shear * 1e-6,  np.squeeze(attenuation_shear), 'o',
         markeredgecolor='k', markerfacecolor='None')
ax4.set_xlim(0, f_max_shear * 1e-6)
ax4.set_ylim(0, attenuation_th_shear[f_max_shear_index] * 1.1)
ax4.set_xlabel('Frequency [MHz]')
ax4.set_ylabel(r'$\alpha$ [dB/cm]')

plt.show()