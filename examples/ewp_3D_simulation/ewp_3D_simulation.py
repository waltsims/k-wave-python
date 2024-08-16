
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.pstdElastic3D import pstd_elastic_3d

from kwave.utils.signals import tone_burst
from kwave.utils.colormap import get_color_map

from kwave.options.simulation_options import SimulationOptions, SimulationType

# from io import BytesIO
import pyvista as pv
# import meshio
from skimage import measure

def focus(kgrid, input_signal, source_mask, focus_position, sound_speed):
    """
    focus Create input signal based on source mask and focus position.

    focus takes a single input signal and a source mask and creates an
    input signal matrix (with one input signal for each source point).
    The appropriate time delays required to focus the signals at a given
    position in Cartesian space are automatically added based on the user
    inputs for focus_position and sound_speed.

    Args:
         kgrid:             k-Wave grid object returned by kWaveGrid
         input_signal:      single time series input
         source_mask:       matrix specifying the positions of the time
                            varying source distribution (i.e., source.p_mask
                            or source.u_mask)
         focus_position:    position of the focus in Cartesian coordinates
         sound_speed:       scalar sound speed

    Returns:
         input_signal_mat:  matrix of time series following the source points
    """

    assert not isinstance(kgrid.t_array, str), "kgrid.t_array must be a numeric array."

    if isinstance(sound_speed, int):
        sound_speed = float(sound_speed)

    assert isinstance(sound_speed, float), "sound_speed must be a scalar."

    # calculate the distance from every point in the source mask to the focus position
    if kgrid.dim == 1:
        dist = np.abs(kgrid.x[source_mask == 1] - focus_position[0])
    elif kgrid.dim == 2:
        dist = np.sqrt((kgrid.x[source_mask == 1] - focus_position[0])**2 +
                       (kgrid.y[source_mask == 1] - focus_position[1])**2 )
    elif kgrid.dim == 3:
        dist = np.sqrt((kgrid.x[source_mask == 1] - focus_position[0])**2 +
                       (kgrid.y[source_mask == 1] - focus_position[1])**2 +
                       (kgrid.z[source_mask == 1] - focus_position[2])**2 )

    # convert distances to time delays
    delays = np.round(dist / (kgrid.dt * sound_speed)).astype(int)

    # convert time points to delays relative to the maximum delays
    relative_delays = delays.max() - delays

    # largest time delay
    max_delay = np.max(relative_delays)

    signal_mat = np.zeros((relative_delays.size, input_signal.size + max_delay), order='F')

    # assign the input signal
    for source_index, delay in enumerate(relative_delays):
        signal_mat[source_index, :] = np.hstack([np.zeros((delay,)),
                                                 np.squeeze(input_signal),
                                                 np.zeros((max_delay - delay,))])

    return signal_mat


def get_focus(p):
    """
    Gets value of maximum pressure and the indices of the location
    """
    max_pressure = np.max(p)
    mx, my, mz = np.unravel_index(np.argmax(p, axis=None), p.shape)
    return max_pressure, [mx, my, mz]


def getPVImageData(kgrid, p, order='F'):
    """Create the pyvista image data container with data label hardwired"""
    pv_grid = pv.ImageData()
    pv_grid.dimensions = (kgrid.Nx, kgrid.Ny, kgrid.Nz)
    pv_grid.origin = (0, 0, 0)
    pv_grid.spacing = (kgrid.dx, kgrid.dy, kgrid.dz)
    pv_grid.point_data["p_max"] = p.flatten(order=order)
    pv_grid.deep_copy = False
    return pv_grid


def getIsoVolume(kgrid, p, dB=-6):
    """"Returns a triangulation of a volume, warning: may not be connected or closed"""

    max_pressure, _ = get_focus(p)
    ratio = 10**(dB / 20.0) * max_pressure
    # don't need normals or values
    verts, faces, _, _ = measure.marching_cubes(p, level=ratio, spacing=[kgrid.dx, kgrid.dy, kgrid.dz])
    return verts, faces


def getFWHM(kgrid, p):
    """"Gets volume of -6dB field"""
    verts, faces = getIsoVolume(kgrid, p)

    totalArea: float = 0.0

    m: int = np.max(np.shape(faces)) - 1
    for i in np.arange(0, m, dtype=int):
        p0 = verts[faces[m, 0]]
        p1 = verts[faces[m, 1]]
        p2 = verts[faces[m, 2]]

        a = np.asarray(p1 - p0)
        b = np.asarray(p2 - p0)

        n = np.cross(a, b)
        nn = np.abs(n)

        area = nn / 2.0
        normal = n / nn
        centre = (p0 + p1 + p2) / 3.0

        totalArea += area * (centre[0] * normal[0] + centre[1] * normal[1] + centre[2] * normal[2])

    d13 = [[verts[faces[:, 1], 0] - verts[faces[:, 2], 0]],
           [verts[faces[:, 1], 1] - verts[faces[:, 2], 1]],
           [verts[faces[:, 1], 2] - verts[faces[:, 2], 2]] ]

    d12 = [[verts[faces[:, 0], 0] - verts[faces[:, 1], 0]],
           [verts[faces[:, 0], 1] - verts[faces[:, 1], 1]],
           [verts[faces[:, 0], 2] - verts[faces[:, 1], 2]] ]

    # cross-product vectorized
    cr = np.cross(np.squeeze(np.transpose(d13)), np.squeeze(np.transpose(d12)))
    cr = np.transpose(cr)

    # Area of each triangle
    area = 0.5 * np.sqrt(cr[0, :]**2 + cr[1, :]**2 + cr[2, :]**2)

    # Total area
    totalArea = np.sum(area)

    # norm of cross product
    crNorm = np.sqrt(cr[0, :]**2 + cr[1, :]**2 + cr[2, :]**2)

    # centroid
    zMean = (verts[faces[:, 0], 2] + verts[faces[:, 1], 2] + verts[faces[:, 2], 2]) / 3.0

    # z component of normal for each triangle
    nz = -cr[2, :] / crNorm

    # contribution of each triangle
    volume = np.abs(np.multiply(np.multiply(area, zMean), nz))

    # divergence theorem
    totalVolume = np.sum(volume)

    # display volume to screen
    print('\n\tTotal volume of FWHM {vol:8.5e} [m^3]'.format(vol=totalVolume))

    return verts, faces


def plot3D(kgrid, p, tx_plane_coords, verbose=False):
    """Plots using pyvista"""

    max_pressure, max_loc = get_focus(p)
    if verbose:
        print(max_pressure, max_loc)

    min_pressure = np.min(p)
    if verbose:
        print(min_pressure)

    pv_grid = getPVImageData(kgrid, p)
    if verbose:
        print(pv_grid)

    verts, faces = getFWHM(kgrid, p)

    # cells = [("triangle", faces)]
    # mesh = meshio.Mesh(verts, cells)

    # buffer = BytesIO()

    # mesh.write(buffer, file_format="ply")

    # buffer.seek(0)
    # # Read the buffer with PyVista
    # dataset = pv.read(buffer.seek(0), file_format='ply')
    # mesh.write("foo2.vtk")
    # dataset = pv.read('foo2.vtk')

    num_faces = faces.shape[0]
    faces_pv = np.hstack([np.full((num_faces, 1), 3), faces])

    dataset = pv.PolyData(verts, faces_pv)

    pv_x = np.linspace(0, (kgrid.Nx - 1.0) * kgrid.dx, kgrid.Nx)
    pv_y = np.linspace(0, (kgrid.Ny - 1.0) * kgrid.dy, kgrid.Ny)
    pv_z = np.linspace(0, (kgrid.Nz - 1.0) * kgrid.dz, kgrid.Nz)

    islands = dataset.connectivity(largest=False)
    split_islands = islands.split_bodies(label=True)
    region = []
    xx = []
    for i, body in enumerate(split_islands):
        region.append(body)
        pntdata = body.GetPoints()
        xx.append(np.zeros((pntdata.GetNumberOfPoints(), 3)))
        for j in range(pntdata.GetNumberOfPoints()):
            xx[i][j, 0] = pntdata.GetPoint(j)[0]
            xx[i][j, 1] = pntdata.GetPoint(j)[1]
            xx[i][j, 2] = pntdata.GetPoint(j)[2]

    tx_plane = [pv_x[tx_plane_coords[0]],
                pv_y[tx_plane_coords[1]],
                pv_z[tx_plane_coords[2]]]

    mx, my, mz = max_loc
    max_loc = [pv_x[mx], pv_y[my], pv_z[mz]]

    single_slice_x = pv_grid.slice(origin=max_loc, normal=[1, 0, 0])
    single_slice_y = pv_grid.slice(origin=max_loc, normal=[0, 1, 0])
    single_slice_z = pv_grid.slice(origin=max_loc, normal=[0, 0, 1])

    single_slice_tx = pv_grid.slice(origin=tx_plane, normal=[1, 0, 0])

    # formatting of colorbar
    sargs = dict(interactive=True,
                 title='Pressure [Pa]',
                 height=0.90,
                 vertical=True,
                 position_x=0.90,
                 position_y=0.05,
                 title_font_size=20,
                 label_font_size=16,
                 shadow=False,
                 n_labels=6,
                 italic=False,
                 fmt="%.5e",
                 font_family="arial")

    # dictionary for annotations of colorbar
    ratio = 10**(-6 / 20.0) * max_pressure

    annotations = dict([(float(ratio), "-6dB")])

    # plotter object
    plotter = pv.Plotter()

    # slice data
    _ = plotter.add_mesh(single_slice_x,
                         cmap='turbo',
                         clim=[min_pressure, max_pressure],
                         opacity=0.5,
                         scalar_bar_args=sargs,
                         annotations=annotations)
    _ = plotter.add_mesh(single_slice_y, cmap='turbo', clim=[min_pressure, max_pressure], opacity=0.5, show_scalar_bar=False)
    _ = plotter.add_mesh(single_slice_z, cmap='turbo', clim=[min_pressure, max_pressure], opacity=0.5, show_scalar_bar=False)

    # transducer plane
    _ = plotter.add_mesh(single_slice_tx, cmap='spring', clim=[min_pressure, max_pressure], opacity=1, show_scalar_bar=False)

    # full width half maximum
    _ = plotter.add_mesh(region[0], color='red', opacity=0.75, label='-6 dB')

    # add the frame around the image
    _ = plotter.show_bounds(grid='front',
                            location='outer',
                            ticks='outside',
                            color='black',
                            minor_ticks=False,
                            padding=0.0,
                            show_xaxis=True,
                            show_xlabels=True,
                            xtitle='',
                            n_xlabels=5,
                            ytitle="",
                            ztitle="")

    _ = plotter.add_axes(color='pink', labels_off=False)
    # plotter.camera_position = 'yz'

    # # plotter.camera.elevation = 45
    # plotter.camera.roll = 0
    # plotter.camera.azimuth = 125
    # plotter.camera.elevation = 5

    # # extensions = ("svg", "eps", "ps", "pdf", "tex")
    # fname = "fwhm" + "." + "svg"
    # plotter.save_graphic(fname, title="PyVista Export", raster=True, painter=True)

    plotter.show()


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

myOrder = 'F'

# create the computational grid
pml_size: int = 10

Nx: int = 64
Ny: int = 64
Nz: int = 64
dx: float = 0.1e-3
dy: float = 0.1e-3
dz: float = 0.1e-3
kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

# define the properties of the upper layer of the propagation medium
c0: float = 1500.0
rho0: float = 1000.0
sound_speed_compression = c0 * np.ones((Nx, Ny, Nz), order=myOrder)  # [m/s]
sound_speed_shear = np.zeros((Nx, Ny, Nz), order=myOrder)            # [m/s]
density = rho0 * np.ones((Nx, Ny, Nz), order=myOrder)                # [kg/m^3]

# define the properties of the lower layer of the propagation medium
sound_speed_compression[Nx // 2 - 1:, :, :] = 2000.0  # [m/s]
sound_speed_shear[Nx // 2 - 1:, :, :] = 800.0         # [m/s]
density[Nx // 2 - 1:, :, :] = 1200.0                  # [kg/m^3]

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
source.u_mask = np.zeros((Nx, Ny, Nz), dtype=int, order=myOrder)
source.u_mask[source_x_pos,
              Ny // 2 - source_radius:Ny // 2 + source_radius,
              Nz // 2 - source_radius:Nz // 2 + source_radius] = 1

# define source to be a velocity source
source_freq = 2e6      # [Hz]
source_cycles = 3      # []
source_mag = 1e-6      # [m/s]
fs = 1.0 / kgrid.dt    # [Hz]
ux = source_mag * tone_burst(fs, source_freq, source_cycles)

# set source focus
source.ux = focus(kgrid, ux, source.u_mask, [0, 0, 0], c0)

# define sensor mask in x-y plane using cuboid corners, where a rectangular
# mask is defined using the xyz coordinates of two opposing corners in the
# form [x1, y1, z1, x2, y2, z2].'
# In this case the sensor mask in the slice through the xy-plane at z = Nz // 2 - 1
# cropping the pml
sensor = kSensor()
sensor.mask = np.array([[pml_size, pml_size, Nz // 2 - 1, Nx - pml_size, Ny - pml_size, Nz // 2]]).T

# sensor.mask = np.ones((Nx, Ny, Nz), order=myOrder)

# record the maximum pressure in the sensor.mask plane
sensor.record = ['p_max']

# define input arguments
simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                       kelvin_voigt_model=True,
                                       use_sensor=True,
                                       nonuniform_grid=False,
                                       blank_sensor=False)

# run the simulation
sensor_data = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                              source=deepcopy(source),
                              sensor=deepcopy(sensor),
                              medium=deepcopy(medium),
                              simulation_options=deepcopy(simulation_options))


# =========================================================================
# VISUALISATION
# =========================================================================

# define axes
x_vec = np.squeeze(kgrid.x_vec) * 1000.0
y_vec = np.squeeze(kgrid.y_vec) * 1000.0
x_vec = x_vec[pml_size:Nx - pml_size]
y_vec = y_vec[pml_size:Ny - pml_size]

# p_max_f = np.reshape(sensor_data[0].p_max, (x_vec.size, y_vec.size), order='F')
# p_max_c = np.reshape(sensor_data[0].p_max, (x_vec.size, y_vec.size), order='C')

# sensor_data.p_max = np.reshape(sensor_data.p_max, (Nx, Ny, Nz), order='F')

# p_max = np.reshape(sensor_data.p_max[pml_size:Nx - pml_size, pml_size:Ny - pml_size, Nz // 2 - 1], (x_vec.size, y_vec.size), order='F')

p_max = np.reshape(sensor_data.p_max, (x_vec.size, y_vec.size), order='F')

# plot
fig1, ax1 = plt.subplots(nrows=1, ncols=1)
pcm1 = ax1.pcolormesh(x_vec, y_vec, p_max,
                      cmap = get_color_map(), shading='gouraud', alpha=1.0, vmin=0, vmax=6)
cb1 = fig1.colorbar(pcm1, ax=ax1)
ax1.set_ylabel('$x$ [mm]')
ax1.set_xlabel('$y$ [mm]')

plt.show()

# # indices of transducer location
# coordinates = np.argwhere(source.u_mask == 1)
# coordinates = np.reshape(coordinates, (-1, 3))

# # 3D plotting
# plot3D(kgrid, sensor_data.p_max, coordinates[0])