""" 
k-wave python example which replicates the fullwave25 simple_plane_wave example, directly 
copying methods (maps-to_coords and gaussian_modulated_sinusoidal_signal) from the package.
"""

from copy import deepcopy

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from tqdm import tqdm

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.utils.colormap import get_color_map
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions



def map_to_coords(
    map_data: NDArray[np.float64 | np.int64 | np.bool],
    *,
    export_as_xyz: bool = False,
) -> NDArray[np.int64]:
    """Map the mask map to coordinates.

    Returns:
        NDArray[np.int64]: An array of coordinates corresponding to non-zero elements in the mask.

    """
    is_3d = map_data.ndim == 3
    # indices = np.where(map_data.T != 0)
    indices = np.where(map_data != 0)
    if is_3d:
        # out = np.array([indices[2], indices[1], indices[0]]).T
        # out = np.array([indices[2], indices[1], indices[0]]).T
        out = np.array([*indices]).T

        if export_as_xyz:
            out = np.stack([out[:, 2], out[:, 1], out[:, 0]], axis=1)
    else:
        out = np.array([*indices]).T
        if export_as_xyz:
            out = np.stack([out[:, 1], out[:, 0]], axis=1)
    return out


def gaussian_modulated_sinusoidal_signal(
    nt: int,
    duration: float,
    ncycles: int,
    drop_off: int,
    f0: float,
    p0: float,
    delay_sec: float = 0.0,
    i_layer: int | None = None,
    dt_for_layer_delay: float | None = None,
    cfl_for_layer_delay: float | None = None,
) -> NDArray[np.float64]:
    """Generate a pulse signal based on input parameters.

    Parameters
    ----------
    nt: int
        Number of time samples of the simulation.
    duration: float
        Total duration of the simulation.
    ncycles: int
        Number of cycles in the pulse.
    drop_off: int
        Controls the pulse decay.
    f0: float
        Frequency of the pulse.
    p0: float
        Amplitude scaling factor.
    delay_sec: float
        Delay in seconds. Default is 0.0.
    i_layer: int
        Index of the layer where the source is located. Default is None.
        This variable is used to shift the pulse signal in time
        so that the signal is emmitted within the transducer layer correctly.
    dt_for_layer_delay: float
        Time step of the simulation. Default is None.
        This variable is used to shift the pulse signal in time
        so that the signal is emmitted within the transducer layer correctly.
    cfl_for_layer_delay: float
        Courant-Friedrichs-Lewy number. Default is None.
        This variable is used to shift the pulse signal in time
        so that the signal is emmitted within the transducer layer correctly.

    Returns
    -------
    NDArray[np.float64]: The generated pulse signal.

    """

    t = (np.arange(0, nt)) / nt * duration - ncycles / f0
    t = t - delay_sec

    if i_layer:
        assert dt_for_layer_delay, "dt must be provided if i_layer is provided"
        assert cfl_for_layer_delay, "cfl must be provided if i_layer is provided"
        t = t - (dt_for_layer_delay / cfl_for_layer_delay) * i_layer

    omega0 = 2 * np.pi * f0
    return (
        np.multiply(
            np.exp(
                -((1.05 * t * omega0 / (ncycles * np.pi)) ** (2 * drop_off)),
            ),
            np.sin(t * omega0),
        )
        * p0
    )



domain_size = (3e-2, 2e-2)  # meters

f0 = 3.0e6
c0 = 1540.0

duration = domain_size[0] / c0 * 2

cfl = 0.2
ppw = 12

wavelength = c0 / f0

dx = wavelength / ppw
dy = dx
dt = cfl * dx / c0  

Nx = int(np.round(domain_size[0] / dx))
Ny = int(np.round(domain_size[1] / dy))

grid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

duration = domain_size[0] / c0 * 2
Nt = int(np.round(duration / dt))
grid.setTime(Nt, dt)

#
# --- define the acoustic medium properties ---
#
# Define the base 2D medium arrays
sound_speed_map = 1540 * np.ones((grid.Nx, grid.Ny))  # m/s
density_map = 1000 * np.ones((grid.Nx, grid.Ny))  # kg/m^3
alpha_coeff_map = 0.5 * np.ones((grid.Nx, grid.Ny))  # dB/(MHz^y cm)

# embed an object with different properties in the center of the medium
obj_x_start = grid.Nx // 3
obj_x_end = 2 * grid.Nx // 3
obj_y_start = grid.Ny // 3
obj_y_end = 2 * grid.Ny // 3

sound_speed_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1600  # m/s
density_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1100  # kg/m^3
alpha_coeff_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 0.75  # dB/(MHz^y cm)

# setup the Medium instance
medium = kWaveMedium(sound_speed=sound_speed_map,
                     density=density_map,
                     alpha_coeff=alpha_coeff_map,
                     alpha_power=1.1)

# initialize the pressure source mask
p_mask = np.zeros((grid.Nx, grid.Ny), dtype=bool)

# set the source location at the top rows of the grid with specified thickness
element_thickness_px = 3
p_mask[0:element_thickness_px, :] = True

# define the pressure source [n_sources, nt]
p0 = np.zeros((p_mask.sum(), grid.Nt)) 

p_coordinates = map_to_coords(p_mask)

for i_thickness in range(element_thickness_px):
    # create a gaussian-modulated sinusoidal pulse as the source signal with layer delay
    p0_vec = gaussian_modulated_sinusoidal_signal(nt=grid.Nt,  # number of time steps
                                                  f0=f0,  # center frequency [Hz]
                                                  duration=duration,  # duration [s]
                                                  ncycles=2,  # number of cycles
                                                  drop_off=2,  # drop off factor
                                                  p0=1e5,  # maximum amplitude [Pa]
                                                  i_layer=i_thickness,
                                                  dt_for_layer_delay=grid.dt,
                                                  cfl_for_layer_delay=cfl)

    # assign the source signal to the corresponding layer
    n_y = p_coordinates.shape[0] // element_thickness_px
    p0[n_y * i_thickness : n_y * (i_thickness + 1), :] = p0_vec.copy()

# create the kSource instance
source = kSource()
source.p_mask = p_mask
source.p = p0

# setup the Sensor instance
sensor_mask = np.ones((grid.Nx, grid.Ny), dtype=bool)
sensor = kSensor()
sensor.mask = sensor_mask
sensor.record = ["p"]


# --------------------
# SIMULATION
# --------------------

simulation_options = SimulationOptions(pml_auto=True, data_recast=True, save_to_disk=True, save_to_disk_exit=False, pml_inside=False)

execution_options = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False, verbose_level=2)

sensor_data = kspaceFirstOrder2D(kgrid=deepcopy(grid),
                                 source=deepcopy(source),
                                 sensor=deepcopy(sensor),
                                 medium=deepcopy(medium),
                                 simulation_options=simulation_options,
                                 execution_options=execution_options)

# --------------------
# VISUALIZATION
# --------------------

propagation_map = np.reshape(sensor_data["p"], (grid.Nt, grid.Nx, grid.Ny), order='F')  # Store the recorded pressure data

p_max_plot = np.abs(propagation_map).max().item() / 4

time_step = propagation_map.shape[0] // 3



cmap = get_color_map()

fig, ax = plt.subplots(1, 1)
im = ax.imshow(propagation_map[time_step, :, :],
               extent=[grid.y_vec.min() * 1e3, grid.y_vec.max() * 1e3, grid.x_vec.min() * 1e3, grid.x_vec.max() * 1e3],
               vmin=-p_max_plot,
               vmax=p_max_plot,
               cmap=cmap)
title = "Snapshot"
ax.set_title(title)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad="2%")
ax.set_ylabel("x-position [mm]")
ax.set_xlabel("y-position [mm]")
fig.colorbar(im, cax=cax)
plt.show()
    

fig, ax = plt.subplots(1, 1, figsize=(4,6))

num_plot_image: int = 50
skip_every_n_frame =int(grid.Nt / num_plot_image)

c_map = medium.sound_speed
rho_map = medium.density    

start = 0
end = None
z_map = c_map * rho_map
z_map = (z_map - np.min(z_map)) / (np.max(z_map) - np.min(z_map) + 1e-9)

z_map_offset = p_max_plot * 0.8

animation_list = []

# propagation_map = propagation_map.transpose(2, 0, 1)
for i, p_map_i in tqdm(
    enumerate(propagation_map[::skip_every_n_frame, start:end, start:end]),
    total=len(propagation_map[::skip_every_n_frame, start:end, start:end]),
    desc="plotting animation"):

    processed_p_map = p_map_i + z_map_offset * (z_map)

    image2 = ax.imshow(
        processed_p_map,
        vmin=-p_max_plot,
        vmax=p_max_plot,
        interpolation="nearest"
    )
    # set text to show the current time step
    text = ax.text(0.5, 1.05,  f"t = {i * skip_every_n_frame} / {propagation_map.shape[0]}",  fontsize=4,
                   ha="center",
                   animated=True,
                   transform=ax.transAxes)
    animation_list.append([image2, text])

animation_data = animation.ArtistAnimation(
    fig,
    animation_list,
    interval=150,
    blit=True,
    repeat_delay=500,
)
animation_data.save("fullwave_plane_wave.mp4", writer="ffmpeg", dpi=300)
plt.close("all")
