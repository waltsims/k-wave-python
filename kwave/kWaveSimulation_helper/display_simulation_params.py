import logging
import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.data import scale_SI


def display_simulation_params(kgrid: kWaveGrid, medium: kWaveMedium, elastic_code: bool):
    dt = kgrid.dt
    t_array_end = kgrid.t_array[0][-1]
    Nt = int(kgrid.Nt)
    k_size = kgrid.size

    # display time step information
    logging.log(logging.INFO, f"  dt: {scale_SI(dt)[0]} s, t_end: {scale_SI(t_array_end)[0]}s, time steps: {Nt}")

    c_min, c_min_comp, c_min_shear = get_min_sound_speed(medium, elastic_code)

    # get suitable scaling factor
    _, scale, _, _ = scale_SI(np.min(k_size[k_size != 0]))

    print_grid_size(kgrid, scale)
    print_max_supported_freq(kgrid, c_min)


def get_min_sound_speed(medium, is_elastic_code):
    # if using the elastic code,
    # get the minimum sound speeds (not including zero if set for the shear speed)
    if not is_elastic_code:
        c_min = np.min(medium.sound_speed)
        return c_min, None, None
    else:  # pragma: no cover
        c_min_comp = np.min(medium.sound_speed_compression)
        c_min_shear = np.min(medium.sound_speed_shear[medium.sound_speed_shear != 0])
        return None, c_min_comp, c_min_shear


def print_grid_size(kgrid, scale):
    k_size = kgrid.size

    grid_size_pts = [int(kgrid.Nx)]
    if kgrid.dim >= 2:
        grid_size_pts.append(int(kgrid.Ny))
    if kgrid.dim == 3:
        grid_size_pts.append(int(kgrid.Nz))

    if kgrid.dim == 1:
        grid_size_scale = [scale_SI(k_size[0])[0]]
    elif kgrid.dim == 2:
        grid_size_scale = [k_size[0] * scale, k_size[1] * scale]
    elif kgrid.dim == 3:
        grid_size_scale = [round(k_size[0] * scale, 4), round(k_size[1] * scale, 4), round(k_size[2] * scale, 4)]
    else:
        raise NotImplementedError

    grid_size_str = " by ".join(map(str, grid_size_pts))
    grid_scale_str = " by ".join(map(str, grid_size_scale))

    # display grid size
    logging.log(logging.INFO, f"  input grid size: {grid_size_str} grid points ({grid_scale_str}m)")


def print_max_supported_freq(kgrid, c_min):
    # display the grid size and maximum supported frequency
    k_max, k_max_all = kgrid.k_max, kgrid.k_max_all

    def max_freq_str(kfreq):
        scaled_num_str, _, _, _ = scale_SI(kfreq * c_min / (2 * np.pi))
        return scaled_num_str

    if kgrid.dim == 1:
        # display maximum supported frequency
        logging.log(logging.INFO, f"  maximum supported frequency: {max_freq_str(k_max_all)}Hz")

    elif kgrid.dim == 2:
        # display maximum supported frequency
        if k_max.x == k_max.y:
            logging.log(logging.INFO, f"  maximum supported frequency: {max_freq_str(k_max_all)}Hz")
        else:
            logging.log(logging.INFO, f"  maximum supported frequency: {max_freq_str(k_max.x)}Hz by {max_freq_str(k_max.y)}Hz")

    elif kgrid.dim == 3:
        # display maximum supported frequency
        if k_max.x == k_max.z and k_max.x == k_max.y:
            logging.log(logging.INFO, f"  maximum supported frequency: {max_freq_str(k_max_all)}Hz")
        else:
            logging.log(
                logging.INFO,
                f"  maximum supported frequency: {max_freq_str(k_max.x)}Hz by {max_freq_str(k_max.y)}Hz by {max_freq_str(k_max.z)}Hz",
            )
