import os
from tempfile import gettempdir

import numpy as np
import scipy.io

from example_utils import download_from_gdrive_if_does_not_exist
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.reconstruction.beamform import beamform
from kwave.reconstruction.converter import build_channel_data
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst

# Define the pathname for the input and output files
pathname = gettempdir()

# Define the simulation settings
data_cast = 'single'
run_simulation = False

# Define the k-wave grid
pml_sizes = [20, 10, 10]  # [grid points]
grid_sizes = [256, 128, 128]  # [grid points]
dx = 40e-3 / (np.array(grid_sizes) - 2 * np.array(pml_sizes))
kgrid = kWaveGrid(grid_sizes, dx)

# Define the medium parameters
medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)
t_end = (np.prod(grid_sizes) * np.prod(dx)) * 2.2 / medium.sound_speed
kgrid.makeTime(medium.sound_speed, t_end=t_end)

# Define the input signal
source_strength = 1e6  # [Pa]
tone_burst_freq = 1.5e6  # [Hz]
tone_burst_cycles = 4
input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
input_signal *= source_strength / (medium.sound_speed * medium.density)

# Define the ultrasound transducer
transducer = dotdict({
    'number_elements': 32,
    'element_width': 2,
    'element_length': 24,
    'element_spacing': 0,
    'radius': float('inf'),
})
transducer_width = transducer.number_elements * transducer.element_width + (
            transducer.number_elements - 1) * transducer.element_spacing
transducer_position = [1, (grid_sizes[1] - transducer_width) / 2, (grid_sizes[2] - transducer.element_length) / 2]
not_transducer = dotdict({
    'sound_speed': medium.sound_speed,
    'focus_distance': 20e-3,
    'elevation_focus_distance': 19e-3,
    'steering_angle': 0,
    'transmit_apodization': 'Hanning',
    'receive_apodization': 'Rectangular',
    'active_elements': np.ones((transducer.number_elements, 1)),
    'input_signal': input_signal,
})
transducer = kWaveTransducerSimple(kgrid, **transducer)
not_transducer = NotATransducer(transducer, kgrid)

# =========================================================================
# DEFINE THE MEDIUM PROPERTIES
# =========================================================================
# define a large image size to move across
number_scan_lines = 96

print("Fetching phantom data...")
phantom_data_path = 'phantom_data.mat'
PHANTOM_DATA_GDRIVE_ID = '1ZfSdJPe8nufZHz0U9IuwHR4chaOGAWO4'
download_from_gdrive_if_does_not_exist(PHANTOM_DATA_GDRIVE_ID, phantom_data_path)

phantom = scipy.io.loadmat(phantom_data_path)
sound_speed_map = phantom['sound_speed_map']
density_map = phantom['density_map']

# =========================================================================
# RUN THE SIMULATION
# =========================================================================
if run_simulation:
    print("Running the simulation...")

    # assign the medium parameters
    medium.set_medium(kgrid)

    # define the sensor mask covering the entire grid
    sensor_mask = np.zeros(grid_sizes)
    sensor_mask[:, :, :] = 1

    # run the simulation
    input_args = {
        'PMLSize': pml_sizes,
        'PlotPML': False,
        'PMLInside': False,
        'DataCast': data_cast,
        'PMLAlpha': 2,
        'PMLSizeLateral': 1.5,
        'PMLSizeTop': 1.5,
        'PMLSizeBottom': 1.5,
        'RecordMovie': False,
        'SaveToDisk': True,
        'MovieArgs': {
            'SaveGif': False,
            'FileName': os.path.join(pathname, 'movie.gif')
        }
    }

    sensor_data = kspaceFirstOrder3DC(kgrid, medium, not_transducer, sensor_mask, **input_args)

    # set the filename of the HDF5 file
    file_name = os.path.join(pathname, 'sensor_data.mat')

    # save the sensor data
    scipy.io.savemat(file_name, {'sensor_data': sensor_data})

# download the data if the file does not exist
download_from_gdrive_if_does_not_exist(file_name, '1oLI4wx4OmGdqyJi32a8nXkEjL7PzyYfu')

# load the data
sensor_data = scipy.io.loadmat(file_name)['sensor_data']

# build channel data
channel_data = build_channel_data(sensor_data)

# beamforming
not_transducer.update_data(channel_data)
beamform(kgrid, medium, not_transducer, None)  # Beamform and plot
