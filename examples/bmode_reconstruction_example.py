import os
from tempfile import gettempdir

import numpy as np
import scipy.io

from example_utils import download_from_gdrive_if_does_not_exist
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction.beamform import beamform
from kwave.reconstruction.converter import build_channel_data
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst

if __name__ == '__main__':
    # pathname for the input and output files
    pathname = gettempdir()
    phantom_data_path = 'phantom_data.mat'
    PHANTOM_DATA_GDRIVE_ID = '1ZfSdJPe8nufZHz0U9IuwHR4chaOGAWO4'

    # simulation settings
    DATA_CAST = 'single'
    RUN_SIMULATION = False

    pml_size_points = Vector([20, 10, 10])  # [grid points]
    grid_size_points = Vector([256, 128, 128]) - 2 * pml_size_points  # [grid points]
    grid_size_meters = 40e-3  # [m]
    grid_spacing_meters = grid_size_meters / Vector([grid_size_points.x, grid_size_points.x, grid_size_points.x])

    c0 = 1540
    rho0 = 1000
    source_strength = 1e6  # [Pa]
    tone_burst_freq = 1.5e6  # [Hz]
    tone_burst_cycles = 4

    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)
    t_end = (grid_size_points.x * grid_spacing_meters.x) * 2.2 / c0  # [s]
    kgrid.makeTime(c0, t_end=t_end)

    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)
    input_signal = (source_strength / (c0 * rho0)) * input_signal

    medium = kWaveMedium(
        sound_speed=None,  # will be set later
        alpha_coeff=0.75,
        alpha_power=1.5,
        BonA=6
    )

    transducer = dotdict()
    transducer.number_elements = 32  # total number of transducer elements
    transducer.element_width = 2  # width of each element [grid points/voxels]
    transducer.element_length = 24  # length of each element [grid points/voxels]
    transducer.element_spacing = 0  # spacing (kerf  width) between the elements [grid points/voxels]
    transducer.radius = float('inf')  # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = transducer.number_elements * transducer.element_width + (
            transducer.number_elements - 1) * transducer.element_spacing

    # use this to position the transducer in the middle of the computational grid
    transducer.position = np.round(
        [1, grid_size_points.y / 2 - transducer_width / 2,
         grid_size_points.z / 2 - transducer.element_length / 2])

    transducer = kWaveTransducerSimple(kgrid, **transducer)

    # properties used to derive the beamforming delays
    not_transducer = dotdict()
    not_transducer.sound_speed = c0  # sound speed [m/s]
    not_transducer.focus_distance = 20e-3  # focus distance [m]
    not_transducer.elevation_focus_distance = 19e-3  # focus distance in the elevation plane [m]
    not_transducer.steering_angle = 0  # steering angle [degrees]
    not_transducer.transmit_apodization = 'Hanning'
    not_transducer.receive_apodization = 'Rectangular'
    not_transducer.active_elements = np.ones((transducer.number_elements, 1))
    not_transducer.input_signal = input_signal

    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

    number_scan_lines = 96

    print("Fetching phantom data...")
    download_from_gdrive_if_does_not_exist(PHANTOM_DATA_GDRIVE_ID, phantom_data_path)

    phantom = scipy.io.loadmat(phantom_data_path)
    sound_speed_map = phantom['sound_speed_map']
    density_map = phantom['density_map']

    print(f"RUN_SIMULATION set to {RUN_SIMULATION}")

    # preallocate the storage set medium position
    scan_lines = np.zeros((number_scan_lines, not_transducer.number_active_elements, kgrid.Nt))
    medium_position = 0

    for scan_line_index in range(0, number_scan_lines):

        # load the current section of the medium
        medium.sound_speed = \
            sound_speed_map[:, medium_position:medium_position + grid_size_points.y, :]
        medium.density = density_map[:, medium_position:medium_position + grid_size_points.y, :]

        # set the input settings
        input_filename = f'example_input_{scan_line_index}.h5'
        input_file_full_path = os.path.join(pathname, input_filename)
        # set the input settings
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=pml_size_points,
            data_cast=DATA_CAST,
            data_recast=True,
            save_to_disk=True,
            input_filename=input_filename,
            save_to_disk_exit=False
        )
        # run the simulation
        if RUN_SIMULATION:
            sensor_data = kspaceFirstOrder3D(
                medium=medium,
                kgrid=kgrid,
                source=not_transducer,
                sensor=not_transducer,
                simulation_options=simulation_options,
                execution_options=SimulationExecutionOptions(is_gpu_simulation=True)
            )

            scan_lines[scan_line_index, :] = not_transducer.combine_sensor_data(sensor_data['p'].T)

        # update medium position
        medium_position = medium_position + transducer.element_width

    if RUN_SIMULATION:
        simulation_data = scan_lines
        # scipy.io.savemat('sensor_data.mat', {'sensor_data_all_lines': simulation_data})

    else:
        print("Downloading data from remote server...")
        SENSOR_DATA_GDRIVE_ID = '168wACeJOyV9urSlf7Q_S8dMnpvRNsc9C'
        sensor_data_path = 'sensor_data.mat'
        download_from_gdrive_if_does_not_exist(SENSOR_DATA_GDRIVE_ID, sensor_data_path)

        simulation_data = scipy.io.loadmat(sensor_data_path)['sensor_data_all_lines']

    # temporary fix for dimensionality
    simulation_data = simulation_data[None, :]

    sampling_frequency = 2.772e+07
    prf = 1e4
    focal_depth = 20e-3  # [m]
    channel_data = build_channel_data(simulation_data, kgrid, not_transducer,
                                      sampling_frequency, prf, focal_depth)

    print("Beamforming channel data and reconstructing the image...")
    beamform(channel_data)
