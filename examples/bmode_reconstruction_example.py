import logging
import os
import time
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
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst, get_win
from kwave.utils.filters import gaussian_filter
from kwave.reconstruction.tools import log_compression, db
from kwave.reconstruction.beamform import envelope_detection



def main():
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
    transducer.position = np.round([
        1,
        grid_size_points.y / 2 - transducer_width / 2,
        grid_size_points.z / 2 - transducer.element_length / 2
    ])

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

    logging.log(logging.INFO, "Fetching phantom data...")
    download_from_gdrive_if_does_not_exist(PHANTOM_DATA_GDRIVE_ID, phantom_data_path)

    phantom = scipy.io.loadmat(phantom_data_path)
    sound_speed_map = phantom['sound_speed_map']
    density_map = phantom['density_map']

    logging.log(logging.INFO, f"RUN_SIMULATION set to {RUN_SIMULATION}")

    # preallocate the storage set medium position
    scan_lines = np.zeros((number_scan_lines, kgrid.Nt))
    medium_position = 0

    for scan_line_index in range(0, number_scan_lines):

        # load the current section of the medium
        medium.sound_speed = \
            sound_speed_map[:, medium_position:medium_position + grid_size_points.y, :]
        medium.density = density_map[:, medium_position:medium_position + grid_size_points.y, :]

        # set the input settings
        input_filename = f'example_input_{scan_line_index}.h5'
        input_file_full_path = os.path.join(pathname, input_filename) # noqa: F841
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

            scan_lines[scan_line_index, :] = not_transducer.scan_line(not_transducer.combine_sensor_data(sensor_data['p'].T))

        # update medium position
        medium_position = medium_position + transducer.element_width

    if RUN_SIMULATION:
        simulation_data = scan_lines
        scipy.io.savemat('sensor_data.mat', {'sensor_data_all_lines': simulation_data})

    else:
        logging.log(logging.INFO, "Downloading data from remote server...")
        SENSOR_DATA_GDRIVE_ID = '1lGFTifpOrzBYT4Bl_ccLu_Kx0IDxM0Lv'
        sensor_data_path = 'sensor_data.mat'
        download_from_gdrive_if_does_not_exist(SENSOR_DATA_GDRIVE_ID, sensor_data_path)

        simulation_data = scipy.io.loadmat(sensor_data_path)['sensor_data_all_lines']

        # temporary fix for dimensionality
    import matplotlib.pyplot as plt

    scan_lines = simulation_data


    # =========================================================================
    # PROCESS THE RESULTS
    # =========================================================================

    # -----------------------------
    # Remove Input Signal
    # -----------------------------
    # Trim the delay offset from the scan line data
    t0_offset = 85 # round(len(input_signal) / 2) + (not_transducer.stored_appended_zeros - not_transducer.beamforming_delays_offset)
    scan_lines = scan_lines[:, t0_offset:]

    # Get the new length of the scan lines
    Nt = len(scan_lines[0, :])

    # TODO: fix this
    scan_line_win, cg = get_win(N= Nt* 2,type_='Tukey', plot_win=False,param=0.05)
    # scan_line_win = np.concatenate((np.zeros([1, t0_offset * 2]), scan_line_win.T[:, :Nt - t0_offset * 2]), axis=1)

    # -----------------------------
    # Time Gain Compensation
    # -----------------------------

    # Create radius variable
    r = c0 * np.arange(1, Nt + 1) * kgrid.dt / 2

    # Define absorption value and convert to correct units
    tgc_alpha_db_cm = medium.alpha_coeff * (tone_burst_freq * 1e-6)**medium.alpha_power
    tgc_alpha_np_m = tgc_alpha_db_cm / 8.686 * 100

    # Create time gain compensation function
    tgc = np.exp(tgc_alpha_np_m * 2 * r)

    # Apply the time gain compensation to each of the scan lines
    scan_lines *= tgc

    # -----------------------------
    # Frequency Filtering
    # -----------------------------

    scan_lines_fund = gaussian_filter(scan_lines, 1/kgrid.dt, tone_burst_freq, 100)
    scan_lines_harm = gaussian_filter(scan_lines, 1/kgrid.dt, 2 * tone_burst_freq, 30)  # plotting was not impl.

    # -----------------------------
    # Envelope Detection
    # -----------------------------

    scan_lines_fund = envelope_detection(scan_lines_fund)
    scan_lines_harm = envelope_detection(scan_lines_harm)

    # -----------------------------
    # Log Compression
    # -----------------------------

    compression_ratio = 3
    scan_lines_fund = log_compression(scan_lines_fund, compression_ratio, True)
    scan_lines_harm = log_compression(scan_lines_harm, compression_ratio, True)

    # =========================================================================
    # VISUALISATION
    # =========================================================================

    # Set the desired size of the image
    image_size = kgrid.size 
    harm_img = db(scan_lines_harm.T)
    fund_img = db(scan_lines_fund.T)

    # Create the axis variables
    x_axis = [0, image_size[0] * 1e3] # [mm]      
    y_axis = [-0.5 * image_size[0] * 1e3, 0.5 * image_size[1] * 1e3]  # [mm]
    # make plotting non-blocking
    plt.ion()
    # Plot the data before and after scan conversion
    plt.figure()
    # plot the sound speed map
    plt.subplot(1, 3, 1)
    plt.imshow(sound_speed_map[int(t0_offset/4):, 64:-64, int(grid_size_points.z / 2)], aspect='auto',
                extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]])
    plt.title('Sound Speed Map')
    plt.xlabel('Image width [mm]')
    plt.ylabel('Depth [mm]')
    plt.subplot(1, 3, 2)
    plt.imshow(fund_img, cmap='bone', aspect='auto',  extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]],
               vmax=np.max(fund_img), vmin=np.max(fund_img)-40)
    plt.xlabel('Image width [mm]')
    plt.title('Fundamental')
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(harm_img, cmap='bone', aspect='auto', vmax=np.max(harm_img), vmin=np.max(harm_img)-40,
                extent=[y_axis[0], y_axis[1], x_axis[1], x_axis[0]])
    plt.yticks([])
    plt.xlabel('Image width [mm]')
    plt.title('Harmonic')
    
    plt.show()
    
    # sleep for 1 min and then close all figures for CI completion
    plt.pause(60)
    plt.close('all')



if __name__ == '__main__':
    main()
