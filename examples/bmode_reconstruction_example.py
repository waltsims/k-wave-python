import os
from tempfile import gettempdir

import numpy as np
import scipy.io
from kwave.options import SimulationOptions, SimulationExecutionOptions

from example_utils import download_from_gdrive_if_does_not_exist
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.reconstruction.beamform import beamform
from kwave.reconstruction.converter import build_channel_data
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst

if __name__ == '__main__':
    # pathname for the input and output files
    pathname = gettempdir()

    # simulation settings
    DATA_CAST = 'single'
    RUN_SIMULATION = False

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================
    print("Setting up the k-wave grid...")

    # set the size of the perfectly matched layer (PML)
    PML_X_SIZE = 20            # [grid points]
    PML_Y_SIZE = 10            # [grid points]
    PML_Z_SIZE = 10            # [grid points]

    # set total number of grid points not including the PML
    Nx = 256 - 2*PML_X_SIZE    # [grid points]
    Ny = 128 - 2*PML_Y_SIZE    # [grid points]
    Nz = 128 - 2*PML_Z_SIZE     # [grid points]

    # set desired grid size in the x-direction not including the PML
    x = 40e-3                  # [m]

    # calculate the spacing between the grid points
    dx = x/Nx                  # [m]
    dy = dx                    # [m]
    dz = dx                    # [m]

    # create the k-space grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================
    # define the properties of the propagation medium
    c0 = 1540
    rho0 = 1000

    medium = kWaveMedium(
        sound_speed=None,  # will be set later
        alpha_coeff=0.75,
        alpha_power=1.5,
        BonA=6
    )

    # create the time array
    t_end = (Nx * dx) * 2.2 / c0   # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================
    print("Defining the input signal...")

    # define properties of the input signal
    source_strength = 1e6          # [Pa]
    tone_burst_freq = 1.5e6        # [Hz]
    tone_burst_cycles = 4

    # create the input signal using tone_burst
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)

    # scale the source magnitude by the source_strength divided by the
    # impedance (the source is assigned to the particle velocity)
    input_signal = (source_strength / (c0 * rho0)) * input_signal

    # =========================================================================
    # DEFINE THE ULTRASOUND TRANSDUCER
    # =========================================================================
    print("Setting up the transducer configuration...")

    # physical properties of the transducer
    transducer = dotdict()
    transducer.number_elements = 32    # total number of transducer elements
    transducer.element_width = 2       # width of each element [grid points/voxels]
    transducer.element_length = 24     # length of each element [grid points/voxels]
    transducer.element_spacing = 0     # spacing (kerf  width) between the elements [grid points/voxels]
    transducer.radius = float('inf')   # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = transducer.number_elements * transducer.element_width + (transducer.number_elements - 1) * transducer.element_spacing

    # use this to position the transducer in the middle of the computational grid
    transducer.position = np.round([1, Ny/2 - transducer_width/2, Nz/2 - transducer.element_length/2])

    # properties used to derive the beamforming delays
    not_transducer = dotdict()
    not_transducer.sound_speed = c0                    # sound speed [m/s]
    not_transducer.focus_distance = 20e-3              # focus distance [m]
    not_transducer.elevation_focus_distance = 19e-3    # focus distance in the elevation plane [m]
    not_transducer.steering_angle = 0                  # steering angle [degrees]

    # apodization
    not_transducer.transmit_apodization = 'Hanning'
    not_transducer.receive_apodization = 'Rectangular'

    # define the transducer elements that are currently active
    not_transducer.active_elements = np.ones((transducer.number_elements, 1))

    # append input signal used to drive the transducer
    not_transducer.input_signal = input_signal

    # create the transducer using the defined settings
    transducer = kWaveTransducerSimple(kgrid, **transducer)
    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

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
    sound_speed_map     = phantom['sound_speed_map']
    density_map         = phantom['density_map']

    # =========================================================================
    # RUN THE SIMULATION
    # =========================================================================
    print(f"RUN_SIMULATION set to {RUN_SIMULATION}")
    # run the simulation if set to true, otherwise, load previous results from disk
    if RUN_SIMULATION:
        print("Running simulation locally...")

        # set medium position
        medium_position = 0

        # preallocate the storage
        simulation_data = []

        # loop through the scan lines
        for scan_line_index in range(1, number_scan_lines + 1):
            # for scan_line_index in range(1, 10):
            # update the command line status
            print(f'Computing scan line {scan_line_index} of {number_scan_lines}')

            # load the current section of the medium
            medium.sound_speed = sound_speed_map[:, medium_position:medium_position + Ny, :]
            medium.density = density_map[:, medium_position:medium_position + Ny, :]

            # set the input settings
            input_filename  = f'example_input_{scan_line_index}.h5'
            input_file_full_path = os.path.join(pathname, input_filename)
            # set the input settings
            simulation_options = SimulationOptions(
                pml_inside=False,
                pml_size=[PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE],
                data_cast=DATA_CAST,
                data_recast=True,
                save_to_disk=True,
                input_filename=input_filename,
                save_to_disk_exit=False
            )
            # run the simulation
            sensor_data = kspaceFirstOrder3DC(
                medium=medium,
                kgrid=kgrid,
                source=not_transducer,
                sensor=not_transducer,
                simulation_options=simulation_options,
                execution_options=SimulationExecutionOptions()
            )
            simulation_data.append(sensor_data)

            # update medium position
            medium_position = medium_position + transducer.element_width

        simulation_data = np.stack(simulation_data, axis=0)
        scipy.io.savemat('sensor_data.mat', {'sensor_data_all_lines': simulation_data})

    else:
        print("Downloading data from remote server...")
        SENSOR_DATA_GDRIVE_ID = '168wACeJOyV9urSlf7Q_S8dMnpvRNsc9C'
        sensor_data_path = 'sensor_data.mat'
        download_from_gdrive_if_does_not_exist(SENSOR_DATA_GDRIVE_ID, sensor_data_path)

        simulation_data = scipy.io.loadmat(sensor_data_path)['sensor_data_all_lines']

    # temporary fix for dimensionality
    simulation_data = simulation_data[None, :]

    sampling_frequency = 2.772000000000000e+07
    prf = 10000
    focal_depth = 0.020000000000000
    channel_data = build_channel_data(simulation_data, kgrid, not_transducer,
                                      sampling_frequency, prf, focal_depth)

    print("Beamforming channel data and reconstructing the image...")
    beamform(channel_data)
