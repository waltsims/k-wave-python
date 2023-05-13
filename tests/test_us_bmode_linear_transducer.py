"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
import os
from tempfile import gettempdir

import numpy as np

from kwave.data import Vector
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.utils.dotdictionary import dotdict
from kwave.utils.mapgen import make_ball
from kwave.utils.signals import tone_burst
from tests.diff_utils import compare_against_ref

"""
    Differences compared to original example:
        - "randn" function call was replaced by "ones" call when creating the following variables
            - background_map
            - scattering_map
"""


def test_us_bmode_linear_transducer():

    # simulation settings
    DATA_CAST = 'single'
    RUN_SIMULATION = True

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    pml_size = Vector([20, 10, 10])  # [grid points]

    # set total number of grid points not including the PML
    grid_size_points = Vector([256, 128, 128]) - 2 * pml_size  # [grid points]

    # set desired grid size in the x-direction not including the PML
    grid_size_meters = 40e-3                  # [m]

    # calculate the spacing between the grid points
    grid_spacing_meters = grid_size_meters / Vector([grid_size_points.x, grid_size_points.x, grid_size_points.x])  # [m]

    # create the k-space grid
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

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
    t_end = (grid_size_points.x * grid_spacing_meters.x) * 2.2 / c0   # [s]
    kgrid.makeTime(c0, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================

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
    transducer.position = np.round([1, grid_size_points.y/2 - transducer_width/2, grid_size_points.z/2 - transducer.element_length/2])


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
    N_tot = grid_size_points.copy()
    N_tot.y = N_tot.y + number_scan_lines * transducer.element_width

    # define a random distribution of scatterers for the medium
    background_map_mean = 1
    background_map_std = 0.008
    background_map = background_map_mean + background_map_std * np.ones(N_tot)  # randn([Nx_tot, Ny_tot, Nz_tot]) => is random in original example

    # define a random distribution of scatterers for the highly scattering region
    scattering_map = np.ones(N_tot)  # randn([Nx_tot, Ny_tot, Nz_tot]) => is random in original example
    scattering_c0 = c0 + 25 + 75 * scattering_map
    scattering_c0[scattering_c0 > 1600] = 1600
    scattering_c0[scattering_c0 < 1400] = 1400
    scattering_rho0 = scattering_c0 / 1.5

    # define properties
    sound_speed_map = c0 * np.ones(N_tot) * background_map
    density_map = rho0 * np.ones(N_tot) * background_map

    # when the division result is in the half-way (x.5), numpy will round it to the nearest multiple of 2.
    # This behaviour results in different results compared to Matlab.
    # To ensure similar results, we add epsilon to the division results
    rounding_eps = 1e-12
    dx = grid_spacing_meters.x

    # define a sphere for a highly scattering region
    radius = 6e-3       # [m]
    x_pos = 27.5e-3     # [m]
    y_pos = 20.5e-3     # [m]
    ball_location = Vector([round(x_pos / dx + rounding_eps), round(y_pos / dx + rounding_eps), N_tot.z / 2])
    scattering_region1 = make_ball(N_tot, ball_location, round(radius / dx + rounding_eps))

    # assign region
    sound_speed_map[scattering_region1 == 1] = scattering_c0[scattering_region1 == 1]
    density_map[scattering_region1 == 1] = scattering_rho0[scattering_region1 == 1]

    # define a sphere for a highly scattering region
    radius = 5e-3       # [m]
    x_pos = 30.5e-3     # [m]
    y_pos = 37e-3       # [m]
    ball_location = Vector([round(x_pos / dx + rounding_eps), round(y_pos / dx + rounding_eps), N_tot.z / 2])
    scattering_region2 = make_ball(N_tot, ball_location, round(radius / dx + rounding_eps))

    # assign region
    sound_speed_map[scattering_region2 == 1] = scattering_c0[scattering_region2 == 1]
    density_map[scattering_region2 == 1] = scattering_rho0[scattering_region2 == 1]

    # define a sphere for a highly scattering region
    radius = 4.5e-3     # [m]
    x_pos = 15.5e-3     # [m]
    y_pos = 30.5e-3     # [m]
    ball_location = Vector([round(x_pos / dx + rounding_eps), round(y_pos / dx + rounding_eps), N_tot.z / 2])
    scattering_region3 = make_ball(N_tot, ball_location, round(radius / dx + rounding_eps))

    # assign region
    sound_speed_map[scattering_region3 == 1] = scattering_c0[scattering_region3 == 1]
    density_map[scattering_region3 == 1] = scattering_rho0[scattering_region3 == 1]


    # =========================================================================
    # RUN THE SIMULATION
    # =========================================================================

    # preallocate the storage
    scan_lines = np.zeros((number_scan_lines, kgrid.Nt))

    # run the simulation if set to true, otherwise, load previous results from disk
    if RUN_SIMULATION:

        # set medium position
        medium_position = 0

        # loop through the scan lines
        # for scan_line_index in range(1, number_scan_lines + 1):
        for scan_line_index in range(1, 10):
            # update the command line status
            print(f'Computing scan line {scan_line_index} of {number_scan_lines}')

            # load the current section of the medium
            medium.sound_speed = sound_speed_map[:, medium_position:medium_position + grid_size_points.y, :]
            medium.density = density_map[:, medium_position:medium_position + grid_size_points.y, :]

            # set the input settings
            input_filename = f'example_lin_tran_input.h5'
            pathname = gettempdir()
            input_file_full_path = os.path.join(pathname, input_filename)
            simulation_options = SimulationOptions(
                pml_inside=False,
                pml_size=pml_size,
                data_cast=DATA_CAST,
                data_recast=True,
                save_to_disk=True,
                input_filename=input_filename,
                save_to_disk_exit=True,
                data_path=pathname
            )
            # run the simulation
            kspaceFirstOrder3DC(
                medium=medium,
                kgrid=kgrid,
                source=not_transducer,
                sensor=not_transducer,
                simulation_options=simulation_options,
                execution_options=SimulationExecutionOptions()
            )

            assert compare_against_ref(f'out_us_bmode_linear_transducer/input_{scan_line_index}', input_file_full_path, precision=6), \
                'Files do not match!'

            # extract the scan line from the sensor data
            # scan_lines(scan_line_index, :) = transducer.scan_line(sensor_data);

            # update medium position
            medium_position = medium_position + transducer.element_width

        # % save the scan lines to disk
        # save example_us_bmode_scan_lines scan_lines;
    else:
        raise NotImplementedError
