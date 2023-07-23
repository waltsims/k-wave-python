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

# noinspection PyUnresolvedReferences
import setup_test
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst
from tests.diff_utils import compare_against_ref


def test_us_defining_transducer():
    # input and output filenames (these must have the .h5 extension)
    input_filename = 'example_input.h5'
    output_filename = 'example_output.h5'

    # pathname for the input and output files
    pathname = gettempdir()

    # remove input file if it already exists
    input_file_full_path = os.path.join(pathname, input_filename)
    output_file_full_path = os.path.join(pathname, output_filename)
    if os.path.exists(input_file_full_path):
        os.remove(input_file_full_path)

    # simulation settings
    DATA_CAST = 'single'

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    pml_size = Vector([20, 10, 10])  # [grid points]

    # set total number of grid points not including the PML
    grid_size_points = Vector([128, 128, 64]) - 2 * pml_size  # [grid points]

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
    medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    # create the time array
    t_end = 40e-6                  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================

    # define properties of the input signal
    source_strength = 1e6           # [Pa]
    tone_burst_freq = 0.5e6         # [Hz]
    tone_burst_cycles = 5

    # create the input signal using tone_burst
    input_signal = tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)

    # scale the source magnitude by the source_strength divided by the
    # impedance (the source is assigned to the particle velocity)
    input_signal = (source_strength / (medium.sound_speed * medium.density)) * input_signal

    # =========================================================================
    # DEFINE THE ULTRASOUND TRANSDUCER
    # =========================================================================

    # physical properties of the transducer
    transducer = dotdict()
    transducer.number_elements = 72    # total number of transducer elements
    transducer.element_width = 1       # width of each element [grid points/voxels]
    transducer.element_length = 12     # length of each element [grid points/voxels]
    transducer.element_spacing = 0     # spacing (kerf  width) between the elements [grid points/voxels]
    transducer.radius = float('inf')   # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = transducer.number_elements * transducer.element_width + (transducer.number_elements - 1) * transducer.element_spacing

    # use this to position the transducer in the middle of the computational grid
    transducer.position = np.round([1, grid_size_points.y/2 - transducer_width/2, grid_size_points.z/2 - transducer.element_length/2])

    # properties used to derive the beamforming delays
    not_transducer = dotdict()
    not_transducer.sound_speed = 1540                  # sound speed [m/s]
    not_transducer.focus_distance = 20e-3              # focus distance [m]
    not_transducer.elevation_focus_distance = 19e-3    # focus distance in the elevation plane [m]
    not_transducer.steering_angle = 0                  # steering angle [degrees]

    # apodization
    not_transducer.transmit_apodization = 'Rectangular'
    not_transducer.receive_apodization = 'Rectangular'

    # define the transducer elements that are currently active
    not_transducer.active_elements = np.zeros((transducer.number_elements, 1))
    not_transducer.active_elements[20:52] = 1

    # append input signal used to drive the transducer
    not_transducer.input_signal = input_signal

    # create the transducer using the defined settings
    transducer = kWaveTransducerSimple(kgrid, **transducer)
    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

    # print out transducer properties
    # transducer.properties;

    # =========================================================================
    # DEFINE SENSOR MASK
    # =========================================================================

    # create a binary sensor mask with four detection positions
    sensor_mask = np.zeros(grid_size_points)
    x_locs = (np.array([1/4, 2/4, 3/4]) * grid_size_points.x).astype(int)
    sensor_mask[x_locs - 1, grid_size_points.y // 2 - 1, grid_size_points.z // 2 - 1] = 1  # -1 compatibility
    sensor = kSensor(sensor_mask)

    # set the input settings
    input_filename = f'example_def_tran_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=np.array(pml_size),
        data_cast=DATA_CAST,
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
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )

    # display the required syntax to run the C++ simulation
    print(f'Using a terminal window, navigate to the {os.path.sep}binaries folder of the k-Wave Toolbox')
    print('Then, use the syntax shown below to run the simulation:')
    if os.name == 'posix':
        print(f'./kspaceFirstOrder-OMP -i {input_file_full_path} -o {output_file_full_path} --p_final --p_max')
    else:
        print(f'kspaceFirstOrder-OMP.exe -i {input_file_full_path} -o {output_file_full_path} --p_final --p_max')

    assert compare_against_ref('out_us_defining_transducer', input_file_full_path), 'Files do not match!'

    # extract a single scan line from the sensor data using the current
    # beamforming settings
    # scan_line = transducer.scan_line(sensor_data)
