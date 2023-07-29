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
import setup_test  # noqa: F401
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst
from tests.diff_utils import compare_against_ref


def test_us_beam_patterns():
    # simulation settings
    DATA_CAST = 'single'       # set to 'single' or 'gpuArray-single' to speed up computations
    MASK_PLANE = 'xy'          # set to 'xy' or 'xz' to generate the beam pattern in different planes
    USE_STATISTICS = True      # set to true to compute the rms or peak beam patterns, set to false to compute the harmonic beam patterns

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    pml_size_points = Vector([20, 10, 10])  # [grid points]

    # set total number of grid points not including the PML
    grid_size_points = Vector([128, 64, 64]) - 2 * pml_size_points  # [grid points]

    # set desired grid size in the x-direction not including the PML
    grid_size_meters = 40e-3                  # [m]

    # calculate the spacing between the grid points
    grid_spacing_meters = grid_size_meters / Vector([grid_size_points.x, grid_size_points.x, grid_size_points.x])

    # create the k-space grid
    kgrid = kWaveGrid(grid_size_points, grid_spacing_meters)

    # =========================================================================
    # DEFINE THE MEDIUM PARAMETERS
    # =========================================================================

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1540, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    # create the time array
    t_end = 45e-6                  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE THE INPUT SIGNAL
    # =========================================================================

    # define properties of the input signal
    source_strength = 1e6          # [Pa]
    tone_burst_freq = 0.5e6    	   # [Hz]
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
    transducer_spec = dotdict()
    transducer_spec.number_elements = 32    # total number of transducer elements
    transducer_spec.element_width = 1       # width of each element [grid points]
    transducer_spec.element_length = 12     # length of each element [grid points]
    transducer_spec.element_spacing = 0     # spacing (kerf  width) between the elements [grid points]
    transducer_spec.radius = float('inf')   # radius of curvature of the transducer [m]

    # calculate the width of the transducer in grid points
    transducer_width = transducer_spec.number_elements * transducer_spec.element_width + (transducer_spec.number_elements - 1) * transducer_spec.element_spacing

    # use this to position the transducer in the middle of the computational grid
    transducer_spec.position = np.array([1, grid_size_points.y//2 - transducer_width//2, grid_size_points.z//2 - transducer_spec.element_length//2])

    # properties used to derive the beamforming delays
    not_transducer_spec = dotdict()
    not_transducer_spec.sound_speed = 1540                  # sound speed [m/s]
    not_transducer_spec.focus_distance = 20e-3              # focus distance [m]
    not_transducer_spec.elevation_focus_distance = 19e-3    # focus distance in the elevation plane [m]
    not_transducer_spec.steering_angle = 0                  # steering angle [degrees]

    # apodization
    not_transducer_spec.transmit_apodization = 'Rectangular'
    not_transducer_spec.receive_apodization = 'Rectangular'

    # define the transducer elements that are currently active
    not_transducer_spec.active_elements = np.ones((transducer_spec.number_elements, 1))

    # append input signal used to drive the transducer
    not_transducer_spec.input_signal = input_signal

    # create the transducer using the defined settings
    transducer = kWaveTransducerSimple(kgrid, **transducer_spec)
    not_transducer = NotATransducer(transducer, kgrid, **not_transducer_spec)

    # print out transducer properties
    # transducer.properties

    # =========================================================================
    # DEFINE SENSOR MASK
    # =========================================================================

    # define a sensor mask through the central plane
    sensor = kSensor()
    sensor.mask = np.zeros(grid_size_points)

    if MASK_PLANE == 'xy':
        # define mask
        sensor.mask[:, :, grid_size_points.z//2 - 1] = 1

        # store y axis properties
        Nj = grid_size_points.y
        j_vec = kgrid.y_vec
        j_label = 'y'

    if MASK_PLANE == 'xz':
        # define mask
        sensor.mask[:, grid_size_points.y//2 - 1, :] = 1

        # store z axis properties
        Nj = grid_size_points.z  # noqa: F841
        j_vec = kgrid.z_vec  # noqa: F841
        j_label = 'z'  # noqa: F841

    # set the record mode such that only the rms and peak values are stored
    if USE_STATISTICS:
        sensor.record = ['p_rms', 'p_max']

    # =========================================================================
    # RUN THE SIMULATION
    # =========================================================================

    # set the input settings
    input_filename = 'example_beam_pat_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=pml_size_points,
        data_cast=DATA_CAST,
        data_recast=True,
        save_to_disk=True,
        input_filename=input_filename,
        save_to_disk_exit=True,
        data_path=pathname
    )

    # stream the data to disk in blocks of 100 if storing the complete time history
    if not USE_STATISTICS:
        simulation_options.stream_to_disk = 100

    # run the simulation
    kspaceFirstOrder3DC(
        medium=medium,
        kgrid=kgrid,
        source=not_transducer,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions()
    )
    assert compare_against_ref('out_us_beam_patterns', input_file_full_path), \
        'Files do not match!'
