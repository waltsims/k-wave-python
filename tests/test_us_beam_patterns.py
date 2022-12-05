"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
from tempfile import gettempdir

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import *
from tests.diff_utils import compare_against_ref


def test_us_beam_patterns():
    # pathname for the input and output files
    pathname = gettempdir()

    # simulation settings
    DATA_CAST = 'single'       # set to 'single' or 'gpuArray-single' to speed up computations
    MASK_PLANE = 'xy'          # set to 'xy' or 'xz' to generate the beam pattern in different planes
    USE_STATISTICS = True      # set to true to compute the rms or peak beam patterns, set to false to compute the harmonic beam patterns

    # =========================================================================
    # DEFINE THE K-WAVE GRID
    # =========================================================================

    # set the size of the perfectly matched layer (PML)
    PML_X_SIZE = 20            # [grid points]
    PML_Y_SIZE = 10            # [grid points]
    PML_Z_SIZE = 10            # [grid points]

    # set total number of grid points not including the PML
    Nx = 128 - 2*PML_X_SIZE    # [grid points]
    Ny = 64 - 2*PML_Y_SIZE     # [grid points]
    Nz = 64 - 2*PML_Z_SIZE     # [grid points]

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
    transducer_spec.position = np.array([1, Ny//2 - transducer_width//2, Nz//2 - transducer_spec.element_length//2])

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
    sensor.mask = np.zeros((Nx, Ny, Nz))

    if MASK_PLANE == 'xy':
        # define mask
        sensor.mask[:, :, Nz//2 - 1] = 1

        # store y axis properties
        Nj = Ny
        j_vec = kgrid.y_vec
        j_label = 'y'

    if MASK_PLANE == 'xz':
        # define mask
        sensor.mask[:, Ny//2 - 1, :] = 1

        # store z axis properties
        Nj = Nz
        j_vec = kgrid.z_vec
        j_label = 'z'

    # set the record mode such that only the rms and peak values are stored
    if USE_STATISTICS:
        sensor.record = ['p_rms', 'p_max']

    # =========================================================================
    # RUN THE SIMULATION
    # =========================================================================

    # set the input settings
    input_filename = f'example_beam_pat'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename + '_input.h5')
    input_args = {
        'pml_inside': False,
        'pml_size': [PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE],
        'data_cast': DATA_CAST,
        'data_recast': True,
        'save_to_disk': True,
        'data_name': input_filename,
        'data_path': gettempdir(),
        'save_to_disk_exit': True
    }

    # stream the data to disk in blocks of 100 if storing the complete time history
    if not USE_STATISTICS:
        input_args['stream_to_disk'] = 100

    # run the simulation
    kspaceFirstOrder3DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': not_transducer,
        'sensor': sensor,
        **input_args
    })
    assert compare_against_ref(f'out_us_beam_patterns', input_file_full_path), \
        'Files do not match!'
