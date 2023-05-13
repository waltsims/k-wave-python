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
from kwave.options import SimulationOptions, SimulationExecutionOptions

# noinspection PyUnresolvedReferences
import setup_test
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.utils.dotdictionary import dotdict
from kwave.utils.mapgen import make_ball
from kwave.utils.signals import tone_burst
from tests.diff_utils import compare_against_ref


def test_us_transducer_as_sensor():
    # input and output filenames (these must have the .h5 extension)
    input_filename  = 'example_input.h5'
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
    PML_X_SIZE = 20            # [grid points]
    PML_Y_SIZE = 10            # [grid points]
    PML_Z_SIZE = 10            # [grid points]

    # set total number of grid points not including the PML
    Nx = 128 - 2*PML_X_SIZE    # [grid points]
    Ny = 128 - 2*PML_Y_SIZE    # [grid points]
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
    medium = kWaveMedium(sound_speed=1500, density=1000, alpha_coeff=0.75, alpha_power=1.5, BonA=6)

    # create the time array
    t_end = 40e-6                  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE THE SOURCE
    # =========================================================================

    # create source mask
    source = kSource()
    source.p_mask = make_ball(Nx, Ny, Nz, round(25e-3 / dx), Ny / 2, Nz / 2, 3) + make_ball(Nx, Ny, Nz, round(8e-3 / dx), Ny / 4, Nz / 2, 3)

    # define properties of the input signal
    source_strength = 1e6          # [Pa]
    tone_burst_freq = 0.5e6        # [Hz]
    tone_burst_cycles = 5

    # create the input signal using tone_burst
    source.p = source_strength * tone_burst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles)

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
    transducer.position = np.round([1, Ny/2 - transducer_width/2, Nz/2 - transducer.element_length/2])

    # properties used to derive the beamforming delays
    not_transducer = dotdict()
    not_transducer.sound_speed = 1540                  # sound speed [m/s]
    not_transducer.focus_distance = 25e-3              # focus distance [m]
    not_transducer.elevation_focus_distance = 19e-3    # focus distance in the elevation plane [m]
    not_transducer.steering_angle = 0                  # steering angle [degrees]

    # apodization
    not_transducer.transmit_apodization = 'Rectangular'
    not_transducer.receive_apodization = 'Rectangular'

    # define the transducer elements that are currently active
    not_transducer.active_elements = np.zeros((transducer.number_elements, 1))
    not_transducer.active_elements[20:52] = 1

    # create the transducer using the defined settings
    transducer = kWaveTransducerSimple(kgrid, **transducer)
    not_transducer = NotATransducer(transducer, kgrid, **not_transducer)

    # print out transducer properties
    # transducer.properties

    # set the input settings
    input_filename = f'example_tran_as_sen_input.h5'
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    # run the simulation
    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=np.array([PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE]),
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
        source=source,
        sensor=not_transducer,
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

    assert compare_against_ref('out_us_transducer_as_sensor', input_file_full_path), 'Files do not match!'

    # extract a single scan line from the sensor data using the current
    # beamforming settings
    # scan_line = transducer.scan_line(sensor_data)
