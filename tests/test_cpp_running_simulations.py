"""
    Running C++ Simulations Example

    This example demonstrates how to use the C++ versions of
    kspaceFirstOrder3D. Before use, the appropriate C++ codes must be
    downloaded from http://www.k-wave.org/download.php and placed in the
    binaries folder of the toolbox.
"""
# noinspection PyUnresolvedReferences
import setup_test
import os
import h5py
import numpy as np
from tempfile import gettempdir
from kwave.ksource import kSource
from kwave.kgrid import kWaveGrid
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC, kspaceFirstOrder3DG
from kwave.utils import *
from tests.diff_utils import compare_against_ref
from kwave.utils import dotdict
from kwave.kmedium import kWaveMedium


def test_cpp_running_simulations():
    # modify this parameter to run the different examples
    # 1: Save the input data to disk
    # 2: Reload the output data from disk
    # 3: Run the C++ simulation from MATLAB
    # 4: Run the C++ simulation on a CUDA-enabled GPU from MATLAB

    example_number = 1

    # input and output filenames (these must have the .h5 extension)
    input_filename  = 'example_input.h5'
    output_filename = 'example_output.h5'

    # pathname for the input and output files
    pathname = gettempdir()

    # remove input file if it already exists
    input_file_full_path = os.path.join(pathname, input_filename)
    output_file_full_path = os.path.join(pathname, output_filename)
    if example_number == 1 and os.path.exists(input_file_full_path):
        os.remove(input_file_full_path)

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 256                   # number of grid points in the x direction
    Ny = 128                   # number of grid points in the y direction
    Nz = 64                    # number of grid points in the z direction
    dx = 0.1e-3                # grid point spacing in the x direction [m]
    dy = 0.1e-3                # grid point spacing in the y direction [m]
    dz = 0.1e-3                # grid point spacing in the z direction [m]
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

    # set the size of the PML
    pml_size = 10              # [grid points]

    # define a scattering ball
    ball_radius = 20           # [grid points]
    ball_x      = Nx/2 + 40    # [grid points]
    ball_y      = Ny/2         # [grid points]
    ball_z      = Nz/2         # [grid points]
    ball        = makeBall(Nx, Ny, Nz, ball_x, ball_y, ball_z, ball_radius)

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500 * np.ones((Nx, Ny, Nz)),       # [m/s]
        density=1000 * np.ones((Nx, Ny, Nz)),           # [kg/m^3],
        alpha_coeff=0.75,                               # [dB/(MHz^y cm)]
        alpha_power=1.5
    )
    medium.sound_speed[ball == 1] = 1800                                # [m/s]
    medium.density[ball == 1]     = 1200                                # [kg/m^3]

    # create the time array
    Nt = 1200                  # number of time steps
    dt = 15e-9                 # time step [s]
    kgrid.setTime(Nt, dt)

    # define a square source element facing in the x-direction
    source_y_size = 60         # [grid points]
    source_z_size = 30         # [grid points]
    source = kSource()
    source.p_mask = np.zeros((Nx, Ny, Nz))
    source.p_mask[pml_size, Ny//2 - source_y_size//2 - 1:Ny//2 + source_y_size//2, Nz//2 - source_z_size//2 - 1:Nz//2 + source_z_size//2] = 1  # ???

    # define a time varying sinusoidal source
    source_freq     = 2e6      # [Hz]
    source_strength = 0.5e6    # [Pa]
    source.p        = source_strength * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filterTimeSeries(kgrid, medium, source.p)
    source.p = np.array(source.p)

    # define a sensor mask through the central plane
    sensor_mask = np.zeros((Nx, Ny, Nz))
    sensor_mask[:, :, Nz//2 - 1] = 1
    sensor = kSensor(sensor_mask)

    # set the input arguments
    input_args = {
        'PMLSize': pml_size
    }

    if example_number == 1:
        # save the input data to disk and then exit
        kspaceFirstOrder3DC(**{
            'medium': medium,
            'kgrid': kgrid,
            'source': source,
            'sensor': sensor,
            **input_args,
            'SaveToDisk': input_file_full_path
        })

        # display the required syntax to run the C++ simulation
        print(f'Using a terminal window, navigate to the {os.path.sep}binaries folder of the k-Wave Toolbox')
        print('Then, use the syntax shown below to run the simulation:')
        if os.name == 'posix':
            print(f'./kspaceFirstOrder-OMP -i {input_file_full_path} -o {output_file_full_path} --p_final --p_max')
        else:
            print(f'kspaceFirstOrder-OMP.exe -i {input_file_full_path} -o {output_file_full_path} --p_final --p_max')

        assert compare_against_ref('out_cpp_running_simulations', input_file_full_path), 'Files do not match!'

    elif example_number == 2:
        # load output data from the C++ simulation
        with h5py.File(output_file_full_path, 'r') as hf:
            sensor_data = dotdict({
                'p_final': np.array(hf['p_final']),
                'p_max': np.array(hf['p_max']),
            })

    elif example_number == 3:
        # define the field parameters to record
        sensor.record = ['p_final', 'p_max']

        # run the C++ simulation using the wrapper function
        sensor_data = kspaceFirstOrder3DC(**{
            'kgrid': kgrid,
            'medium': medium,
            'source': source,
            'sensor': sensor,
            **input_args
        })

    elif example_number == 4:
        # define the field parameters to record
        sensor.record = ['p_final', 'p_max']

        # run the C++ simulation using the wrapper function
        sensor_data = kspaceFirstOrder3DG(**{
            'kgrid': kgrid,
            'medium': medium,
            'source': source,
            'sensor': sensor,
            **input_args
        })
