import os
from operator import itemgetter
from tempfile import gettempdir
from warnings import warn

import numpy as np
from numpy.fft import ifftshift
from scipy.io import savemat

from kwave.kgrid import *
from kwave.ktransducer import *
from kwave.utils import dotdict
import math
import functools

from kwave.utils import is_unix


def kspaceFirstOrderG(func):
    """
        Decorator for the kspaceFO-GPU functions

    Args:
        func: kspaceFirstOrderNDG function where 1 <= N <= 3

    Returns:
        Function wrapper
    """
    @functools.wraps(func)
    def wrapper(**kwargs):
        # Check for the binary name input. If not defined, set the default name of the GPU binary
        if 'BinaryName' not in kwargs.keys():
            kwargs['BinaryName'] = 'kspaceFirstOrder-CUDA' if is_unix() else 'kspaceFirstOrder-CUDA.exe'
        return func(**kwargs)
    return wrapper


def kspaceFirstOrderC():
    """
        Decorator for the kspaceFO-CPU functions

    Args:
        func: kspaceFirstOrderNDC function where 1 <= N <= 3

    Returns:
        Function wrapper
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            # set empty options string
            options_string = ''

            # set OS string for setting environment variables
            if is_unix():
                env_set_str = ''
                sys_sep_str = ' '
            else:
                env_set_str = 'set '
                sys_sep_str = ' & '

            # set system string to define domain for thread migration
            system_string = env_set_str + 'OMP_PLACES=cores' + sys_sep_str

            args = dotdict()

            # check for user input on axisymmetric code
            if 'Axisymmetric' in kwargs.keys():
                args.axisymmetric = kwargs['Axisymmetric']
                # check option is true or false
                assert isinstance(args.axisymmetric, bool), "Axisymmetric argument must be bool"
                del kwargs['Axisymmetric']
            else:
                # set axisymmetric to false
                args.axisymmetric = False

            # check for a user defined location for the binary
            if 'BinaryPath' in kwargs.keys():
                binary_path = kwargs['BinaryPath']

                # check for a trailing slash
                if not binary_path.endswith(os.path.sep):
                    binary_path = binary_path + os.path.sep
                del kwargs['BinaryPath']
            else:
                # set default path from environment variable
                binary_path = os.getenv('KWAVE_BINARY_PATH')

            # check for a user defined name for the binary
            if 'BinaryName' in kwargs.keys():
                binary_name = kwargs['BinaryName']
                del kwargs['BinaryName']
            else:
                # set default name for the binary
                binary_name = 'kspaceFirstOrder-OMP' if is_unix() else 'kspaceFirstOrder-OMP.exe'

            # check the binary exists and is in the correct place before doing anything else
            if not os.path.exists(f'{binary_path}{binary_name}'):
                warn(f'''
                    The binary file {binary_name} could not be found in {binary_path}. 
                    To use the C++ code, the C++ binaries for your operating system must be downloaded 
                    from www.k-wave.org/download.php and placed in the binaries folder.'''
                     )

            # check for a user defined name for the MATLAB function to call
            if 'FunctionName' in kwargs.keys():
                kwave_function_name = kwargs['FunctionName']
                del kwargs['FunctionName']
            else:
                # set default name for the k-Wave MATLAB function to call
                kwave_function_name = 'kspaceFirstOrder3D'

            # check for a user defined location for the input and output files
            if 'DataPath' in kwargs.keys():
                data_path = kwargs['DataPath']

                # check for a trailing slash
                if not data_path.endswith(os.path.sep):
                    data_path = data_path + os.path.sep
                del kwargs['DataPath']
            else:
                # set default path
                data_path = gettempdir()

            # check for a user defined name for the input and output files
            if 'DataName' in kwargs.keys():
                name_prefix = kwargs['DataName']
                input_filename = f'{name_prefix}_input.h5'
                output_filename = f'{name_prefix}_output.h5'
                del kwargs['DataName']
            else:
                # set the filename inputs to store data in the default temp directory
                date_string = get_date_string()
                input_filename = 'kwave_input_data' + date_string + '.h5'
                output_filename = 'kwave_output_data' + date_string + '.h5'

            # add pathname to input and output filenames
            input_filename = os.path.join(data_path, input_filename)
            output_filename = os.path.join(data_path, output_filename)

            # check for delete data input
            if 'DeleteData' in kwargs.keys():
                delete_data = kwargs['DeleteData']
                assert isinstance(delete_data, bool), 'DeleteData argument must be bool'
                del kwargs['DeleteData']
            else:
                # set data to be deleted
                delete_data = True

            # check for GPU device flag
            if 'DeviceNum' in kwargs.keys():
                # force to be positive integer or zero
                device_num = int(abs(kwargs['DeviceNum']))
                # add the value of the parameter to the input options
                options_string = options_string + ' -g ' + str(device_num)
                del kwargs['DeviceNum']

            # check for user defined number of threads
            if 'NumThreads' in kwargs.keys():
                num_threads = kwargs['NumThreads']

                if num_threads != 'all':
                    # check value
                    isinstance(num_threads, int) and num_threads > 0 and num_threads != float('inf')
                    # add the value of the parameter to the input options
                    options_string = options_string + ' -t ' + str(num_threads)
                del kwargs['NumThreads']

            # check for user defined thread binding option
            if 'ThreadBinding' in kwargs.keys():
                thread_binding = kwargs['ThreadBinding']
                # check value
                assert isinstance(thread_binding, int) and 0 <= thread_binding <= 1
                # read the parameters and update the system options
                if thread_binding == 0:
                    system_string = system_string + ' ' + env_set_str + 'OMP_PROC_BIND=SPREAD' + sys_sep_str
                elif thread_binding == 1:
                    system_string = system_string + ' ' + env_set_str + 'OMP_PROC_BIND=CLOSE' + sys_sep_str

                del kwargs['ThreadBinding']
            else:
                # set to round robin over places
                system_string = system_string + ' ' + env_set_str + 'OMP_PROC_BIND=SPREAD' + sys_sep_str

            # check for user input for system string
            if 'SystemCall' in kwargs.keys():
                # read the value of the parameter and add to the system options
                system_string = system_string + ' ' + kwargs['SystemCall'] + sys_sep_str
                del kwargs['SystemCall']

            # check for user defined number of threads
            if 'VerboseLevel' in kwargs.keys():
                verbose_level = kwargs['VerboseLevel']
                # check value
                assert isinstance(verbose_level, int) and 0 <= verbose_level <= 2
                # add the value of the parameter to the input options
                options_string = options_string + ' --verbose ' + str(verbose_level)
                del kwargs['VerboseLevel']

            # assign pseudonyms for input structures
            kgrid, source, sensor, medium = itemgetter('kgrid', 'source', 'sensor', 'medium')(kwargs)
            del kwargs['kgrid']
            del kwargs['source']
            del kwargs['sensor']
            del kwargs['medium']

            # check if the sensor mask is defined as cuboid corners
            if sensor.mask is not None  and sensor.mask.shape[0] == (2 * kgrid.dim):
                args.cuboid_corners = True
            else:
                args.cuboid_corners = False

            # check if performing time reversal, and replace inputs to explicitly use a
            # source with a dirichlet boundary condition
            if sensor.time_reversal_boundary_data is not None:
                # define a new source structure
                source = {
                    'p_mask': sensor.p_mask,
                    'p': np.flip(sensor.time_reversal_boundary_data, 2),
                    'p_mode': 'dirichlet'
                }

                # define a new sensor structure
                Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
                sensor = kSensor(
                    mask=np.ones((Nx, Ny, max(1, Nz))),
                    record=['p_final']
                )
                # set time reversal flag
                args.time_rev = True
            else:
                # set time reversal flag
                args.time_rev = False

            # check if sensor.record is given
            if sensor.record is not None:
                record = sensor.record

                # set the options string to record the required output fields
                record_options_map = {
                    'p':            'p_raw',
                    'p_max':        'p_max',
                    'p_min':        'p_min',
                    'p_rms':        'p_rms',
                    'p_max_all':    'p_max_all',
                    'p_min_all':    'p_min_all',
                    'p_final':      'p_final',
                    'u':            'u_raw',
                    'u_max':        'u_max',
                    'u_min':        'u_min',
                    'u_rms':        'u_rms',
                    'u_max_all':    'u_max_all',
                    'u_min_all':    'u_min_all',
                    'u_final':      'u_final'
                }
                for k, v in record_options_map.items():
                    if k in record:
                        options_string = options_string + f' --{v}'

                if 'u_non_staggered' in record or 'I_avg' in record or 'I' in record:
                    options_string = options_string + ' --u_non_staggered_raw'

                if ('I_avg' in record or 'I' in record) and ('p' not in record):
                    options_string = options_string + ' --p_raw'
            else:
                # if sensor.record is not given, record the raw time series of p
                options_string = options_string + ' --p_raw'

            # check if sensor.record_start_imdex is given
            if sensor.record_start_index is not None:
                options_string = options_string + ' -s ' + str(sensor.record_start_index)

            # append the save to disk parameter
            # farid | modified behaviour here!
            # Originally, kspaceFO-nD GPU and CPU version would ALWAYS add 'SaveToDisk' option to kwargs
            # And kspaceFO-nD would not add this option
            # In our case all examples use CPU version. So some usages already pass 'SaveToDisk' as kwarg
            # For them, we don't override 'SaveToDisk' options
            # (originally it would have and it was required to be changed using DataPath & DataName options)
            if 'SaveToDisk' not in kwargs:
                kwargs['SaveToDisk'] = input_filename
                kwargs['SaveToDiskExit'] = True
            res = func(kgrid=kgrid, medium=medium, source=source, sensor=sensor, **args, **kwargs)
            return res
        return wrapper
    return decorator
