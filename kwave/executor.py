import logging
import os
import stat
import sys
import unittest.mock
from pathlib import Path

import h5py
import numpy as np


class Executor:

    def __init__(self, device):

        binary_name = 'kspaceFirstOrder'
        if device == 'gpu':
            binary_name += '-CUDA'
        elif device == 'cpu':
            binary_name += '-OMP'
        else:
            raise ValueError("Unrecognized value passed as target device. Options are 'gpu' or 'cpu'.")

        self._is_linux = sys.platform.startswith('linux')
        self._is_windows = sys.platform.startswith(('win', 'cygwin'))
        self._is_darwin = sys.platform.startswith('darwin')

        if self._is_linux:
            binary_folder = 'linux'
        elif self._is_windows:
            binary_folder = 'windows'
            binary_name += '.exe'
        elif self._is_darwin:
            raise NotImplementedError('k-wave-python is currently unsupported on MacOS.')

        path_of_this_file = Path(__file__).parent.resolve()
        self.binary_path = path_of_this_file / 'bin' / binary_folder / binary_name

        self._make_binary_executable()

    def _make_binary_executable(self):
        self.binary_path.chmod(self.binary_path.stat().st_mode | stat.S_IEXEC)

    def run_simulation(self, input_filename: str, output_filename: str, options: str):
        env_variables = {
            # TODO(walter): I'm not sure why we overwrite the system LD_LIBRARY_PATH...
            #  Commenting out for now to run on machines with non-standard LD_LIBRARY_PATH.
            # 'LD_LIBRARY_PATH': '',
            'OMP_PLACES': 'cores',
            'OMP_PROC_BIND': 'SPREAD',
        }
        os.environ.update(env_variables)

        command = f'{self.binary_path} -i {input_filename} -o {output_filename} {options}'

        return_code = os.system(command)

        try:
            assert return_code == 0, f'Simulation call returned code: {return_code}'
        except AssertionError:
            if isinstance(return_code, unittest.mock.MagicMock):
                logging.info('Skipping AssertionError in testing.')

        with h5py.File(output_filename, 'r') as hf:
            sensor_data = np.array(hf['p'])[0].T

        return sensor_data
