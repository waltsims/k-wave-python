import stat
import sys
from pathlib import Path
import os
import h5py
import numpy as np


class Executor:

    def __init__(self, use_gpu_if_possible):
        self._is_linux = sys.platform.startswith('linux')

        if not self._is_linux:
            raise NotImplementedError('Running on Windows is not implemented yet.')

        binary_folder = 'linux' if self._is_linux else 'windows'
        binary_name = 'kspaceFirstOrder'
        if use_gpu_if_possible:
            binary_name += '-CUDA'
        else:
            binary_name += '-OMP'

        path_of_this_file = Path(__file__).parent.resolve()
        self.binary_path = path_of_this_file / 'bin' / binary_folder / binary_name

        self._make_binary_executable()

    def _make_binary_executable(self):
        self.binary_path.chmod(self.binary_path.stat().st_mode | stat.S_IEXEC)

    def run_simulation(self, input_filename: str, output_filename: str, options: str):
        env_variables = 'export LD_LIBRARY_PATH=;' \
                        'OMP_PLACES=cores;' \
                        'OMP_PROC_BIND=SPREAD;'

        command = f'{env_variables} {self.binary_path} ' \
                  f'-i {input_filename} -o {output_filename} {options}'

        return_code = os.system(command)
        assert return_code == 0

        with h5py.File(output_filename, 'r') as hf:
            sensor_data = np.array(hf['p'])[0].T

        return sensor_data
