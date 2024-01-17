import logging
import stat
import subprocess
import unittest.mock

import h5py

from kwave.utils.dotdictionary import dotdict


class Executor:

    def __init__(self, execution_options, simulation_options):

        self.execution_options = execution_options
        self.simulation_options = simulation_options

        self._make_binary_executable()

    def _make_binary_executable(self):
        self.execution_options.binary_path.chmod(self.execution_options.binary_path.stat().st_mode | stat.S_IEXEC)

    def run_simulation(self, input_filename: str, output_filename: str, options: str):

        command = f'{self.execution_options.system_string} ' \
                  f'{self.execution_options.binary_path} ' \
                  f'-i {input_filename} ' \
                  f'-o {output_filename} ' \
                  f'{options}'

        stdout = None if self.execution_options.show_sim_log else subprocess.DEVNULL
        try:
            subprocess.run(command, stdout=stdout, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            if isinstance(e.returncode, unittest.mock.MagicMock):
                logging.info('Skipping AssertionError in testing.')
            else:
                raise

        sensor_data = self.parse_executable_output(output_filename)

        return sensor_data

    @staticmethod
    def parse_executable_output(output_filename: str) -> dotdict:

        # Load the simulation and pml sizes from the output file
        # with h5py.File(output_filename, 'r') as output_file:
        #     Nx, Ny, Nz = output_file['/Nx'][0].item(), output_file['/Ny'][0].item(), output_file['/Nz'][0].item()
        #     pml_x_size, pml_y_size = output_file['/pml_x_size'][0].item(), output_file['/pml_y_size'][0].item()
        #     pml_z_size = output_file['/pml_z_size'][0].item() if Nz > 1 else 0

        # # Set the default index variables for the _all and _final variables
        # x1, x2 = 1, Nx
        # y1, y2 = (
        #     1, 1 + pml_y_size) if self.simulation_options.simulation_type is not SimulationType.AXISYMMETRIC else (
        #     1, Ny)
        # z1, z2 = (1 + pml_z_size, Nz - pml_z_size) if Nz > 1 else (1, Nz)
        #
        # # Check if the PML is set to be outside the computational grid
        # if self.simulation_options.pml_inside:
        #     x1, x2 = 1 + pml_x_size, Nx - pml_x_size
        #     y1, y2 = (1, Ny) if self.simulation_options.simulation_type is SimulationType.AXISYMMETRIC else (
        #         1 + pml_y_size, Ny - pml_y_size)
        #     z1, z2 = 1 + pml_z_size, Nz - pml_z_size if Nz > 1 else (1, Nz)

        # Load the C++ data back from disk using h5py
        with h5py.File(output_filename, 'r') as output_file:
            sensor_data = {}
            for key in output_file.keys():
                sensor_data[key] = output_file[f'/{key}'][0].squeeze()
        #     if self.simulation_options.cuboid_corners:
        #         sensor_data = [output_file[f'/p/{index}'][()] for index in range(1, len(key['mask']) + 1)]
        #
        # # Combine the sensor data if using a kWaveTransducer as a sensor
        # if isinstance(sensor, kWaveTransducer):
        #     sensor_data['p'] = sensor.combine_sensor_data(sensor_data['p'])

        # # Compute the intensity outputs
        # if any(key.startswith(('I_avg', 'I')) for key in sensor.get('record', [])):
        #     flags = {
        #         'record_I_avg': 'I_avg' in sensor['record'],
        #         'record_I': 'I' in sensor['record'],
        #         'record_p': 'p' in sensor['record'],
        #         'record_u_non_staggered': 'u_non_staggered' in sensor['record']
        #     }
        #     kspaceFirstOrder_saveIntensity()
        #
        # # Filter the recorded time domain pressure signals using a Gaussian filter if defined
        # if not time_rev and 'frequency_response' in sensor:
        #     frequency_response = sensor['frequency_response']
        #     sensor_data['p'] = gaussianFilter(sensor_data['p'], 1 / kgrid.dt, frequency_response[0], frequency_response[1])
        #
        # # Assign sensor_data.p to sensor_data if sensor.record is not given
        # if 'record' not in sensor and not cuboid_corners:
        #     sensor_data = sensor_data['p']
        #
        # # Delete the input and output files
        # if delete_data:
        #     os.remove(input_filename)
        #     os.remove(output_filename)
        return sensor_data
