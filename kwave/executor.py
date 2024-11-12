import os
import stat
import subprocess
import sys

import h5py

import kwave
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.dotdictionary import dotdict


class Executor:
    def __init__(self, execution_options: SimulationExecutionOptions, simulation_options):
        self.execution_options = execution_options
        self.simulation_options = simulation_options

        if os.environ.get("KWAVE_FORCE_CPU") == "1":
            self.execution_options.is_gpu_simulation = False
            self.execution_options.binary_name = "kspaceFirstOrder-OMP"
            self.execution_options.binary_path = kwave.BINARY_PATH / self.execution_options.binary_name
        self._make_binary_executable()

    def _make_binary_executable(self):
        binary_path = self.execution_options.binary_path
        if not binary_path.exists():
            if kwave.PLATFORM == "darwin" and self.execution_options.is_gpu_simulation:
                raise ValueError(
                    "GPU simulations are currently not supported on MacOS. "
                    "Try running the simulation on CPU by setting is_gpu_simulation=False."
                )
            raise FileNotFoundError(f"Binary not found at {binary_path}")
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)

    def run_simulation(self, input_filename: str, output_filename: str, options: str):
        command = [str(self.execution_options.binary_path), "-i", input_filename, "-o", output_filename, options]

        try:
            with subprocess.Popen(
                command, env=self.execution_options.env_vars, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ) as proc:
                stdout, stderr = "", ""
                if self.execution_options.show_sim_log:
                    # Stream stdout in real-time
                    for line in proc.stdout:
                        print(line, end="")

                stdout, stderr = proc.communicate()

                proc.wait()  # wait for process to finish before checking return code
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, command, stdout, stderr)

        except subprocess.CalledProcessError as e:
            # This ensures stdout is printed regardless of show_sim_logs value if an error occurs
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
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
        with h5py.File(output_filename, "r") as output_file:
            sensor_data = {}
            for key in output_file.keys():
                sensor_data[key] = output_file[f"/{key}"][:].squeeze()
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
