import io
import os
import stat
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from kwave.data import SimulationResult
from kwave.executor import Executor
from kwave.utils.dotdictionary import dotdict


class TestExecutor(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.execution_options = Mock()
        self.simulation_options = Mock()
        self.execution_options.binary_path = Path("/fake/path/to/binary")
        self.execution_options.system_string = "fake_system"
        self.execution_options.show_sim_log = False
        self.simulation_options.pml_inside = True

        # Mock for stat result
        self.mock_stat_result = Mock()
        self.mock_stat_result.st_mode = 0o100644

        # Patchers
        self.patcher_stat = patch("pathlib.Path.stat", return_value=self.mock_stat_result)
        self.patcher_chmod = patch("pathlib.Path.chmod")
        self.patcher_popen = patch("subprocess.Popen")
        self.patcher_h5py = patch("h5py.File", autospec=True)

        # Start patchers
        self.mock_stat = self.patcher_stat.start()
        self.mock_chmod = self.patcher_chmod.start()
        self.mock_popen = self.patcher_popen.start()
        self.mock_h5py_file = self.patcher_h5py.start()

        # Mock Popen object
        self.mock_proc = MagicMock()
        self.mock_proc.communicate.return_value = ("stdout content", "stderr content")
        self.mock_popen.return_value.__enter__.return_value = self.mock_proc

        # Mock return dictionary
        N = np.array([20, 20, 20])
        pml = np.array([2, 2, 2])
        two_d_output = MagicMock()
        two_d_output.ndim = 2
        three_d_output = MagicMock()
        three_d_output.ndim = 3
        self.mock_dict_values = {
            "Nx": N[0],
            "Ny": N[1],
            "Nz": N[2],
            "pml_x_size": pml[0],
            "pml_y_size": pml[1],
            "pml_z_size": pml[2],
            "axisymmetric_flag": np.array(False),
            "p_final": two_d_output,
            "p_max_all": three_d_output,
        }
        # Use a real dictionary instead of a mock
        self.mock_dict = self.mock_dict_values.copy()

    def tearDown(self):
        # Stop patchers
        self.patcher_stat.stop()
        self.patcher_chmod.stop()
        self.patcher_popen.stop()
        self.patcher_h5py.stop()

    def test_make_binary_executable(self):
        """Test that the binary executable is correctly set to executable mode."""
        # Instantiate the Executor
        _ = Executor(self.execution_options, self.simulation_options)

        # Assert chmod was called with the correct parameters
        self.mock_chmod.assert_called_once_with(self.mock_stat_result.st_mode | stat.S_IEXEC)

    @patch("builtins.print")
    def test_run_simulation_show_sim_log(self, mock_print):
        """Test handling real-time output when show_sim_log is True."""
        self.execution_options.show_sim_log = True

        # Create a generator to simulate real-time output
        def mock_stdout_gen():
            yield "line 1\n"
            yield "line 2\n"
            yield "line 3\n"

        # Create a mock Popen object with stdout as a generator
        self.mock_proc.stdout = mock_stdout_gen()
        self.mock_proc.returncode = 0

        # Instantiate the Executor
        executor = Executor(self.execution_options, self.simulation_options)

        # Mock the parse_executable_output method
        with patch.object(executor, "parse_executable_output", return_value=dotdict()):
            sensor_data = executor.run_simulation("input.h5", "output.h5", ["options"])

        # Assert that the print function was called with the expected lines
        expected_calls = [call("line 1\n", end=""), call("line 2\n", end=""), call("line 3\n", end="")]
        mock_print.assert_has_calls(expected_calls, any_order=False)

        # Check that sensor_data is returned correctly
        self.assertIsInstance(sensor_data, SimulationResult)

    def test_run_simulation_success(self):
        """Test running the simulation successfully."""
        self.mock_proc.returncode = 0

        # Instantiate the Executor
        executor = Executor(self.execution_options, self.simulation_options)

        # Mock the parse_executable_output method
        with patch.object(executor, "parse_executable_output", return_value=dotdict()):
            sensor_data = executor.run_simulation("input.h5", "output.h5", ["options"])

        normalized_path = os.path.normpath(self.execution_options.binary_path)
        expected_command = [normalized_path, "-i", "input.h5", "-o", "output.h5", "options"]

        self.mock_popen.assert_called_once_with(
            expected_command, env=self.execution_options.env_vars, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.mock_proc.communicate.assert_called_once()
        self.assertIsInstance(sensor_data, SimulationResult)

    def test_run_simulation_failure(self):
        """Test handling a simulation failure."""
        self.mock_proc.returncode = 1

        # Capture the printed output
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout, patch("sys.stderr", new=io.StringIO()) as mock_stderr:
            # Instantiate the Executor
            executor = Executor(self.execution_options, self.simulation_options)

            # Mock the parse_executable_output method
            with patch.object(executor, "parse_executable_output", return_value=dotdict()):
                with self.assertRaises(subprocess.CalledProcessError):
                    executor.run_simulation("input.h5", "output.h5", ["options"])

            # Get the printed output
            stdout_output = mock_stdout.getvalue()
            stderr_output = mock_stderr.getvalue()

            # Assert that stdout and stderr are printed
            self.assertIn("stdout content", stdout_output)
            self.assertIn("stderr content", stderr_output)

    def test_parse_executable_output(self):
        """Test parsing the executable output correctly."""
        # Set up the mock for h5py.File
        mock_file = self.mock_h5py_file.return_value.__enter__.return_value
        mock_file.keys.return_value = ["data"]
        mock_dataset = MagicMock()
        mock_file.__getitem__.return_value = mock_dataset

        # Mock the squeeze method on the mock dataset
        mock_dataset.__getitem__.return_value.squeeze.return_value = np.ones((10, 10))

        # Call the method with a fake file path
        result = Executor.parse_executable_output("/fake/output.h5")

        self.mock_h5py_file.assert_called_once_with("/fake/output.h5", "r")
        mock_file.keys.assert_called_once()
        mock_file.__getitem__.assert_called_once_with("/data")
        mock_dataset.__getitem__.assert_called_once_with(slice(None))
        mock_dataset.__getitem__.return_value.squeeze.assert_called_once()
        self.assertIn("data", result)
        self.assertTrue(np.all(result["data"] == np.ones((10, 10))))

    def test_sensor_data_cropping_with_pml_outside(self):
        """If pml is outside, fields like p_final and p_max_all should be cropped."""
        self.mock_proc.returncode = 0
        self.simulation_options.pml_inside = False

        # Create a mock that tracks calls before being replaced
        original_two_d_output = MagicMock()
        original_two_d_output.ndim = 2
        original_three_d_output = MagicMock()
        original_three_d_output.ndim = 3

        # Update the mock dict to use these tracking mocks
        self.mock_dict["p_final"] = original_two_d_output
        self.mock_dict["p_max_all"] = original_three_d_output

        # Instantiate the Executor
        executor = Executor(self.execution_options, self.simulation_options)

        # Mock the parse_executable_output method
        with patch.object(executor, "parse_executable_output", return_value=self.mock_dict):
            sensor_data = executor.run_simulation("input.h5", "output.h5", ["options"])

        # Verify that sensor_data is a SimulationResult
        self.assertIsInstance(sensor_data, SimulationResult)

        # Verify that the original mock objects were called for cropping
        original_two_d_output.__getitem__.assert_called_once_with((slice(2, 18), slice(2, 18)))
        original_three_d_output.__getitem__.assert_called_once_with((slice(2, 18), slice(2, 18), slice(2, 18)))

        # check that the other fields are unchanged
        for field in self.mock_dict.keys():
            if field not in ["p_final", "p_max_all"]:
                self.assertEqual(sensor_data[field], self.mock_dict[field])

    def test_sensor_data_cropping_with_pml_inside(self):
        """If pml is inside, no field should be cropped."""
        self.mock_proc.returncode = 0

        # Create a mock that tracks calls before being replaced
        original_two_d_output = MagicMock()
        original_two_d_output.ndim = 2
        original_three_d_output = MagicMock()
        original_three_d_output.ndim = 3

        # Update the mock dict to use these tracking mocks
        self.mock_dict["p_final"] = original_two_d_output
        self.mock_dict["p_max_all"] = original_three_d_output

        # Instantiate the Executor
        executor = Executor(self.execution_options, self.simulation_options)

        # Mock the parse_executable_output method
        with patch.object(executor, "parse_executable_output", return_value=self.mock_dict):
            sensor_data = executor.run_simulation("input.h5", "output.h5", ["options"])

        # Verify that sensor_data is a SimulationResult
        self.assertIsInstance(sensor_data, SimulationResult)

        # because pml is inside, the output should not be cropped
        # The mock objects should not have been called for cropping
        original_two_d_output.__getitem__.assert_not_called()
        original_three_d_output.__getitem__.assert_not_called()

        # check that the other fields are unchanged
        for field in self.mock_dict.keys():
            if field not in ["p_final", "p_max_all"]:
                self.assertEqual(sensor_data[field], self.mock_dict[field])

    def test_executor_file_not_found_on_non_darwin(self):
        # Configure the mock path object
        mock_binary_path = MagicMock(spec=Path)
        mock_binary_path.chmod.side_effect = FileNotFoundError

        # Mock the execution options to use the mocked path
        mock_execution_options = MagicMock()
        mock_execution_options.binary_path = mock_binary_path
        mock_execution_options.is_gpu_simulation = False

        with patch("kwave.PLATFORM", "windows"):
            with pytest.raises(FileNotFoundError):
                _ = Executor(execution_options=mock_execution_options, simulation_options=MagicMock())

    def test_cpu_environment_variable(self):
        """Test that environment variable KWAVE_FORCE_CPU sets CPU simulation."""
        # Set the environment variable
        os.environ["KWAVE_FORCE_CPU"] = "1"

        # Create mock execution options with default GPU simulation
        mock_execution_options = MagicMock()
        mock_execution_options.is_gpu_simulation = True
        mock_execution_options.binary_name = "kspaceFirstOrder-CUDA"
        mock_execution_options.binary_path = MagicMock()

        # Create the Executor instance
        executor = Executor(execution_options=mock_execution_options, simulation_options=MagicMock())

        # Assert that the environment variable has changed the simulation to CPU
        self.assertFalse(executor.execution_options.is_gpu_simulation)
        self.assertEqual(executor.execution_options.binary_name, "kspaceFirstOrder-OMP")

        # Cleanup environment variable
        del os.environ["KWAVE_FORCE_CPU"]

    def test_executor_gpu_cuda_failure_darwin(self):
        expected_error_msg = (
            "GPU simulations are currently not supported on MacOS. Try running the simulation on CPU by setting is_gpu_simulation=False."
        )
        # Configure the mock path object
        mock_binary_path = MagicMock(spec=Path)
        mock_binary_path.chmod.side_effect = FileNotFoundError
        mock_binary_path.exists.return_value = False

        # Mock the execution options to use the mocked path
        mock_execution_options = MagicMock()
        mock_execution_options.binary_path = mock_binary_path
        mock_execution_options.is_gpu_simulation = True

        with patch("kwave.PLATFORM", "darwin"):
            with pytest.raises(ValueError, match=expected_error_msg):
                _ = Executor(execution_options=mock_execution_options, simulation_options=MagicMock())


if __name__ == "__main__":
    pytest.main([__file__])
