import unittest
from unittest.mock import Mock
from kwave.ksensor import kSensor
from kwave import PLATFORM
from kwave.options.simulation_execution_options import SimulationExecutionOptions
import os

from kwave.utils.checks import is_unix

OMP_BINARY_NAME = "kspaceFirstOrder-OMP{}".format(".exe" if PLATFORM == "windows" else "")
CUDA_BINARY_NAME = "kspaceFirstOrder-CUDA{}".format(".exe" if PLATFORM == "windows" else "")


class TestSimulationExecutionOptions(unittest.TestCase):
    def setUp(self):
        self.default_options = SimulationExecutionOptions()

    def test_default_initialization(self):
        """Test default values during initialization."""
        options = self.default_options
        self.assertFalse(options.is_gpu_simulation)
        self.assertEqual(options._binary_path, None)
        self.assertIsNone(options._binary_name)
        self.assertEqual(options.kwave_function_name, "kspaceFirstOrder3D")
        self.assertTrue(options.delete_data)
        self.assertIsNone(options.device_num)
        self.assertEqual(options.num_threads, os.cpu_count())  # "all" should default to CPU count
        self.assertIsNone(options.thread_binding)
        self.assertIsNone(options.system_call)
        self.assertEqual(options.verbose_level, 0)
        self.assertTrue(options.auto_chunking)
        self.assertTrue(options.show_sim_log)

    def test_num_threads_setter_valid(self):
        """Test setting a valid number of threads."""
        options = self.default_options
        options.num_threads = os.cpu_count()
        self.assertEqual(options.num_threads, os.cpu_count())

        options.num_threads = "all"
        self.assertEqual(options.num_threads, os.cpu_count())

    def test_num_threads_setter_invalid(self):
        """Test setting an invalid number of threads."""
        options = self.default_options
        with self.assertRaises(ValueError):
            options.num_threads = -1
        with self.assertRaises(ValueError):
            options.num_threads = "invalid_value"

    def test_verbose_level_setter_valid(self):
        """Test setting a valid verbose level."""
        options = self.default_options
        for level in range(3):
            options.verbose_level = level
            self.assertEqual(options.verbose_level, level)

    def test_verbose_level_setter_invalid(self):
        """Test setting an invalid verbose level."""
        options = self.default_options
        with self.assertRaises(ValueError):
            options.verbose_level = 3
        with self.assertRaises(ValueError):
            options.verbose_level = -1

    def test_is_gpu_simulation_setter(self):
        """Test setting is_gpu_simulation and its impact on binary_name."""
        options = self.default_options
        options.is_gpu_simulation = True
        self.assertTrue(options.is_gpu_simulation)
        self.assertEqual(options.binary_name, CUDA_BINARY_NAME)

        options.is_gpu_simulation = False
        self.assertFalse(options.is_gpu_simulation)
        self.assertEqual(options.binary_name, OMP_BINARY_NAME)

    def test_binary_name_custom(self):
        """Test setting a custom binary name."""
        options = self.default_options
        options.binary_name = "custom_binary"
        self.assertEqual(options.binary_name, "custom_binary")

    def test_binary_path_property(self):
        """Test the binary_path property for correct formatting based on PLATFORM."""
        options = self.default_options
        if PLATFORM == "windows":
            self.assertTrue(str(options.binary_path).endswith(".exe"))
        else:
            self.assertFalse(str(options.binary_path).endswith(".exe"))

    def test_get_options_string(self):
        """Test the get_options_string method with a mock sensor."""
        mock_sensor = Mock(spec=kSensor)
        mock_sensor.record = ["p", "u_max"]
        mock_sensor.record_start_index = 10

        options = self.default_options
        options.device_num = 1
        options.num_threads = os.cpu_count()
        options.verbose_level = 2

        options_string = options.get_options_string(mock_sensor)
        expected_substrings = [" -g 1", f" -t {os.cpu_count}", " --verbose 2", " --p_raw", " --u_max", " -s 10"]
        for substring in expected_substrings:
            self.assertIn(substring, options_string)

    def test_construct_system_string(self):
        """Test the _construct_system_string method for proper environment variable formatting."""
        options = self.default_options
        options.thread_binding = True

        system_string = options.system_string
        expected_omp_bind = "SPREAD" if options.thread_binding else "CLOSE"
        self.assertIn(f"OMP_PROC_BIND={expected_omp_bind}", system_string)

    def test_system_string_property(self):
        """Test the system_string property for appropriate construction."""
        options = self.default_options
        system_string = options.system_string
        if is_unix():
            self.assertIn("OMP_PLACES=cores", system_string)
        else:
            self.assertIn("set OMP_PLACES=cores", system_string)

    def test_gpu_dependency_on_binary_name_and_path(self):
        """Test that the binary_name and binary_path are updated correctly based on is_gpu_simulation."""
        options = SimulationExecutionOptions(is_gpu_simulation=True)
        self.assertEqual(options.binary_name, CUDA_BINARY_NAME)

        options.is_gpu_simulation = False
        self.assertEqual(options.binary_name, OMP_BINARY_NAME)
        self.assertTrue(str(options.binary_path).endswith(OMP_BINARY_NAME))


if __name__ == "__main__":
    unittest.main()
