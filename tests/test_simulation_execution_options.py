import unittest
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.ksensor import kSensor
from unittest.mock import patch


class TestSimulationExecutionOptions(unittest.TestCase):
    def setUp(self):
        # Set up a default kSensor object for testing
        self.sensor = kSensor()
        self.sensor.record = ["p", "u"]
        self.sensor.record_start_index = 10

    def test_default_initialization(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "linux"):
            options = SimulationExecutionOptions()
            self.assertFalse(options.is_gpu_simulation)
            self.assertEqual(options.binary_name, "kspaceFirstOrder-OMP")
            self.assertTrue(options.delete_data)
            self.assertEqual(options.num_threads, None)  # TODO: confusing logic here
            self.assertEqual(options.verbose_level, 0)

    def test_gpu_simulation_initialization(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "linux"):
            options = SimulationExecutionOptions(is_gpu_simulation=True)
            self.assertTrue(options.is_gpu_simulation)
            self.assertEqual(options.binary_name, "kspaceFirstOrder-CUDA")

    def test_binary_name_extension_on_windows(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "windows"):
            options = SimulationExecutionOptions()
            self.assertTrue(options.binary_name.endswith(".exe"))

    def test_get_options_string(self):
        options = SimulationExecutionOptions()
        options_string = options.get_options_string(self.sensor)
        self.assertIn("--p_raw", options_string)
        self.assertIn("--u_raw", options_string)
        self.assertIn("-s 10", options_string)

    def test_env_vars_linux(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "linux"):
            options = SimulationExecutionOptions()
            env_vars = options.env_vars
            self.assertIn("OMP_PLACES", env_vars)
            self.assertEqual(env_vars["OMP_PLACES"], "cores")
            self.assertIn("OMP_PROC_BIND", env_vars)
            self.assertEqual(env_vars["OMP_PROC_BIND"], "SPREAD")

    def test_thread_binding_linux(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "linux"):
            options = SimulationExecutionOptions(thread_binding=True)
            env_vars = options.env_vars
            self.assertEqual(env_vars["OMP_PROC_BIND"], "SPREAD")

    def test_thread_binding_darwin(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "darwin"):
            options = SimulationExecutionOptions(thread_binding=True)
            with self.assertRaises(ValueError, msg="Thread binding is not supported in MacOS."):
                _ = options.env_vars

    def test_env_vars_darwin(self):
        with patch("kwave.options.simulation_execution_options.PLATFORM", "darwin"):
            options = SimulationExecutionOptions()
            env_vars = options.env_vars
            self.assertNotIn("OMP_PLACES", env_vars)
            self.assertNotIn("OMP_PROC_BIND", env_vars)


if __name__ == "__main__":
    unittest.main()
