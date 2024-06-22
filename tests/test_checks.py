import unittest
from unittest.mock import patch
import numpy as np
import pytest
import kwave
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions as ExecutionOptions
from kwave.utils.signals import tone_burst
from kwave.utils.matlab import rem


class TestUltrasoundSimulation(unittest.TestCase):
    def setUp(self):
        # Parameters
        self.steering_angle = -45  # Using a single angle for the test

        # Initialize
        self.source = kSource()
        self.sensor = kSensor()

        # Simulation settings
        self.DATA_CAST = "single"  # Use float32 for GPU computations

        # Create the computational grid
        self.ROIx = 120e-3  # ROI [m]
        self.ROIy = 120e-3  # ROI [m]
        self.PML_size = 100  # Size of the PML in grid points
        self.Nx = 3000  # Number of grid points in the x (row) direction
        self.Ny = 3000  # Number of grid points in the y (column) direction
        self.dx = self.ROIx / self.Nx  # Grid point spacing in the x direction [m]
        self.dy = self.ROIy / self.Ny  # Grid point spacing in the y direction [m]

        self.kgrid = kWaveGrid(Vector([self.Nx, self.Ny]), Vector([self.dx, self.dy]))
        self.f0 = 3.5e6  # Center frequency [Hz]

        # Sensor
        self.pitch = 0.25e-3  # Length between each sensor [m]
        self.N_active_tx = 80  # Number of elements of the transducer

        # Define the medium properties
        self.c0_exact = 1540  # [m/s]
        self.rho0 = 1000  # [kg/m^3]
        self.alpha_coeff_exact = 0.5  # [dB/(MHz^y cm)]
        self.medium = kWaveMedium(self.c0_exact, self.rho0, self.alpha_coeff_exact)

        # Create the time array
        t_end = (self.Nx * self.dx) * 2.5 / self.c0_exact  # [s]
        dt = 1 / 62.5e6 / 2  # Sampling time [s]
        cfl = dt * self.c0_exact / self.dx  # Default: 0.3
        self.kgrid.makeTime(self.c0_exact, cfl, t_end)

        # Define the input signal
        self.source_strength = 1e1  # [Pa]
        self.tone_burst_freq = self.f0  # [Hz]
        self.tone_burst_cycles = 5  # 5 is optimum

        # Source position
        self.el_pos_cart = np.column_stack(
            (self.pitch * (np.arange(1, self.N_active_tx + 1) - (self.N_active_tx + 1) / 2), np.zeros(self.N_active_tx))
        )
        self.pitch_n = round(self.pitch / self.dx)

        self.pitch_n_temp = self.pitch_n - 1 if not rem(self.pitch_n, 2) else self.pitch_n

    def run_simulation(self, is_gpu_simulation, binary_name="kspaceFirstOrder-OMP"):
        num_elements = self.N_active_tx
        element_pos_x = self.el_pos_cart[:, 0]
        tone_burst_offset = element_pos_x * np.sin(self.steering_angle * np.pi / 180) / (self.c0_exact * self.kgrid.dt)
        offset_time = np.min(tone_burst_offset)
        tone_burst_offset = np.round(-(offset_time) + tone_burst_offset)
        pulse_waveform = (self.source_strength / (self.c0_exact * self.rho0)) * tone_burst(
            1 / self.kgrid.dt,
            self.tone_burst_freq,
            self.tone_burst_cycles,
            signal_offset=tone_burst_offset.astype(np.int32),
            signal_length=self.kgrid.Nt,
        )

        self.source.ux = np.empty((num_elements * self.pitch_n_temp, self.kgrid.Nt))
        self.source.u_mask = np.zeros((self.Nx, self.Ny))
        tx_apodization = np.ones(num_elements)

        for n in range(self.N_active_tx):
            source_y_pos = round(self.Ny / 2) - round((self.pitch / self.ROIy * self.Ny) * ((self.N_active_tx + 1) / 2 - n))
            start_idx = round(source_y_pos - (self.pitch_n_temp - 1) / 2)
            stop_idx = round(source_y_pos + (self.pitch_n_temp - 1) / 2) + 1
            self.source.u_mask[0, start_idx:stop_idx] = 1
            self.source.ux[n * self.pitch_n_temp : (n + 1) * self.pitch_n_temp, :] = np.tile(
                tx_apodization[n] * pulse_waveform[n], (self.pitch_n_temp, 1)
            )

        self.sensor.mask = self.source.u_mask
        simulation_options = SimulationOptions(
            pml_inside=False, pml_size=self.PML_size, data_cast=self.DATA_CAST, save_to_disk=True, data_recast=True
        )
        execution_options = ExecutionOptions(is_gpu_simulation=is_gpu_simulation, binary_name=binary_name)

        _ = kspaceFirstOrder2D(
            kgrid=self.kgrid,
            medium=self.medium,
            source=self.source,
            sensor=self.sensor,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )

    @patch("kwave.kspaceFirstOrder2D.Executor.run_simulation")
    def test_simulation_cpu_none(self, mock_run_simulation):
        mock_run_simulation.return_value = None
        self._test_simulation(mock_run_simulation, is_gpu_simulation=False, binary_name=None)

    @pytest.mark.skipif(kwave.PLATFORM == "darwin", reason="GPU simulations are currently not supported on MacOS")
    @patch("kwave.kspaceFirstOrder2D.Executor.run_simulation")
    def test_simulation_gpu_none(self, mock_run_simulation):
        mock_run_simulation.return_value = None
        self._test_simulation(mock_run_simulation, is_gpu_simulation=True, binary_name=None)

    @pytest.mark.skipif(kwave.PLATFORM == "darwin", reason="GPU simulations are currently not supported on MacOS")
    @patch("kwave.kspaceFirstOrder2D.Executor.run_simulation")
    def test_simulation_gpu_cuda(self, mock_run_simulation):
        mock_run_simulation.return_value = None
        self._test_simulation(mock_run_simulation, is_gpu_simulation=True, binary_name="kspaceFirstOrder-CUDA")

    @patch("kwave.kspaceFirstOrder2D.Executor.run_simulation")
    def test_simulation_gpu_cuda_failure_darwin(self, mock_run_simulation):
        mock_run_simulation.return_value = None
        expected_error_msg = "GPU simulations are currently not supported on MacOS. Try running the simulation on CPU by setting is_gpu_simulation=False."
        with patch("kwave.PLATFORM", "darwin"):
            with pytest.raises(ValueError, match=expected_error_msg):
                self._test_simulation(mock_run_simulation, is_gpu_simulation=True, binary_name="kspaceFirstOrder-CUDA")

    @patch("kwave.kspaceFirstOrder2D.Executor.run_simulation")
    def test_simulation_cpu_omp(self, mock_run_simulation):
        mock_run_simulation.return_value = None
        self._test_simulation(mock_run_simulation, is_gpu_simulation=False, binary_name="kspaceFirstOrder-OMP")

    def _test_simulation(self, mock_run_simulation, is_gpu_simulation, binary_name):
        self.run_simulation(is_gpu_simulation=is_gpu_simulation, binary_name=binary_name)
        mock_run_simulation.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
