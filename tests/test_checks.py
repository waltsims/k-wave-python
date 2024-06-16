import unittest
from unittest.mock import patch
import numpy as np
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions as ExecutionOptions
from kwave.utils.signals import tone_burst


class TestUltrasoundSimulation(unittest.TestCase):
    @patch("kwave.kspaceFirstOrder2D.Executor.run_simulation")
    def test_simulation(self, mock_run_simulation):
        mock_run_simulation.return_value = None

        # Parameters
        steering_angle = np.arange(-45, -40, 5)
        n_steering_angle = len(steering_angle)

        # Initialize
        source = kSource()
        sensor = kSensor()

        # Simulation settings
        DATA_CAST = "single"  # Use float32 for GPU computations

        # Create the computational grid
        ROIx = 120e-3  # ROI [m]
        ROIy = 120e-3  # ROI [m]
        PML_size = 100  # Size of the PML in grid points
        Nx = 3000  # Number of grid points in the x (row) direction
        Ny = 3000  # Number of grid points in the y (column) direction
        dx = ROIx / Nx  # Grid point spacing in the x direction [m]
        dy = ROIy / Ny  # Grid point spacing in the y direction [m]

        kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))
        f0 = 3.5e6  # Center frequency [Hz]

        # Sensor
        pitch = 0.25e-3  # Length between each sensor [m]
        N_active_tx = 80  # Number of elements of the transducer

        # Define the medium properties
        c0_exact = 1540  # [m/s]
        rho0 = 1000  # [kg/m^3]
        alpha_coeff_exact = 0.5  # [dB/(MHz^y cm)]
        medium = kWaveMedium(c0_exact, rho0, alpha_coeff_exact)

        # Create the time array
        t_end = (Nx * dx) * 2.5 / c0_exact  # [s]
        dt = 1 / 62.5e6 / 2  # Sampling time [s]
        cfl = dt * c0_exact / dx  # Default: 0.3
        kgrid.makeTime(c0_exact, cfl, t_end)

        # Define the input signal
        source_strength = 1e1  # [Pa]
        tone_burst_freq = f0  # [Hz]
        tone_burst_cycles = 5  # 5 is optimum

        # Source position
        el_pos_cart = np.column_stack((pitch * (np.arange(1, N_active_tx + 1) - (N_active_tx + 1) / 2), np.zeros(N_active_tx)))
        pitch_n = round(pitch / dx)
        from kwave.utils.matlab import rem

        pitch_n_temp = pitch_n - 1 if not rem(pitch_n, 2) else pitch_n

        for n_input_signal in range(n_steering_angle):
            num_elements = N_active_tx
            element_pos_x = el_pos_cart[:, 0]
            tone_burst_offset = element_pos_x * np.sin(steering_angle[n_input_signal] * np.pi / 180) / (c0_exact * dt)
            offset_time = np.min(tone_burst_offset)
            tone_burst_offset = np.round(-(offset_time) + tone_burst_offset)
            pulse_waveform = (source_strength / (c0_exact * rho0)) * tone_burst(
                1 / kgrid.dt, tone_burst_freq, tone_burst_cycles, signal_offset=tone_burst_offset.astype(np.int32), signal_length=kgrid.Nt
            )

            source.ux = np.empty((num_elements * pitch_n_temp, kgrid.Nt))
            source.u_mask = np.zeros((Nx, Ny))
            tx_apodization = np.ones(num_elements)

            for n in range(N_active_tx):
                source_y_pos = round(Ny / 2) - round((pitch / ROIy * Ny) * ((N_active_tx + 1) / 2 - n))
                start_idx = round(source_y_pos - (pitch_n_temp - 1) / 2)
                stop_idx = round(source_y_pos + (pitch_n_temp - 1) / 2) + 1
                source.u_mask[0, start_idx:stop_idx] = 1
                source.ux[n * pitch_n_temp : (n + 1) * pitch_n_temp, :] = np.tile(tx_apodization[n] * pulse_waveform[n], (pitch_n_temp, 1))

            sensor.mask = source.u_mask
            simulation_options = SimulationOptions(
                pml_inside=False, pml_size=PML_size, data_cast=DATA_CAST, save_to_disk=True, data_recast=True
            )
            _ = kspaceFirstOrder2D(
                kgrid=kgrid,
                medium=medium,
                source=source,
                sensor=sensor,
                simulation_options=simulation_options,
                execution_options=ExecutionOptions(is_gpu_simulation=True),
            )

            mock_run_simulation.assert_called_once()
