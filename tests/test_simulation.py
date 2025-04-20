import unittest

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.kWaveSimulation import kWaveSimulation
from kwave.options import SimulationExecutionOptions, SimulationOptions
from kwave.reconstruction import TimeReversal
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc


class TestSimulation(unittest.TestCase):
    def setUp(self):
        # Initialize
        self.source = kSource()
        self.sensor = kSensor()

        # Simulation settings
        self.DATA_CAST = "single"  # Use float32 for GPU computations
        self.binary_name = "kspaceFirstOrder-OMP"

        # create the computational grid
        self.PML_size = 20  # size of the PML in grid points
        self.N = Vector([64, 64])  # number of grid points
        self.d = Vector([0.1e-3, 0.1e-3])  # grid point spacing [m]
        self.kgrid = kWaveGrid(self.N, self.d)

        # Define the source
        self.disc_magnitude = 5  # [Pa]
        self.disc_pos = self.N // 2  # [grid points]
        self.disc_radius = 8  # [grid points]
        self.disc = self.disc_magnitude * make_disc(self.N, self.disc_pos, self.disc_radius)
        self.p0 = smooth(self.disc, restore_max=True)
        self.source.p0 = self.p0

        # Define the medium properties
        self.c0_exact = 1540  # [m/s]
        self.rho0 = 1000  # [kg/m^3]
        self.alpha_coeff_exact = 0.5  # [dB/(MHz^y cm)]
        self.medium = kWaveMedium(self.c0_exact, self.rho0, self.alpha_coeff_exact)

        # Define the sensor
        self.sensor.mask = np.zeros(self.N)
        self.sensor.mask[0] = 1

        # define simulation options
        self.simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.PML_size,
            data_cast=self.DATA_CAST,
            save_to_disk=True,
            data_recast=True,
            smooth_p0=False,
        )

        # create the time array
        self.kgrid.makeTime(self.medium.sound_speed)

    def test_record_final_pressure(self):
        self.sensor.record = ["p_final"]
        k_sim = kWaveSimulation(
            kgrid=self.kgrid, source=self.source, sensor=self.sensor, medium=self.medium, simulation_options=self.simulation_options
        )
        k_sim.input_checking("kspaceFirstOrder2D")

        recorder = k_sim.record.__dict__
        for key, val in recorder.items():
            if key == "p_final":
                assert val
            elif key.startswith(("p", "u", "I")):
                assert not val

    def test_time_reversal(self):
        """Test time reversal reconstruction."""
        # Create test data
        kgrid = kWaveGrid([100, 100], [0.1, 0.1])
        kgrid.setTime(100, 1e-6)
        medium = kWaveMedium(sound_speed=1500)
        sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

        # Create simulation options
        simulation_options = SimulationOptions(save_to_disk=True, save_to_disk_exit=True, pml_inside=False, pml_size=20, data_cast="single")
        execution_options = SimulationExecutionOptions()

        # Create time reversal handler
        tr = TimeReversal(kgrid, medium, sensor)

        # Mock simulation function
        def mock_simulation(*args, **kwargs):
            return {"p_final": np.ones((100, 100))}

        # Set up recorded pressure data
        sensor.recorded_pressure = np.ones((100, 100))  # Mock recorded pressure data

        # Run reconstruction
        p0_recon = tr(mock_simulation, simulation_options, execution_options)

        # Verify reconstruction
        assert p0_recon.shape == (100, 100)
        assert np.all(p0_recon >= 0)  # Check positivity condition
        assert np.all(p0_recon <= 2.0)  # Check compensation factor

    def test_record_pressure(self):
        self.sensor.record = ["p"]
        k_sim = kWaveSimulation(
            kgrid=self.kgrid, source=self.source, sensor=self.sensor, medium=self.medium, simulation_options=self.simulation_options
        )
        k_sim.input_checking("kspaceFirstOrder2D")

        recorder = k_sim.record.__dict__
        for key, val in recorder.items():
            if key == "p":
                assert val
            elif key.startswith(("p", "u", "I")):
                assert not val


if __name__ == "__main__":
    pytest.main([__file__])
