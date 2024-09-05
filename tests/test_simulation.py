import unittest
import numpy as np
import pytest
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D

from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions as ExecutionOptions
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc


class TestUltrasoundSimulationRecording(unittest.TestCase):
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

        # create the time array
        self.kgrid.makeTime(self.medium.sound_speed)


    def run_simulation(self, record_fields):
        self.sensor.record = record_fields

        simulation_options = SimulationOptions(
            pml_inside=False, pml_size=self.PML_size, data_cast=self.DATA_CAST,
            save_to_disk=True, data_recast=True, smooth_p0=False,
        )
        execution_options = ExecutionOptions(is_gpu_simulation=False, binary_name=self.binary_name)

        sensor_data = kspaceFirstOrder2D(
            kgrid=self.kgrid,
            medium=self.medium,
            source=self.source,
            sensor=self.sensor,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )

        return sensor_data

    def test_record_final_pressure(self):
        record_fields = ['p_final']
        sensor_data = self._test_simulation(record_fields=record_fields)
        for field in record_fields:
            assert field in sensor_data

    def test_record_pressure(self):
        record_fields = ['p']
        sensor_data = self._test_simulation(record_fields=record_fields)
        for field in record_fields:
            assert field in sensor_data

    def _test_simulation(self, record_fields):
        return self.run_simulation(record_fields)


if __name__ == "__main__":
    pytest.main([__file__])
