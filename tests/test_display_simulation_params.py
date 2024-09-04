import unittest
from unittest.mock import Mock
import numpy as np
from testfixtures import LogCapture

from kwave.kWaveSimulation_helper.display_simulation_params import print_grid_size, print_max_supported_freq


class TestDisplaySimulationParams(unittest.TestCase):
    def test_print_grid_size_1D(self):
        mock_kgrid = Mock(dim=1, Nx=100, size=np.array([0.1]))

        with LogCapture() as log:
            print_grid_size(mock_kgrid, 1)

        log.check(("root", "INFO", "  input grid size: 100 grid points (100mm)"))

    def test_print_grid_size_2D(self):
        mock_kgrid = Mock(dim=2, Nx=100, Ny=100, size=np.array([0.1, 0.1]))

        with LogCapture() as log:
            print_grid_size(mock_kgrid, 1)

        log.check(("root", "INFO", "  input grid size: 100 by 100 grid points (0.1 by 0.1m)"))

    def test_print_grid_size_3D(self):
        mock_kgrid = Mock(dim=3, Nx=100, Ny=100, Nz=100, size=np.array([0.1, 0.1, 0.1]))

        with LogCapture() as log:
            print_grid_size(mock_kgrid, 1)

        log.check(("root", "INFO", "  input grid size: 100 by 100 by 100 grid points (0.1 by 0.1 by 0.1m)"))

    def test_print_max_supported_freq_1D(self):
        mock_kgrid = Mock(dim=1, k_max_all=1000)
        c_min = 1500

        with LogCapture() as log:
            print_max_supported_freq(mock_kgrid, c_min)

        log.check(("root", "INFO", "  maximum supported frequency: 238.732415kHz"))

    def test_print_max_supported_freq_2D_isotropic(self):
        mock_kgrid = Mock(dim=2, k_max=Mock(x=1000, y=1000), k_max_all=1000)
        c_min = 1500

        with LogCapture() as log:
            print_max_supported_freq(mock_kgrid, c_min)

        log.check(("root", "INFO", "  maximum supported frequency: 238.732415kHz"))

    def test_print_max_supported_freq_2D_anisotropic(self):
        mock_kgrid = Mock(dim=2, k_max=Mock(x=1000, y=2000), k_max_all=2000)
        c_min = 1500

        with LogCapture() as log:
            print_max_supported_freq(mock_kgrid, c_min)

        log.check(("root", "INFO", "  maximum supported frequency: 238.732415kHz by 477.464829kHz"))

    def test_print_max_supported_freq_3D_isotropic(self):
        mock_kgrid = Mock(dim=3, k_max=Mock(x=1000, y=1000, z=1000), k_max_all=1000)
        c_min = 1500

        with LogCapture() as log:
            print_max_supported_freq(mock_kgrid, c_min)

        log.check(("root", "INFO", "  maximum supported frequency: 238.732415kHz"))

    def test_print_max_supported_freq_3D_anisotropic(self):
        mock_kgrid = Mock(dim=3, k_max=Mock(x=1000, y=2000, z=3000), k_max_all=3000)
        c_min = 1500

        with LogCapture() as log:
            print_max_supported_freq(mock_kgrid, c_min)
        log.check(("root", "INFO", "  maximum supported frequency: 238.732415kHz by 477.464829kHz by 716.197244kHz"))


if __name__ == "__main__":
    unittest.main()
