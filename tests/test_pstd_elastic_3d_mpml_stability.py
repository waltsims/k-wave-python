"""
Unit test to test the stability of the pml and m-pml
"""

import numpy as np
from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic3D import pstd_elastic_3d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.signals import tone_burst
from kwave.utils.mapgen import make_spherical_section


def test_pstd_elastic_3d_mpml_stability():
    test_pass: bool = True

    # create the computational grid
    PML_SIZE: int = 10
    Nx: int = 80 - 2 * PML_SIZE
    Ny: int = 64 - 2 * PML_SIZE
    Nz: int = 64 - 2 * PML_SIZE
    dx = 0.1e-3
    dy = 0.1e-3
    dz = 0.1e-3
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the upper layer of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny, Nz))  # [m/s]
    sound_speed_shear       = np.zeros((Nx, Ny, Nz))          # [m/s]
    density                 = 1000.0 * np.ones((Nx, Ny, Nz))    # [kg/m^3]
    medium = kWaveMedium(sound_speed_compression,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear,
                         density=density)

    # define the properties of the lower layer of the propagation medium
    medium.sound_speed_compression[Nx // 2 - 1:, :, :] = 2000.0  # [m/s]
    medium.sound_speed_shear[Nx // 2 - 1:, :, :]       = 1000.0  # [m/s]
    medium.density[Nx // 2 - 1, :, :]                  = 1200.0  # [kg/m^3]

    # create the time array
    cfl   = 0.3   # Courant-Friedrichs-Lewy number
    t_end = 8e-6  # [s]
    kgrid.makeTime(medium.sound_speed_compression.max(), cfl, t_end)

    # define the source mask
    s_rad: int = 15
    s_height: int = 8
    offset: int = 14
    ss, _ = make_spherical_section(s_rad, s_height)

    source = kSource()
    ss_width = np.shape(ss)[1]
    ss_half_width: int = np.floor(ss_width // 2).astype(int)
    y_start_pos: int = Ny // 2 - ss_half_width - 1
    y_end_pos: int = y_start_pos + ss_width
    z_start_pos: int = Nz // 2 - ss_half_width -1
    z_end_pos: int = z_start_pos + ss_width
    source.s_mask = np.zeros((Nx, Ny, Nz), dtype=int)

    print(offset, s_height, y_start_pos, y_end_pos, z_start_pos, z_end_pos)

    source.s_mask[offset:s_height + offset, y_start_pos:y_end_pos, z_start_pos:z_end_pos] = ss.astype(int)
    source.s_mask[:, :, Nz // 2 - 1] = 0

    # define the source signal
    source.sxx = tone_burst(1.0 / kgrid.dt, 1e6, 3)
    source.syy = source.sxx
    source.szz = source.sxx

    # define sensor
    sensor = kSensor()
    sensor.record = ['u_final']

    # define input arguments
    simulation_options_pml = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                               kelvin_voigt_model=True,
                                               use_sensor=True,
                                               pml_inside=False,
                                               pml_size=PML_SIZE,
                                               blank_sensor=True,
                                               binary_sensor_mask=True,
                                               multi_axial_PML_ratio=0.0)

    # run the simulations
    sensor_data_pml = pstd_elastic_3d(deepcopy(kgrid),
                                      medium=deepcopy(medium),
                                      source=deepcopy(source),
                                      sensor=deepcopy(sensor),
                                      simulation_options=deepcopy(simulation_options_pml))

    simulation_options_mpml = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           kelvin_voigt_model=True,
                                           use_sensor=True,
                                           pml_inside=False,
                                           pml_size=PML_SIZE,
                                           blank_sensor=True,
                                           binary_sensor_mask=True,
                                           multi_axial_PML_ratio=0.1)

    sensor_data_mpml = pstd_elastic_3d(deepcopy(kgrid),
                                       deepcopy(medium),
                                       deepcopy(source),
                                       deepcopy(sensor),
                                       simulation_options=deepcopy(simulation_options_mpml))

    # check magnitudes
    pml_max = np.max([sensor_data_pml.ux_final, sensor_data_pml.uy_final, sensor_data_pml.uz_final])
    mpml_max = np.max([sensor_data_mpml.ux_final,sensor_data_mpml.uy_final, sensor_data_mpml.uz_final])

    # set reference magnitude (initial source)
    ref_max = 1.0 / np.max(medium.sound_speed_shear * medium.density)

    # check results - the test should fail if the pml DOES work (i.e., it
    # doesn't become unstable), or if the m-pml DOESN'T work (i.e., it does
    # become unstable)
    if pml_max < ref_max or mpml_max > ref_max:
        test_pass = False

    return test_pass


