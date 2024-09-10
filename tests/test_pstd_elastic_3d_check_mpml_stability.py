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

# import scipy.io as sio

def test_pstd_elastic_3d_check_mpml_stability():

    test_pass: bool = True

    # mat_contents = sio.loadmat('C:/Users/dsinden/dev/octave/k-Wave/testing/unit/mpml_stability.mat')

    # mpml_smask = mat_contents['s_mask']

    # create the computational grid
    PML_SIZE: int = 10
    Nx: int = 80 - 2 * PML_SIZE
    Ny: int = 64 - 2 * PML_SIZE
    Nz: int = 64 - 2 * PML_SIZE
    dx: float = 0.1e-3
    dy: float = 0.1e-3
    dz: float = 0.1e-3
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the upper layer of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny, Nz))  # [m/s]
    sound_speed_shear = np.zeros((Nx, Ny, Nz))                # [m/s]
    density = 1000.0 * np.ones((Nx, Ny, Nz))                  # [kg/m^3]
    # define the properties of the lower layer of the propagation medium
    sound_speed_compression[Nx // 2 - 1:, :, :] = 2000.0      # [m/s]
    sound_speed_shear[Nx // 2 - 1:, :, :] = 1000.0            # [m/s]
    density[Nx // 2 - 1:, :, :] = 1200.0                      # [kg/m^3]
    medium = kWaveMedium(sound_speed_compression,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear,
                         density=density)

    # create the time array
    cfl = 0.3     # Courant-Friedrichs-Lewy number
    t_end = 8e-6  # [s]
    kgrid.makeTime(medium.sound_speed_compression.max(), cfl, t_end)

    # define the source mask
    s_rad: int = 15
    s_height: int = 8
    offset: int = 15
    ss, _ = make_spherical_section(s_rad, s_height)

    source = kSource()
    ss_width: int = np.shape(ss)[1]
    ss_half_width: int = np.floor(ss_width / 2).astype(int)
    y_start_pos: int = Ny // 2 - ss_half_width - 1
    y_end_pos: int = y_start_pos + ss_width
    z_start_pos: int = Nz // 2 - ss_half_width - 1
    z_end_pos: int = z_start_pos + ss_width

    source.s_mask = np.zeros((Nx, Ny, Nz), dtype=int)

    source.s_mask[offset:s_height + offset, y_start_pos:y_end_pos, z_start_pos:z_end_pos] = ss.astype(int)
    source.s_mask[:, :, Nz // 2 - 1:] = int(0)

    # print("diff_mat_pml_smask:", np.shape(mpml_smask), np.shape(source.s_mask),
    #       np.max(np.abs(mpml_smask - source.s_mask)), np.argmax(np.abs(mpml_smask - source.s_mask)),
    #       np.nonzero(np.abs(mpml_smask - source.s_mask)),
    #       np.max(np.abs(source.s_mask)), np.max(np.abs(mpml_smask)))

    # define the source signal
    fs = 1.0 / kgrid.dt
    source.sxx = tone_burst(sample_freq=fs, signal_freq=1e6, num_cycles=3)

    # print(source.sxx)

    source.syy = deepcopy(source.sxx)
    source.szz = deepcopy(source.sxx)

    # define sensor
    sensor = kSensor()
    sensor.record = ['u_final']

    # define input arguments
    simulation_options_pml = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                               kelvin_voigt_model=False,
                                               use_sensor=True,
                                               pml_inside=False,
                                               pml_size=PML_SIZE,
                                               blank_sensor=True,
                                               binary_sensor_mask=True,
                                               multi_axial_PML_ratio=0.0)

    # run the simulations
    sensor_data_pml = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                      medium=deepcopy(medium),
                                      source=deepcopy(source),
                                      sensor=deepcopy(sensor),
                                      simulation_options=deepcopy(simulation_options_pml))

    simulation_options_mpml = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                kelvin_voigt_model=False,
                                                use_sensor=True,
                                                pml_inside=False,
                                                pml_size=PML_SIZE,
                                                blank_sensor=True,
                                                binary_sensor_mask=True,
                                                multi_axial_PML_ratio=0.1)

    sensor_data_mpml = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                       medium=deepcopy(medium),
                                       source=deepcopy(source),
                                       sensor=deepcopy(sensor),
                                       simulation_options=deepcopy(simulation_options_mpml))

    # check magnitudes
    pml_max = np.max([sensor_data_pml.ux_final, sensor_data_pml.uy_final, sensor_data_pml.uz_final])
    mpml_max = np.max([sensor_data_mpml.ux_final, sensor_data_mpml.uy_final, sensor_data_mpml.uz_final])

    # set reference magnitude (initial source)
    ref_max = 1.0 / (np.max(medium.sound_speed_shear) * np.max(medium.density))

    # pml_ux_final = mat_contents['pml_ux_final']
    # pml_uy_final = mat_contents['pml_uy_final']
    # pml_uz_final = mat_contents['pml_uz_final']

    # mpml_ux_final = mat_contents['mpml_ux_final']
    # mpml_uy_final = mat_contents['mpml_uy_final']
    # mpml_uz_final = mat_contents['mpml_uz_final']

    # diff_mat_pml_ux_final = np.max(np.abs(sensor_data_pml.ux_final - pml_ux_final))
    # diff_mat_pml_uy_final = np.max(np.abs(sensor_data_pml.uy_final - pml_uy_final))
    # diff_mat_pml_uz_final = np.max(np.abs(sensor_data_pml.uz_final - pml_uz_final))

    # diff_mat_mpml_ux_final = np.max(np.abs(sensor_data_mpml.ux_final - mpml_ux_final))
    # diff_mat_mpml_uz_final = np.max(np.abs(sensor_data_mpml.uy_final - mpml_uy_final))
    # diff_mat_mpml_uy_final = np.max(np.abs(sensor_data_mpml.uz_final - mpml_uz_final))

    # print("diff_mat_pml_ux_final:", diff_mat_pml_ux_final,
    #       np.max(np.abs(sensor_data_pml.ux_final)), np.argmax(np.abs(sensor_data_pml.ux_final)),
    #       np.max(np.abs(pml_ux_final)), np.argmax(np.abs(pml_ux_final)))
    # print("diff_mat_pml_uy_final:", diff_mat_pml_uy_final,
    #       np.max(np.abs(sensor_data_pml.uy_final)), np.argmax(np.abs(sensor_data_pml.uy_final)),
    #       np.max(np.abs(pml_uy_final)), np.argmax(np.abs(pml_uy_final)))
    # print("diff_mat_pml_uz_final:", diff_mat_pml_uz_final,
    #       np.max(np.abs(sensor_data_pml.uz_final)), np.argmax(np.abs(sensor_data_pml.uz_final)),
    #       np.max(np.abs(pml_uz_final)), np.argmax(np.abs(pml_uz_final)))

    # print("diff_mat_mpml_ux_final:", diff_mat_mpml_ux_final,
    #       np.max(np.abs(sensor_data_mpml.ux_final)), np.argmax(np.abs(sensor_data_mpml.ux_final)),
    #       np.max(np.abs(mpml_ux_final)), np.argmax(np.abs(mpml_ux_final)))
    # print("diff_mat_mpml_uy_final:", diff_mat_mpml_uy_final,
    #       np.max(np.abs(sensor_data_mpml.uy_final)), np.argmax(np.abs(sensor_data_mpml.uy_final)),
    #       np.max(np.abs(mpml_uy_final)), np.argmax(np.abs(mpml_uy_final)))
    # print("diff_mat_mpml_uz_final:", diff_mat_mpml_uz_final,
    #       np.max(np.abs(sensor_data_mpml.uz_final)), np.argmax(np.abs(sensor_data_mpml.uz_final)),
    #       np.max(np.abs(mpml_uz_final)), np.argmax(np.abs(mpml_uz_final)))

    # mat_pml_max = np.max([pml_ux_final, pml_uy_final, pml_uz_final])
    # mat_mpml_max = np.max([mpml_ux_final, mpml_uy_final, mpml_uz_final])

    # print("mat_pml_max < ref_max " + str(mat_pml_max < ref_max) + ", mat_pml_max: " + str(mat_pml_max) + ", ref_max: " + str(ref_max))
    # print("mat_mpml_max > ref_max " + str(mat_mpml_max > ref_max) + ", mpml_max: " + str(mat_mpml_max) + ", ref_max: " + str(ref_max))

    # print("pml_max < ref_max " + str(pml_max < ref_max) + ", pml_max: " + str(pml_max) + ", ref_max: " + str(ref_max))
    # print("mpml_max > ref_max " + str(mpml_max > ref_max) + ", mpml_max: " + str(mpml_max) + ", ref_max: " + str(ref_max))

    # check results - the test should fail if the pml DOES work (i.e., it
    # doesn't become unstable), or if the m-pml DOESN'T work (i.e., it does
    # become unstable). The pml should not work and the mpml should.
    if pml_max < ref_max:
        test_pass = False
    assert test_pass, "pml_max < ref_max " + str(pml_max < ref_max) + ", pml_max: " + str(pml_max) + ", ref_max: " + str(ref_max)

    if mpml_max > ref_max:
        test_pass = False
    assert test_pass, "mpml_max > ref_max " + str(mpml_max > ref_max) + ", mpml_max: " + str(mpml_max) + ", ref_max: " + str(ref_max)



