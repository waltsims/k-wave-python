
"""
Unit test to check that the split field components sum to give the correct field, e.g., ux = ux^p + ux^s.
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
from kwave.utils.mapgen import make_bowl

import scipy.io as sio

def test_pstd_elastic_3d_check_split_field():

    # set comparison threshold
    COMPARISON_THRESH: float = 1e-15

    # set pass variable
    test_pass: bool = True

    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================

    # create the computational grid
    PML_size: int = 10           # [grid points]
    Nx: int = 64 - 2 * PML_size  # [grid points]
    Ny: int = 64 - 2 * PML_size  # [grid points]
    Nz: int = 64 - 2 * PML_size  # [grid points]
    dx: float = 0.5e-3           # [m]
    dy: float = 0.5e-3           # [m]
    dz: float = 0.5e-3           # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the medium properties for the top layer
    cp1: float = 1540.0     # compressional wave speed [m/s]
    cs1: float = 0.0        # shear wave speed [m/s]
    rho1: float = 1000.0    # density [kg/m^3]
    alpha0_p1: float = 0.1  # compressional absorption [dB/(MHz^2 cm)]
    alpha0_s1: float = 0.1  # shear absorption [dB/(MHz^2 cm)]

    # define the medium properties for the bottom layer
    cp2: float = 3000.0     # compressional wave speed [m/s]
    cs2: float = 1400.0     # shear wave speed [m/s]
    rho2: float = 1850.0    # density [kg/m^3]
    alpha0_p2: float = 1.0  # compressional absorption [dB/(MHz^2 cm)]
    alpha0_s2: float = 1.0  # shear absorption [dB/(MHz^2 cm)]

    # create the time array
    cfl: float = 0.1
    t_end: float = 15e-6
    kgrid.makeTime(cp1, cfl, t_end)

    # define position of heterogeneous slab
    slab = np.zeros((Nx, Ny, Nz))
    slab[Nx // 2 - 1:, :, :] = 1

    # define the source geometry in SI units (where 0, 0 is the grid center)
    bowl_pos = [-6e-3, -6e-3, -6e-3]  # [m]
    focus_pos = [5e-3, 5e-3, 5e-3]    # [m]
    radius = 15e-3                    # [m]
    diameter = 10e-3                  # [m]

    # define the driving signal
    source_freq = 500e3    # [Hz]
    source_strength = 1e6  # [Pa]
    source_cycles = 3      # number of tone burst cycles

    # define the sensor to record the maximum particle velocity everywhere
    sensor = kSensor()
    sensor.record = ['u_split_field', 'u_non_staggered']
    sensor.mask = np.zeros((Nx, Ny, Nz))
    sensor.mask[:, :, Nz // 2 - 1] = 1

    # convert the source parameters to grid points
    bowl_pos    = np.round(np.asarray(bowl_pos) / dx).astype(int) + np.asarray([Nx // 2 - 1, Ny // 2 - 1, Nz // 2 - 1])
    focus_pos   = np.round(np.asarray(focus_pos) / dx).astype(int) + np.asarray([Nx // 2 - 1, Ny // 2 - 1, Nz // 2 - 1])
    radius      = int(round(radius / dx))
    diameter    = int(round(diameter / dx))

    print(bowl_pos)
    print(focus_pos)

    # force the diameter to be odd
    if diameter % 2 == 0:
        diameter: int = diameter + int(1)

    # define the medium properties
    sound_speed_compression = cp1 * np.ones((Nx, Ny, Nz))
    sound_speed_shear = cs1 * np.ones((Nx, Ny, Nz))
    density = rho1 * np.ones((Nx, Ny, Nz))
    alpha_coeff_compression = alpha0_p1 * np.ones((Nx, Ny, Nz))
    alpha_coeff_shear = alpha0_s1 * np.ones((Nx, Ny, Nz))

    medium = kWaveMedium(sound_speed_compression,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear,
                         density=density,
                         alpha_coeff_compression=alpha_coeff_compression,
                         alpha_coeff_shear=alpha_coeff_shear)

    medium.sound_speed_compression[slab == 1] = cp2
    medium.sound_speed_shear[slab == 1] = cs2
    medium.density[slab == 1] = rho2
    medium.alpha_coeff_compression[slab == 1] = alpha0_p2
    medium.alpha_coeff_shear[slab == 1] = alpha0_s2

    # generate the source geometry
    source_mask = make_bowl(Vector([Nx, Ny, Nz]), Vector(bowl_pos), radius, diameter, Vector(focus_pos))

    # assign the source
    source = kSource()
    source.s_mask = source_mask
    fs = 1.0 / kgrid.dt
    source.sxx = -source_strength * tone_burst(fs, source_freq, source_cycles)
    source.syy = source.sxx
    source.szz = source.sxx

    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           pml_inside=False,
                                           pml_size=PML_size,
                                           kelvin_voigt_model=False)

    # run the elastic simulation
    sensor_data_elastic = pstd_elastic_3d(deepcopy(kgrid),
                                          medium=deepcopy(medium),
                                          source=deepcopy(source),
                                          sensor=deepcopy(sensor),
                                          simulation_options=deepcopy(simulation_options))

    # compute errors
    diff_ux = np.max(np.abs(sensor_data_elastic['ux_non_staggered'] -
                            sensor_data_elastic['ux_split_p'] -
                            sensor_data_elastic['ux_split_s'])) / np.max(np.abs(sensor_data_elastic['ux_non_staggered']))

    diff_uy = np.max(np.abs(sensor_data_elastic['uy_non_staggered'] -
                            sensor_data_elastic['ux_split_p'] -
                            sensor_data_elastic['uy_split_s'])) / np.max(np.abs(sensor_data_elastic['uy_non_staggered']))

    diff_uz = np.max(np.abs(sensor_data_elastic['uz_non_staggered'] -
                            sensor_data_elastic['uz_split_p'] -
                            sensor_data_elastic['uz_split_s'])) / np.max(np.abs(sensor_data_elastic['uz_non_staggered']))

    mat_contents = sio.loadmat('split_field.mat')

    ux_split_p = mat_contents['ux_split_p']
    uy_split_p = mat_contents['uy_split_p']
    uz_split_p = mat_contents['uz_split_p']

    ux_non_staggered = mat_contents['ux_non_staggered']
    uy_non_staggered = mat_contents['uy_non_staggered']
    uz_non_staggered = mat_contents['uz_non_staggered']

    diff_mat_ux_non_staggered = np.max(np.abs(sensor_data_elastic['ux_non_staggered'] - ux_non_staggered))
    diff_mat_uy_non_staggered = np.max(np.abs(sensor_data_elastic['uy_non_staggered'] - uy_non_staggered))
    diff_mat_uz_non_staggered = np.max(np.abs(sensor_data_elastic['uz_non_staggered'] - uz_non_staggered))

    diff_mat_ux_split_p = np.max(np.abs(sensor_data_elastic['ux_split_p'] - ux_split_p))
    diff_mat_uy_split_p = np.max(np.abs(sensor_data_elastic['uy_split_p'] - uy_split_p))
    diff_mat_uz_split_p = np.max(np.abs(sensor_data_elastic['uz_split_p'] - uz_split_p))

    # diff_mat_ux_split_s = np.max(np.abs(sensor_data_elastic['ux_split_s'] - ux_split_s))
    # diff_mat_uy_split_s = np.max(np.abs(sensor_data_elastic['uy_split_s'] - uy_split_s))
    # diff_mat_uz_split_s = np.max(np.abs(sensor_data_elastic['uz_split_s'] - uz_split_s))

    print("diff_mat_ux_non_staggered:", diff_mat_ux_non_staggered,
          np.max(np.abs(sensor_data_elastic['ux_non_staggered'])), np.argmax(np.abs(sensor_data_elastic['ux_non_staggered'])),
          np.max(np.abs(ux_non_staggered)), np.argmax(np.abs(ux_non_staggered)))
    print("diff_mat_uy_non_staggered:", diff_mat_uy_non_staggered,
          np.max(np.abs(sensor_data_elastic['uy_non_staggered'])), np.argmax(np.abs(sensor_data_elastic['uy_non_staggered'])),
          np.max(np.abs(uy_non_staggered)), np.argmax(np.abs(uy_non_staggered)))
    print("diff_mat_uz_non_staggered:", diff_mat_uz_non_staggered,
          np.max(np.abs(sensor_data_elastic['uz_non_staggered'])), np.argmax(np.abs(sensor_data_elastic['uz_non_staggered'])),
          np.max(np.abs(uz_non_staggered)), np.argmax(np.abs(uz_non_staggered)))

    print("diff_mat_ux_split_p:", diff_mat_ux_split_p,
          np.max(np.abs(sensor_data_elastic['ux_split_p'])), np.argmax(np.abs(sensor_data_elastic['ux_split_p'])),
          np.max(np.abs(ux_split_p)), np.argmax(np.abs(ux_split_p)))
    print("diff_mat_uy_split_p:", diff_mat_uy_split_p,
          np.max(np.abs(sensor_data_elastic['uy_split_p'])), np.argmax(np.abs(sensor_data_elastic['uy_split_p'])),
          np.max(np.abs(uy_split_p)), np.argmax(np.abs(uy_split_p)))
    print("diff_mat_uz_split_p:", diff_mat_uz_split_p,
          np.max(np.abs(sensor_data_elastic['uz_split_p'])), np.argmax(np.abs(sensor_data_elastic['uz_split_p'])),
          np.max(np.abs(uz_split_p)), np.argmax(np.abs(uz_split_p)))

    # check for test pass
    if (diff_ux > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "diff_ux: " + str(diff_ux)

    if (diff_uy > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "diff_uy: " + str(diff_uy)

    if (diff_uz > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "diff_uz: " + str(diff_uz)