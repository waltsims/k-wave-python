"""
#     Unit test to compare an infinite line source in 2D and 3D in an
#     elastic medium to catch any coding bugs between the pstdElastic2D and
#     pstdElastic3D. 20 tests are performed:
#
#         1.  lossless + source.p0 + homogeneous
#         2.  lossless + source.p0 + heterogeneous
#         3.  lossless + source.s (additive) + homogeneous
#         4.  lossless + source.s (additive) + heterogeneous
#         5.  lossless + source.s (dirichlet) + homogeneous
#         6.  lossless + source.s (dirichlet) + heterogeneous
#         7.  lossless + source.u (additive) + homogeneous
#         8.  lossless + source.u (additive) + heterogeneous
#         9.  lossless + source.u (dirichlet) + homogeneous
#         10. lossless + source.u (dirichlet) + heterogeneous
#         11. lossy + source.p0 + homogeneous
#         12. lossy + source.p0 + heterogeneous
#         13. lossy + source.s (additive) + homogeneous
#         14. lossy + source.s (additive) + heterogeneous
#         15. lossy + source.s (dirichlet) + homogeneous
#         16. lossy + source.s (dirichlet) + heterogeneous
#         17. lossy + source.u (additive) + homogeneous
#         18. lossy + source.u (additive) + heterogeneous
#         19. lossy + source.u (dirichlet) + homogeneous
#         20. lossy + source.u (dirichlet) + heterogeneous
#
#     For each test, the infinite line source in 3D is aligned in all three
#     directions.
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
# import pytest

from scipy.io import loadmat

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic2D import pstd_elastic_2d
from kwave.pstdElastic3D import pstd_elastic_3d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.mapgen import make_circle
from kwave.utils.matlab import rem
from kwave.utils.filters import smooth



def setMaterialProperties(medium: kWaveMedium, N1: int, N2: int, N3: int,
                          direction: int, interface_position: int,
                          cp1: float=1500.0, cs1: float=0.0, rho1: float=1000.0,
                          alpha_p1: float=0.5, alpha_s1: float=0.5):

    # sound speed and density
    medium.sound_speed_compression = cp1 * np.ones((N1, N2, N3), dtype=float)
    medium.sound_speed_shear = cs1 * np.ones((N1, N2, N3), dtype=float)
    medium.density = rho1 * np.ones((N1, N2, N3), dtype=float)

    cp2: float = 2000.0
    cs2: float = 800.0
    rho2: float = 1200.0
    alpha_p2: float = 1.0
    alpha_s2: float = 1.0

    # position of the heterogeneous interface
    if direction == 1:
            medium.sound_speed_compression[interface_position:, :, :] = cp2
            medium.sound_speed_shear[interface_position:, :, :] = cs2
            medium.density[interface_position:, :, :] = rho2
    elif direction == 2:
            medium.sound_speed_compression[:, interface_position:, :] = cp2
            medium.sound_speed_shear[:, interface_position:, :] = cs2
            medium.density[:, interface_position:, :] = rho2

    # compress, so 2D simulations on 2d domain
    medium.sound_speed_compression = np.squeeze(medium.sound_speed_compression)
    medium.sound_speed_shear = np.squeeze(medium.sound_speed_shear)
    medium.density = np.squeeze(medium.density)

    # absorption
    if hasattr(medium, 'alpha_coeff_compression'):
        if medium.alpha_coeff_compression is not None or medium.alpha_coeff_shear is not None:
            print("lossy")
            medium.alpha_coeff_compression = alpha_p1 * np.ones((N1, N2, N3), dtype=np.float32)
            medium.alpha_coeff_shear = alpha_s1 * np.ones((N1, N2, N3), dtype=np.float32)
            if direction == 1:
                medium.alpha_coeff_compression[interface_position:, :, :] = alpha_p2
                medium.alpha_coeff_shear[interface_position:, :, :] = alpha_s2
            elif direction == 2:
                medium.alpha_coeff_compression[:, interface_position:, :] = alpha_p2
                medium.alpha_coeff_shear[:, interface_position:, :] = alpha_s2
            # if 2d or 3d
            medium.alpha_coeff_compression = np.squeeze(medium.alpha_coeff_compression)
            medium.alpha_coeff_shear = np.squeeze(medium.alpha_coeff_shear)
        else:
            print("lossless")


# @pytest.mark.skip(reason="not ready")
def test_pstd_elastic_3d_compare_with_pstd_elastic_2d():

    verbose: bool = True

    # set additional literals to give further permutations of the test
    USE_PML             = True
    COMPARISON_THRESH   = 1e-10
    # this smooths everything not just p0
    SMOOTH_P0_SOURCE    = True
    USE_SG              = True

    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================

    # define grid size
    Nx: int = 64
    Ny: int = 64
    Nz: int = 32
    dx: float = 0.1e-3
    dy: float = 0.1e-3
    dz: float = 0.1e-3

    interface_position: int = Nx // 2 - 1

    # define PML properties
    pml_size: int = 10
    if USE_PML:
        pml_alpha: float = 2.0
    else:
        pml_alpha: float = 0.0

    # define material properties
    cp1: float = 1500.0
    cs1: float = 0.0
    rho1: float = 1000.0
    alpha_p1: float = 0.5
    alpha_s1: float = 0.5

    # set pass variable
    test_pass: bool = True
    all_tests: bool = True

    # test names
    test_names = [
                  'lossless + source.p0 + homogeneous',              #0
                  'lossless + source.p0 + heterogeneous',            #1
                  'lossless + source.s (additive) + homogeneous',    #2
                  'lossless + source.s (additive) + heterogeneous',  #3
                  'lossless + source.s (dirichlet) + homogeneous',   #4
                  'lossless + source.s (dirichlet) + heterogeneous', #5
                  'lossless + source.u (additive) + homogeneous',    #6
                  'lossless + source.u (additive) + heterogeneous',  #7
                  'lossless + source.u (dirichlet) + homogeneous',   #8
                  'lossless + source.u (dirichlet) + heterogeneous', #9
                  'lossy + source.p0 + homogeneous',                 #10
                  'lossy + source.p0 + heterogeneous',               #11
                  'lossy + source.s (additive) + homogeneous',       #12
                  'lossy + source.s (additive) + heterogeneous',     #13
                  'lossy + source.s (dirichlet) + homogeneous',      #14
                  'lossy + source.s (dirichlet) + heterogeneous',    #15
                  'lossy + source.u (additive) + homogeneous',       #16
                  'lossy + source.u (additive) + heterogeneous',     #17
                  'lossy + source.u (dirichlet) + homogeneous',      #18
                  'lossy + source.u (dirichlet) + heterogeneous'     #19
                  ]

    # lists used to set properties
    p0_tests = [0, 1, 10, 11]
    s_tests  = [2, 3, 4, 5, 12, 13, 14, 15]
    u_tests  = [6, 7, 8, 9, 16, 17, 18, 19]
    # additive_tests = [2, 3, 6, 7, 12, 13, 16, 17]
    dirichlet_tests = [4, 5, 8, 9, 14, 15, 18, 19]

    # =========================================================================
    # SIMULATIONS
    # =========================================================================

    # loop through tests
    for test_num in [2,]: # np.arange(start=1, stop=2, step=1, dtype=int):
        # np.arange(1, 21, dtype=int):

        test_name = test_names[test_num]

        # update command line
        if verbose:
            print('Running Test: ', test_name)

        # assign medium properties
        medium = kWaveMedium(sound_speed=cp1,
                             density=rho1,
                             sound_speed_compression=cp1,
                             sound_speed_shear=cs1)

        # if lossy include loss terms and set flag
        if test_num > 9:
            medium.alpha_coeff_compression = alpha_p1
            medium.alpha_coeff_shear = alpha_s1

        # ----------------
        # 2D SIMULATION
        # ----------------

        # create computational grid
        kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

        # heterogeneous medium properties
        if bool(rem(test_num, 2)):
            if verbose:
                print("Set material properties [2d] as hetrogeneous: ", bool(rem(test_num, 2)), "for", test_num)
            setMaterialProperties(medium, N1=Nx, N2=Ny, N3=int(1), direction=int(1), interface_position=interface_position)
            #c_max = np.max(np.asarray([np.max(medium.sound_speed_compression), np.max(medium.sound_speed_shear)]))
        else:
            pass
            #c_max = np.max(medium.sound_speed_compression)

        # define time array
        cfl: float = 0.1
        t_end: float = 3e-6
        kgrid.dt = cfl * kgrid.dx / cp1
        kgrid.Nt = int(round(t_end / kgrid.dt))
        kgrid.t_array = np.arange(0, kgrid.Nt) * kgrid.dt

        offset: int = 1

        # define sensor mask
        sensor_mask_2D = make_circle(Vector([Nx, Ny]), Vector([Nx // 2 , Ny // 2]), 15)

        # define source properties
        source_strength: float = 3.0
        source_position_x: int = Nx // 2 - 20 - offset
        source_position_y: int = Ny // 2 - 10 - offset
        source_freq: float = 2e6
        source_signal = source_strength * np.sin(2.0 * np.pi * source_freq * kgrid.t_array)

        # sensor
        sensor = kSensor()
        sensor.record = ['u']

        # source
        source = kSource()
        if test_num in p0_tests:
            p0 = np.zeros((Nx, Ny))
            p0[source_position_x, source_position_y] = source_strength
            if SMOOTH_P0_SOURCE:
                p0 = smooth(p0, True)
            source.p0 = p0

        elif test_num in s_tests:
            source.s_mask = np.zeros((Nx, Ny), dtype=bool)
            source.s_mask[source_position_x, source_position_y] = True
            source.sxx = source_signal
            source.syy = source_signal
            if test_num in dirichlet_tests:
                source.s_mode = 'dirichlet'

        elif test_num in u_tests:
            source.u_mask = np.zeros((Nx, Ny), dtype=bool)
            source.u_mask[source_position_x, source_position_y] = True
            source.ux = source_signal / (cp1 * rho1)
            source.uy = source_signal / (cp1 * rho1)
            if test_num in dirichlet_tests:
                source.u_mode = 'dirichlet'

        else:
            raise RuntimeError('Unknown source condition.')

        # sensor mask
        sensor.mask = sensor_mask_2D

        # run the simulation
        simulation_options_2d = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                  pml_x_alpha=pml_alpha,
                                                  pml_y_alpha=pml_alpha,
                                                  pml_x_size=pml_size,
                                                  pml_y_size=pml_size,
                                                  smooth_p0=SMOOTH_P0_SOURCE,
                                                  smooth_rho0=SMOOTH_P0_SOURCE,
                                                  smooth_c0=SMOOTH_P0_SOURCE,
                                                  use_sg=USE_SG)

        sensor_data_2D = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                         source=deepcopy(source),
                                         sensor=deepcopy(sensor),
                                         medium=deepcopy(medium),
                                         simulation_options=deepcopy(simulation_options_2d))

        # calculate velocity amplitude
        sensor_data_2D['ux'] = np.reshape(sensor_data_2D['ux'], sensor_data_2D['ux'].shape, order='F')
        sensor_data_2D['uy'] = np.reshape(sensor_data_2D['uy'], sensor_data_2D['uy'].shape, order='F')
        sensor_data_2D = np.sqrt(sensor_data_2D['ux']**2 + sensor_data_2D['uy']**2)




        # # ----------------
        # # 3D SIMULATION: Z
        # # ----------------

        # del kgrid
        # del source
        # del sensor

        # # create computational grid
        # kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

        # # heterogeneous medium properties
        # if bool(rem(test_num, 2)):
        #     if verbose:
        #         print("SET MATERIALS [3D Z] AS Hetrogeneous:", bool(rem(test_num, 2)), "for", test_num)
        #     setMaterialProperties(medium, Nx, Ny, Nz, direction=1, interface_position=interface_position, cp1=cp1, cs1=cs1, rho1=rho1)
        #     c_max = np.max(np.asarray([np.max(medium.sound_speed_compression), np.max(medium.sound_speed_shear)]))
        # else:
        #     c_max = np.max(medium.sound_speed_compression)

        # # define time array
        # # cfl = 0.1
        # # t_end = 3e-6
        # # kgrid.makeTime(c_max, cfl, t_end)
        # cfl: float = 0.1
        # t_end: float = 3e-6
        # kgrid.dt = cfl * kgrid.dx / cp1
        # kgrid.Nt = int(round(t_end / kgrid.dt))
        # kgrid.t_array = np.arange(0, kgrid.Nt) * kgrid.dt


        # # source
        # source = kSource()
        # if test_num in p0_tests:
        #     p0 = np.zeros((Nx, Ny, Nz))
        #     p0[source_position_x, source_position_y, :] = source_strength
        #     if SMOOTH_P0_SOURCE:
        #         p0 = smooth(p0, True)
        #     source.p0 = p0

        # elif test_num in s_tests:
        #     source.s_mask = np.zeros((Nx, Ny, Nz))
        #     source.s_mask[source_position_x, source_position_y, :] = 1
        #     source.sxx = source_signal
        #     source.syy = source_signal
        #     source.szz = source_signal
        #     if test_num in dirichlet_tests:
        #         source.s_mode = 'dirichlet'

        # elif test_num in u_tests:
        #     source.u_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        #     source.u_mask[source_position_x, source_position_y, :] = True
        #     source.ux = source_signal / (cp1 * rho1)
        #     source.uy = source_signal / (cp1 * rho1)
        #     source.uz = source_signal / (cp1 * rho1)
        #     if test_num in dirichlet_tests:
        #         source.u_mode = 'dirichlet'

        # else:
        #     raise RuntimeError('Unknown source condition.')

        # # sensor
        # sensor = kSensor()
        # sensor.record = ['u']
        # sensor.mask = np.zeros((Nx, Ny, Nz)) #, order='F')
        # sensor.mask[:, :, Nz // 2 - 1] = sensor_mask_2D

        # # run the simulation
        # simulation_options_3d = SimulationOptions(simulation_type=SimulationType.ELASTIC,
        #                                           pml_x_size=pml_size,
        #                                           pml_y_size=pml_size,
        #                                           pml_z_size=pml_size,
        #                                           pml_x_alpha=pml_alpha,
        #                                           pml_y_alpha=pml_alpha,
        #                                           pml_z_alpha=0.0,
        #                                           smooth_p0=SMOOTH_P0_SOURCE, smooth_c0=SMOOTH_P0_SOURCE, smooth_rho0=SMOOTH_P0_SOURCE)

        # sensor_data_3D_z = pstd_elastic_3d(kgrid=deepcopy(kgrid),
        #                                    source=deepcopy(source),
        #                                    sensor=deepcopy(sensor),
        #                                    medium=deepcopy(medium),
        #                                    simulation_options=deepcopy(simulation_options_3d))

        # if verbose:
        #     print(np.shape(sensor_data_3D_z['ux']), np.shape(sensor_data_3D_z['uy']), pml_size, Nx, kgrid.Nx, Ny, kgrid.Ny, kgrid.Nt)

        # # calculate velocity amplitude
        # sensor_data_3D_z['ux'] = np.reshape(sensor_data_3D_z['ux'], sensor_data_3D_z['ux'].shape, order='F')
        # sensor_data_3D_z['uy'] = np.reshape(sensor_data_3D_z['uy'], sensor_data_3D_z['uy'].shape, order='F')
        # sensor_data_3D_z = np.sqrt(sensor_data_3D_z['ux']**2 + sensor_data_3D_z['uy']**2)

        # # ----------------
        # # 3D SIMULATION: Y
        # # ----------------

        # del kgrid
        # del source
        # del sensor

        # # create computational grid
        # kgrid = kWaveGrid(Vector([Nx, Nz, Ny]), Vector([dx, dz, dy]))

        # # heterogeneous medium properties
        # if bool(rem(test_num, 2)):
        #     if verbose:
        #         print("SET MATERIALS [3D Y] AS Hetrogeneous:", bool(rem(test_num, 2)), "for", test_num)
        #     setMaterialProperties(medium, Nx, Nz, Ny, direction=1, interface_position=interface_position, cp1=cp1, cs1=cs1, rho1=rho1)
        #     c_max = np.max(np.asarray([np.max(medium.sound_speed_compression), np.max(medium.sound_speed_shear)]))
        # else:
        #     c_max = np.max(medium.sound_speed_compression)

        # # define time array
        # # cfl = 0.1
        # # t_end = 3e-6
        # # kgrid.makeTime(c_max, cfl, t_end)
        # cfl: float = 0.1
        # t_end: float = 3e-6
        # kgrid.dt = cfl * kgrid.dx / cp1
        # kgrid.Nt = int(round(t_end / kgrid.dt))
        # kgrid.t_array = np.arange(0, kgrid.Nt) * kgrid.dt

        # # source
        # source = kSource()
        # if test_num in p0_tests:
        #     p0 = np.zeros((Nx, Nz, Ny))
        #     p0[source_position_x, :, source_position_y] = source_strength
        #     if SMOOTH_P0_SOURCE:
        #         p0 = smooth(p0, True)
        #     source.p0 = p0

        # elif test_num in s_tests:
        #     source.s_mask = np.zeros((Nx, Nz, Ny))
        #     source.s_mask[source_position_x, :, source_position_y] = 1
        #     source.sxx = source_signal
        #     source.syy = source_signal
        #     source.szz = source_signal
        #     if test_num in dirichlet_tests:
        #         source.s_mode = 'dirichlet'

        # elif test_num in u_tests:
        #     source.u_mask = np.zeros((Nx, Nz, Ny))
        #     source.u_mask[source_position_x, :, source_position_y] = 1
        #     source.ux = source_signal / (cp1 * rho1)
        #     source.uy = source_signal / (cp1 * rho1)
        #     source.uz = source_signal / (cp1 * rho1)
        #     if test_num in dirichlet_tests:
        #         source.u_mode = 'dirichlet'

        # else:
        #     raise RuntimeError('Unknown source condition.')

        # # sensor
        # sensor = kSensor()
        # sensor.record = ['u']
        # sensor.mask = np.zeros((Nx, Nz, Ny)) #, order='F')
        # sensor.mask[:, Nz // 2 - 1, :] = sensor_mask_2D

        # # run the simulation
        # simulation_options_3d = SimulationOptions(simulation_type=SimulationType.ELASTIC,
        #                                           pml_size=pml_size,
        #                                           pml_x_alpha=pml_alpha,
        #                                           pml_y_alpha=0.0,
        #                                           pml_z_alpha=pml_alpha,
        #                                           smooth_p0=SMOOTH_P0_SOURCE, smooth_c0=SMOOTH_P0_SOURCE, smooth_rho0=SMOOTH_P0_SOURCE)

        # sensor_data_3D_y = pstd_elastic_3d(kgrid=deepcopy(kgrid),
        #                                    source=deepcopy(source),
        #                                    sensor=deepcopy(sensor),
        #                                    medium=deepcopy(medium),
        #                                    simulation_options=deepcopy(simulation_options_3d))

        # # calculate velocity amplitude
        # sensor_data_3D_y['ux'] = np.reshape(sensor_data_3D_y['ux'], sensor_data_3D_y['ux'].shape, order='F')
        # sensor_data_3D_y['uz'] = np.reshape(sensor_data_3D_y['uz'], sensor_data_3D_y['uz'].shape, order='F')
        # sensor_data_3D_y = np.sqrt(sensor_data_3D_y['ux']**2 + sensor_data_3D_y['uz']**2)

        # # ----------------
        # # 3D SIMULATION: X
        # # ----------------

        # del kgrid
        # del source
        # del sensor

        # # create computational grid
        # kgrid = kWaveGrid(Vector([Nz, Nx, Ny]), Vector([dz, dx, dy]))

        # # heterogeneous medium properties
        # if bool(rem(test_num, 2)):
        #     if verbose:
        #         print("SET MATERIALS [3D X] AS Hetrogeneous:", bool(rem(test_num, 2)), "for", test_num)
        #     setMaterialProperties(medium, Nz, Nx, Ny, direction=2, interface_position=interface_position, cp1=cp1, cs1=cs1, rho1=rho1)
        #     c_max = np.max(np.asarray([np.max(medium.sound_speed_compression), np.max(medium.sound_speed_shear)]))
        # else:
        #     c_max = np.max(medium.sound_speed_compression)

        # # define time array
        # # cfl = 0.1
        # # t_end = 3e-6
        # # kgrid.makeTime(c_max, cfl, t_end)
        # cfl: float = 0.1
        # t_end: float = 3e-6
        # kgrid.dt = cfl * kgrid.dx / cp1
        # kgrid.Nt = int(round(t_end / kgrid.dt))
        # kgrid.t_array = np.arange(0, kgrid.Nt) * kgrid.dt


        # # source
        # source = kSource()
        # if test_num in p0_tests:
        #     p0 = np.zeros((Nz, Nx, Ny))
        #     p0[:, source_position_x, source_position_y] = source_strength
        #     if SMOOTH_P0_SOURCE:
        #         p0 = smooth(p0, True)
        #     source.p0 = p0

        # elif test_num in s_tests:
        #     source.s_mask = np.zeros((Nz, Nx, Ny))
        #     source.s_mask[:, source_position_x, source_position_y] = 1
        #     source.sxx = source_signal
        #     source.syy = source_signal
        #     source.szz = source_signal
        #     if test_num in dirichlet_tests:
        #         source.s_mode = 'dirichlet'

        # elif test_num in u_tests:
        #     source.u_mask = np.zeros((Nz, Nx, Ny))
        #     source.u_mask[:, source_position_x, source_position_y] = 1
        #     source.ux = source_signal / (cp1 * rho1)
        #     source.uy = source_signal / (cp1 * rho1)
        #     source.uz = source_signal / (cp1 * rho1)
        #     if test_num in dirichlet_tests:
        #         source.u_mode = 'dirichlet'

        # else:
        #     raise RuntimeError('Unknown source condition.')

        # # sensor
        # sensor = kSensor()
        # sensor.record = ['u']
        # sensor.mask = np.zeros((Nz, Nx, Ny)) #, order='F')
        # sensor.mask[Nz // 2 - 1, :, :] = sensor_mask_2D

        # # run the simulation
        # simulation_options_3d = SimulationOptions(simulation_type=SimulationType.ELASTIC,
        #                                           pml_size=pml_size,
        #                                           pml_x_alpha=0.0,
        #                                           pml_y_alpha=pml_alpha,
        #                                           pml_z_alpha=pml_alpha,
        #                                           smooth_p0=SMOOTH_P0_SOURCE, smooth_c0=SMOOTH_P0_SOURCE, smooth_rho0=SMOOTH_P0_SOURCE)

        # sensor_data_3D_x = pstd_elastic_3d(kgrid=deepcopy(kgrid),
        #                                    source=deepcopy(source),
        #                                    sensor=deepcopy(sensor),
        #                                    medium=deepcopy(medium),
        #                                    simulation_options=deepcopy(simulation_options_3d))

        # # calculate velocity amplitude
        # sensor_data_3D_x['uy'] = np.reshape(sensor_data_3D_x['uy'], sensor_data_3D_x['uy'].shape, order='F')
        # sensor_data_3D_x['uz'] = np.reshape(sensor_data_3D_x['uz'], sensor_data_3D_x['uz'].shape, order='F')
        # sensor_data_3D_x = np.sqrt(sensor_data_3D_x['uy']**2 + sensor_data_3D_x['uz']**2)

        # -------------
        # COMPARISON
        # -------------

        if (test_num == 0):
            matlab_test = loadmat("C:/Users/dsinden/dev/octave/sensor2D.mat")
            matlab_2d = matlab_test['sensor_2d']

        if (test_num == 0):
            print(np.unravel_index(np.argmax(np.abs(matlab_2d)), matlab_2d.shape, order='F'),
                  np.unravel_index(np.argmax(np.abs(sensor_data_2D)), sensor_data_2D.shape, order='F'),
                  # np.unravel_index(np.argmax(np.abs(sensor_data_3D_z)), sensor_data_3D_z.shape, order='F'),
                  # np.unravel_index(np.argmax(np.abs(sensor_data_3D_y)), sensor_data_3D_y.shape, order='F'),
                  # np.unravel_index(np.argmax(np.abs(sensor_data_3D_x)), sensor_data_3D_x.shape, order='F'),
                  )
        else:
            print(np.unravel_index(np.argmax(np.abs(sensor_data_2D)), sensor_data_2D.shape, order='F'),
                  # np.unravel_index(np.argmax(np.abs(sensor_data_3D_z)), sensor_data_3D_z.shape, order='F'),
                  # np.unravel_index(np.argmax(np.abs(sensor_data_3D_y)), sensor_data_3D_y.shape, order='F'),
                  # np.unravel_index(np.argmax(np.abs(sensor_data_3D_x)), sensor_data_3D_x.shape, order='F'),
                  )

        sensor_data_3D_z = np.zeros_like(sensor_data_2D)
        sensor_data_3D_y = np.zeros_like(sensor_data_2D)
        sensor_data_3D_x = np.zeros_like(sensor_data_2D)

        max2d = np.max(np.abs(sensor_data_2D))
        max3d_z = np.max(np.abs(sensor_data_3D_z))
        max3d_y = np.max(np.abs(sensor_data_3D_y))
        max3d_x = np.max(np.abs(sensor_data_3D_x))

        diff_2D_3D_z = np.max(np.abs(sensor_data_2D - sensor_data_3D_z)) / max2d
        if diff_2D_3D_z > COMPARISON_THRESH:
            test_pass = False
            msg = f"Not equal: diff_2D_3D_z: {diff_2D_3D_z} and 2d: {max2d}, 3d: {max3d_z}"
            print(msg)
        all_tests = all_tests and test_pass

        diff_2D_3D_y = np.max(np.abs(sensor_data_2D - sensor_data_3D_y)) / max2d
        if diff_2D_3D_y > COMPARISON_THRESH:
            test_pass = False
            msg = f"Not equal: diff_2D_3D_y: {diff_2D_3D_y} and 2d: {max2d}, 3d: {max3d_y}"
            print(msg)
        all_tests = all_tests and test_pass

        diff_2D_3D_x = np.max(np.abs(sensor_data_2D - sensor_data_3D_x)) / max2d
        if diff_2D_3D_x > COMPARISON_THRESH:
            test_pass = False
            msg = f"Not equal: diff_2D_3D_x: {diff_2D_3D_x} and 2d: {max2d}, 3d: {max3d_x}"
            print(msg)
        all_tests = all_tests and test_pass

        # if (test_num == 0):
        #     fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2)
        #     fig3.suptitle(f"{test_name}: Z")
        #     ax3a.imshow(sensor_data_2D)
        #     # ax3b.imshow(sensor_data_3D_z)
        #     # ax3c.imshow(np.abs(sensor_data_2D - sensor_data_3D_z))
        #     ax3d.imshow(np.abs(matlab_2d))
        # else:
        #     fig3, (ax3a, ax3b, ax3c) = plt.subplots(3, 1)
        #     fig3.suptitle(f"{test_name}: Z")
        #     ax3a.imshow(sensor_data_2D)
        #     # ax3b.imshow(sensor_data_3D_z)
        #     # ax3c.imshow(np.abs(sensor_data_2D - sensor_data_3D_z))

        # fig2, ((ax2a, ax2b, ax2c) ) = plt.subplots(3, 1)
        # fig2.suptitle(f"{test_name}: Y")
        # ax2a.imshow(sensor_data_2D)
        # # ax2b.imshow(sensor_data_3D_y)
        # # ax2c.imshow(np.abs(sensor_data_2D - sensor_data_3D_y))

        # fig1, ((ax1a, ax1b, ax1c) ) = plt.subplots(3, 1)
        # fig1.suptitle(f"{test_name}: X")
        # ax1a.imshow(sensor_data_2D)
        # ax1b.imshow(sensor_data_3D_x)
        # ax1c.imshow(np.abs(sensor_data_2D - sensor_data_3D_x))


        fig0, ax0a = plt.subplots(1, 1)
        ax0a.plot(np.squeeze(kgrid.t_array), sensor_data_2D[Nx // 2 - 1, :], label='2D')
        # ax0a.plot(np.squeeze(kgrid.t_array), sensor_data_3D_x[Nx // 2 - 1, :], label='3D x')
        # ax0a.plot(np.squeeze(kgrid.t_array), sensor_data_3D_y[Nx // 2 - 1, :], label='3D y')
        # ax0a.plot(np.squeeze(kgrid.t_array), sensor_data_3D_z[Nx // 2 - 1, :], label='3D z')
        if (test_num == 0):
            ax0a.plot(np.squeeze(kgrid.t_array), matlab_2d[Nx // 2 - 1, :], 'k-', label='matlab')
        ax0a.legend()
        # ax0b.plot(np.squeeze(kgrid.t_array), sensor_data_2D[:, Ny // 2 - 1], label='2D')
        # ax0b.plot(np.squeeze(kgrid.t_array), sensor_data_3D_x[:, Ny // 2 - 1], label='3D x')
        # ax0b.plot(np.squeeze(kgrid.t_array), sensor_data_3D_y[:, Ny // 2 - 1], label='3D y')
        # ax0b.plot(np.squeeze(kgrid.t_array), sensor_data_3D_z[:, Ny // 2 - 1], label='3D z')
        # ax0b.legend()

        plt.show()

        # clear structures
        del kgrid
        del source
        del medium
        del sensor

    assert all_tests, msg

        # diff_2D_3D_x = np.max(np.abs(sensor_data_2D - sensor_data_3D_x)) / ref_max
        # if diff_2D_3D_x > COMPARISON_THRESH:
        #     test_pass = False
        #     assert test_pass, "Not equal: dff_2D_3D_x"

        # diff_2D_3D_y = np.max(np.abs(sensor_data_2D - sensor_data_3D_y)) / ref_max
        # if diff_2D_3D_y > COMPARISON_THRESH:
        #     test_pass = False
        #     assert test_pass, "Not equal: diff_2D_3D_y"





