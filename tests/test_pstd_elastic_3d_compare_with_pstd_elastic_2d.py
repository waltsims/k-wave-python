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



def setMaterialProperties(medium: kWaveMedium, N1:int, N2:int, N3:int, direction=int,
                          cp1: float=1500.0, cs1: float=0.0, rho1: float=1000.0,
                          alpha_p1: float=0.5, alpha_s1: float=0.5):

    # sound speed and density
    medium.sound_speed_compression = cp1 * np.ones((N1, N2, N3))
    medium.sound_speed_shear = cs1 * np.ones((N1, N2, N3))
    medium.density = rho1 * np.ones((N1, N2, N3))

    cp2         = 2000.0
    cs2         = 800.0
    rho2        = 1200.0
    alpha_p2    = 1.0
    alpha_s2    = 1.0

    # position of the heterogeneous interface
    interface_position: int = N1 // 2

    if direction == 1:
            medium.sound_speed_compression[interface_position:, :, :] = cp2
            medium.sound_speed_shear[interface_position:, :, :] = cs2
            medium.density[interface_position:, :, :] = rho2
    elif direction == 2:
            medium.sound_speed_compression[:, interface_position:, :] = cp2
            medium.sound_speed_shear[:, interface_position:, :] = cs2
            medium.density[:, interface_position:, :] = rho2

    medium.sound_speed_compression = np.squeeze(medium.sound_speed_compression)
    medium.sound_speed_shear = np.squeeze(medium.sound_speed_shear)
    medium.density = np.squeeze(medium.density)

    # absorption
    if hasattr(medium, 'alpha_coeff_compression'):
        if medium.alpha_coeff_compression is not None:
            medium.alpha_coeff_compression = alpha_p1 * np.ones((N1, N2, N3))
            medium.alpha_coeff_shear = alpha_s1 * np.ones((N1, N2, N3))
            if direction == 1:
                medium.alpha_coeff_compression[interface_position:, :, :] = alpha_p2
                medium.alpha_coeff_shear[interface_position:, :, :] = alpha_s2
            elif direction == 2:
                medium.alpha_coeff_compression[:, interface_position:, :] = alpha_p2
                medium.alpha_coeff_shear[:, interface_position:, :] = alpha_s2

    medium.alpha_coeff_compression = np.squeeze(medium.alpha_coeff_compression)
    medium.alpha_coeff_shear = np.squeeze(medium.alpha_coeff_shear)

@pytest.mark.skip(reason="not ready")
def test_pstd_elastic_3d_compare_with_pstd_elastic_2d():

    # set additional literals to give further permutations of the test
    USE_PML             = False
    COMPARISON_THRESH   = 1e-14
    SMOOTH_P0_SOURCE    = False

    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================

    # define grid size
    Nx: int = 64
    Ny: int = 64
    Nz: int = 32
    dx = 0.1e-3
    dy = dx
    dz = dx

    # define PML properties
    PML_size: int = 10
    if USE_PML:
        PML_alpha = 2
    else:
        PML_alpha = 0

    # define material properties
    cp1         = 1500.0
    cs1         = 0.0
    rho1        = 1000.0
    alpha_p1    = 0.5
    alpha_s1    = 0.5

    # define time array
    cfl     = 0.1
    t_end   = 3e-6
    dt      = cfl * dx / cp1
    Nt: int = int(round(t_end / dt))

    #t_array = 0:dt:(Nt - 1) * dt
    t_array = np.linspace(0, (Nt - 1) * dt, Nt)

    # define sensor mask
    sensor_mask_2D = make_circle(Vector([Nx, Ny]), Vector([Nx // 2 - 1, Ny// 2 - 1]), 15)

    # define input arguements
    # input_args = {'PlotScale', [-1, 1, -0.2, 0.2], 'PMLSize', PML_size,
    # 'UseSG', USE_SG, 'Smooth', false, 'PlotSim', plot_simulations};

    # define source properties
    source_strength = 3
    source_position_x: int = Nx // 2 - 21
    source_position_y: int = Ny // 2 - 11
    source_freq = 2e6
    source_signal = source_strength * np.sin(2.0 * np.pi * source_freq * t_array)

    # set pass variable
    test_pass = True

    # test names
    test_names = ['lossless + source.p0 + homogeneous',
        'lossless + source.p0 + heterogeneous',
        'lossless + source.s (additive) + homogeneous',
        'lossless + source.s (additive) + heterogeneous',
        'lossless + source.s (dirichlet) + homogeneous',
        'lossless + source.s (dirichlet) + heterogeneous',
        'lossless + source.u (additive) + homogeneous',
        'lossless + source.u (additive) + heterogeneous',
        'lossless + source.u (dirichlet) + homogeneous',
        'lossless + source.u (dirichlet) + heterogeneous',
        'lossy + source.p0 + homogeneous',
        'lossy + source.p0 + heterogeneous',
        'lossy + source.s (additive) + homogeneous',
        'lossy + source.s (additive) + heterogeneous',
        'lossy + source.s (dirichlet) + homogeneous',
        'lossy + source.s (dirichlet) + heterogeneous',
        'lossy + source.u (additive) + homogeneous',
        'lossy + source.u (additive) + heterogeneous',
        'lossy + source.u (dirichlet) + homogeneous',
        'lossy + source.u (dirichlet) + heterogeneous']

    # lists used to set properties
    p0_tests = [1, 2, 11, 12]
    s_tests  = [3, 4, 5, 6, 13, 14, 15, 16]
    u_tests  = [7, 8, 9, 10, 17, 18, 19, 20]
    dirichlet_tests = [5, 6, 9, 10, 15, 16, 19, 20]

    # =========================================================================
    # SIMULATIONS
    # =========================================================================

    # loop through tests
    for test_num in np.arange(1, 21, dtype=int):

        # update command line
        print('Running Test: ', test_names[test_num])

        # assign medium properties
        medium = kWaveMedium(sound_speed=cp1,
                             density=rho1,
                             sound_speed_compression=cp1,
                             sound_speed_shear=cs1)
        if test_num > 10:
            medium.alpha_coeff_compression = alpha_p1
            medium.alpha_coeff_shear  = alpha_s1
            kelvin_voigt = True
        else:
            kelvin_voigt = False

        # ----------------
        # 2D SIMULATION
        # ----------------

        # create computational grid
        kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))
        kgrid.t_array = t_array

        # heterogeneous medium properties
        if not bool(rem(test_num, 2)):
            setMaterialProperties(medium, Nx, Ny, N3=int(1), direction=1)

        # sensor
        sensor = kSensor()
        sensor.record = ['u']

        # source
        source = kSource()
        if any(p0_tests == test_num):
            p0 = np.zeros((Nx, Ny))
            p0[source_position_x, source_position_y] = source_strength
            if SMOOTH_P0_SOURCE:
                p0 = smooth(source.p0, True)
            source.p0 = p0

        elif any(s_tests == test_num):
            source.s_mask = np.zeros((Nx, Ny))
            source.s_mask[source_position_x, source_position_y] = 1
            source.sxx = source_signal
            source.syy = source_signal
            if any(dirichlet_tests == test_num):
                source.s_mode = 'dirichlet'

        elif any(u_tests == test_num):
            source.u_mask = np.zeros((Nx, Ny))
            source.u_mask[source_position_x, source_position_y] = 1
            source.ux = source_signal / (cp1 * rho1)
            source.uy = source_signal / (cp1 * rho1)
            if any(dirichlet_tests == test_num):
                source.u_mode = 'dirichlet'

        else:
            raise RuntimeError('Unknown source condition.')

        # sensor mask
        sensor.mask = sensor_mask_2D

        # run the simulation
        simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                               kelvin_voigt_model=kelvin_voigt,
                                               pml_alpha=PML_alpha,
                                               pml_size=PML_size,
                                               smooth_p0=False)

        sensor_data_2D = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                         source=deepcopy(source),
                                         sensor=deepcopy(sensor),
                                         medium=deepcopy(medium),
                                         simulation_options=deepcopy(simulation_options))

        # calculate velocity amplitude
        sensor_data_2D = np.sqrt(sensor_data_2D['ux']**2 + sensor_data_2D['uy']**2)

        # ----------------
        # 3D SIMULATION: Z
        # ----------------

        del kgrid
        del source
        del sensor

        # create computational grid
        kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))
        kgrid.t_array = t_array

        # heterogeneous medium properties
        if not bool(rem(test_num, 2)):
            setMaterialProperties(medium, Nx, Ny, Nz, direction=1, cp1=cp1, cs1=cs1, rho=rho1)

        # source
        source = kSource()
        if any(p0_tests == test_num):
            p0 = np.zeros((Nx, Ny, Nz))
            p0[source_position_x, source_position_y, :] = source_strength
            if SMOOTH_P0_SOURCE:
                p0 = smooth(p0, True)
            source.p0 = p0

        elif any(s_tests == test_num):
            source.s_mask = np.zeros((Nx, Ny, Nz))
            source.s_mask[source_position_x, source_position_y, :] = 1
            source.sxx = source_signal
            source.syy = source_signal
            source.szz = source_signal
            if any(dirichlet_tests == test_num):
                source.s_mode = 'dirichlet'

        elif any(u_tests == test_num):
            source.u_mask = np.zeros((Nx, Ny, Nz))
            source.u_mask[source_position_x, source_position_y, :] = 1
            source.ux = source_signal / (cp1 * rho1)
            source.uy = source_signal / (cp1 * rho1)
            source.uz = source_signal / (cp1 * rho1)
            if any(dirichlet_tests == test_num):
                source.u_mode = 'dirichlet'

        else:
            raise RuntimeError('Unknown source condition.')

        # sensor
        sensor = kSensor()
        sensor.record = ['u']
        sensor.mask = np.zeros((Nx, Ny, Nz))
        sensor.mask[:, :, Nz // 2 - 1] = sensor_mask_2D

        # run the simulation
        simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                               kelvin_voigt_model=kelvin_voigt,
                                               pml_x_alpha=PML_alpha,
                                               pml_y_alpha=PML_alpha,
                                               pml_z_alpha=0.0)

        sensor_data_3D_z = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                           source=deepcopy(source),
                                           sensor=deepcopy(sensor),
                                           medium=deepcopy(medium),
                                           simulation_options=deepcopy(simulation_options))

        # calculate velocity amplitude
        sensor_data_3D_z = np.sqrt(sensor_data_3D_z['ux']**2 + sensor_data_3D_z['uy']**2)

    #     # ----------------
    #     # 3D SIMULATION: Y
    #     # ----------------

    #     # create computational grid
    #     kgrid = kWaveGrid(Nx, dx, Nz, dz, Ny, dy);
    #     kgrid.t_array = t_array;

    #     # heterogeneous medium properties
    #     if not bool(rem(test_num, 2)):
    #         setMaterialProperties(Nx, Nz, Ny, 1)
              # setMaterialProperties(medium, Nx, Nz, Ny, direction=1, cp1=cp1, cs2=cs2, rho=rho1)
    #     end

    #     # source
    #     if any(p0_tests == test_num)
    #         source.p0 = zeros(Nx, Nz, Ny);
    #         source.p0(source_position_x, :, source_position_y) = source_strength;
    #         if SMOOTH_P0_SOURCE
    #             source.p0 = smooth(source.p0, true);
    #         end
    #     elseif any(s_tests == test_num)
    #         source.s_mask = zeros(Nx, Nz, Ny);
    #         source.s_mask(source_position_x, :, source_position_y) = 1;
    #         source.sxx = source_signal;
    #         source.syy = source_signal;
    #         source.szz = source_signal;
    #         if any(dirichlet_tests == test_num)
    #             source.s_mode = 'dirichlet';
    #         end
    #     elseif any(u_tests == test_num)
    #         source.u_mask = zeros(Nx, Nz, Ny);
    #         source.u_mask(source_position_x, :, source_position_y) = 1;
    #         source.ux = source_signal ./ (cp1 * rho1);
    #         source.uy = source_signal ./ (cp1 * rho1);
    #         source.uz = source_signal ./ (cp1 * rho1);
    #         if any(dirichlet_tests == test_num)
    #             source.u_mode = 'dirichlet';
    #         end
    #     else
    #         error('Unknown source condition.');
    #     end

    #     # sensor
    #     sensor.mask = zeros(Nx, Nz, Ny);
    #     sensor.mask(:, Nz/2, :) = sensor_mask_2D;

    #     # run the simulation
    #     sensor_data_3D_y = pstdElastic3D(kgrid, medium, source, sensor, ...
    #         input_args{:}, 'PMLAlpha', [PML_alpha, 0, PML_alpha]);

    #     # calculate velocity amplitude
    #     sensor_data_3D_y = sqrt(sensor_data_3D_y.ux.^2 + sensor_data_3D_y.uz.^2);

    #     # ----------------
    #     # 3D SIMULATION: X
    #     # ----------------

    #     # create computational grid
    #     kgrid = kWaveGrid(Nz, dz, Nx, dx, Ny, dy);
    #     kgrid.t_array = t_array;

    #     # heterogeneous medium properties
    #     if not bool(rem(test_num, 2)):
    #         setMaterialProperties(Nz, Nx, Ny, 2)
    #         setMaterialProperties(medium, Nz, Nx, Ny, direction=2, cp1=cp1, cs2=cs2, rho=rho1)
    #     end

    #     # source
    #     if any(p0_tests == test_num)
    #         source.p0 = zeros(Nz, Nx, Ny);
    #         source.p0(:, source_position_x, source_position_y) = source_strength;
    #         if SMOOTH_P0_SOURCE
    #             source.p0 = smooth(source.p0, true);
    #         end
    #     elseif any(s_tests == test_num)
    #         source.s_mask = zeros(Nz, Nx, Ny);
    #         source.s_mask(:, source_position_x, source_position_y) = 1;
    #         source.sxx = source_signal;
    #         source.syy = source_signal;
    #         source.szz = source_signal;
    #         if any(dirichlet_tests == test_num)
    #             source.s_mode = 'dirichlet';
    #         end
    #     elseif any(u_tests == test_num)
    #         source.u_mask = zeros(Nz, Nx, Ny);
    #         source.u_mask(:, source_position_x, source_position_y) = 1;
    #         source.ux = source_signal ./ (cp1 * rho1);
    #         source.uy = source_signal ./ (cp1 * rho1);
    #         source.uz = source_signal ./ (cp1 * rho1);
    #         if any(dirichlet_tests == test_num)
    #             source.u_mode = 'dirichlet';
    #         end
    #     else
    #         error('Unknown source condition.');
    #     end

    #     # sensor
    #     sensor.mask = zeros(Nz, Nx, Ny);
    #     sensor.mask(Nz/2, :, :) = sensor_mask_2D;

    #     # run the simulation
    #     sensor_data_3D_x = pstdElastic3D(kgrid, medium, source, sensor, ...
    #         input_args{:}, 'PMLAlpha', [0, PML_alpha, PML_alpha]);

    #     # calculate velocity amplitude
    #     sensor_data_3D_x = sqrt(sensor_data_3D_x.uy.^2 + sensor_data_3D_x.uz.^2);

        # -------------
        # COMPARISON
        # -------------

        ref_max = np.max(np.abs(sensor_data_2D))

        diff_2D_3D_z = np.max(np.abs(sensor_data_2D - sensor_data_3D_z)) / ref_max
        if diff_2D_3D_z > COMPARISON_THRESH:
            test_pass = False
            assert test_pass, "Not equal: diff_2D_3D_z"

        # diff_2D_3D_x = np.max(np.abs(sensor_data_2D - sensor_data_3D_x)) / ref_max
        # if diff_2D_3D_x > COMPARISON_THRESH:
        #     test_pass = False
        #     assert test_pass, "Not equal: dff_2D_3D_x"

        # diff_2D_3D_y = np.max(np.abs(sensor_data_2D - sensor_data_3D_y)) / ref_max
        # if diff_2D_3D_y > COMPARISON_THRESH:
        #     test_pass = False
        #     assert test_pass, "Not equal: diff_2D_3D_y"

        # clear structures
        del kgrid
        del source
        del medium
        del sensor



