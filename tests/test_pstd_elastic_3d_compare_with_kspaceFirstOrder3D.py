"""
Unit test to compare that the elastic code with the shear wave speed set to
zero gives the same answers as the regular fluid code in k-Wave.
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
#import pytest


from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic3D import pstd_elastic_3d
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_ball

#@pytest.mark.skip(reason="not ready")
def test_pstd_elastic_3D_compare_with_kspaceFirstOrder3D():

    # set additional literals to give further permutations of the test
    HETEROGENEOUS: bool = True
    USE_PML: bool = True
    DATA_CAST: str = 'on'
    COMPARISON_THRESH: float = 5e-10

    # option to skip the first point in the time series (for p0 sources, there
    # is a strange bug where there is a high error for the first stored time
    # point)
    COMP_START_INDEX: int = 1

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx: int = 64
    Ny: int = 62
    Nz: int = 60
    dx: float = 0.1e-3
    dy: float = 0.1e-3
    dz: float = 0.1e-3
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the medium properties
    cp: float = 1500.0
    cs: float = 0.0
    rho: float = 1000.0

    # create the time zarray
    CFL: float = 0.1
    t_end: float = 3e-6
    kgrid.makeTime(cp, CFL, t_end)

    # create and assign the variables
    if HETEROGENEOUS:
        # elastic medium
        sound_speed_compression = cp * np.ones((Nx, Ny, Nz))
        sound_speed_compression[Nx // 2 - 1:, :, :] = 2.0 * cp
        sound_speed_shear = cs * np.ones((Nx, Ny, Nz))
        density = rho * np.ones((Nx, Ny, Nz))
        medium_elastic = kWaveMedium(sound_speed=sound_speed_compression,
                                     density=density,
                                     sound_speed_compression=sound_speed_compression,
                                     sound_speed_shear=sound_speed_shear)
        # fluid medium
        sound_speed = cp * np.ones((Nx, Ny, Nz))
        sound_speed[Nx // 2 - 1:, :, :] = 2.0 * cp
        density = rho * np.ones((Nx, Ny, Nz))
        medium_fluid = kWaveMedium(sound_speed, density=density)
    else:
        # elastic medium
        medium_elastic = kWaveMedium(cp,
                                     density=rho,
                                     sound_speed_compression=cp,
                                     sound_speed_shear=cs)
        # fluid medium
        medium_fluid = kWaveMedium(sound_speed=cp,
                                   density=rho)


    # set pass variable
    test_pass: bool = True

    # test names
    test_names = ['source.p0',
        'source.p, additive',
        'source.p, dirichlet', # gives warning
        'source.ux, additive',
        #'source.ux, dirichlet',
        #'source.uy, additive',
        #'source.uy, dirichlet',
        #'source.uz, additive',
        #'source.uz, dirichlet'
        ]

    # define a single point sensor
    sensor_elastic = kSensor()
    sensor_fluid = kSensor()
    sensor_elastic.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_fluid.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_elastic.mask[3 * Nx // 4 - 1, 3 * Ny // 4 - 1, 3 * Nz // 4 - 1] = True
    sensor_fluid.mask[3 * Nx // 4 - 1, 3 * Ny // 4 - 1, 3 * Nz // 4 - 1] = True

    # set some things to record
    sensor_elastic.record = ['p', 'p_final', 'u', 'u_final']
    sensor_fluid.record = ['p', 'p_final', 'u', 'u_final']


    # loop through tests
    for test_num, test_name in enumerate(test_names):

        # update command line
        print('Running Number: ', test_num, ':', test_name)

        # set up sources
        source_fluid = kSource()
        source_elastic = kSource()

        if test_name == 'source.p0':
            # create initial pressure distribution using makeBall
            disc_magnitude: float = 5.0     # [Pa]
            disc_x_pos: int = Nx // 2 - 11  # [grid points]
            disc_y_pos: int = Ny // 2 - 1   # [grid points]
            disx_z_pos: int = Nz // 2 - 1   # [grid points]
            disc_radius: int = 3            # [grid points]
            source_fluid.p0 = disc_magnitude * make_ball(Vector([Nx, Ny, Nz]),
                                                         Vector([disc_x_pos, disc_y_pos, disx_z_pos]),
                                                         disc_radius)
            disc_magnitude: float = 5.0      # [Pa]
            disc_x_pos: int = Nx // 2 - 11   # [grid points]
            disc_y_pos: int = Ny // 2  - 1   # [grid points]
            disx_z_pos: int = Nz // 2  - 1   # [grid points]
            disc_radius: int = 3             # [grid points]
            source_elastic.p0 = disc_magnitude * make_ball(Vector([Nx, Ny, Nz]),
                                                           Vector([disc_x_pos, disc_y_pos, disx_z_pos]),
                                                           disc_radius)

        elif test_name == 'source.p, additive' or test_name == 'source.p, dirichlet':
            # create pressure source
            source_fluid.p_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.p_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.p = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array))
            source_fluid.p = filter_time_series(deepcopy(kgrid),
                                                deepcopy(medium_fluid),
                                                deepcopy(source_fluid.p))
            # create equivalent elastic source
            source_elastic.s_mask = deepcopy(source_fluid.p_mask)
            source_elastic.sxx = deepcopy(-source_fluid.p)
            source_elastic.syy = deepcopy(-source_fluid.p)
            source_elastic.szz = deepcopy(-source_fluid.p)

        elif test_name == 'source.ux, additive' or test_name == 'source.ux, dirichlet':
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.ux = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.ux = filter_time_series(kgrid, medium_fluid, deepcopy(source_fluid.ux))
            # create equivalent elastic source
            source_elastic.u_mask = np.zeros((Nx, Ny, Nz))
            source_elastic.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_elastic.ux = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_elastic.ux = filter_time_series(kgrid, medium_fluid, deepcopy(source_fluid.ux))
            # source_elastic = deepcopy(source_fluid)

        elif test_name == 'source.uy, additive' or test_name == 'source.uy, dirichlet':
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.uy = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.uy = filter_time_series(kgrid, medium_fluid, deepcopy(source_fluid.uy))
            # create equivalent elastic source
            source_elastic = deepcopy(source_fluid)

        elif test_name == 'source.uz, additive' or test_name == 'source.uz, dirichlet':
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.uz = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.uz = filter_time_series(kgrid, medium_fluid, deepcopy(source_fluid.uz))
            # create equivalent elastic source
            source_elastic = deepcopy(source_fluid)

        # set source mode
        if test_name == 'source.p, additive':
            source_fluid.p_mode   = 'additive'
            source_elastic.s_mode = 'additive'
        elif test_name == 'source.p, dirichlet':
            source_fluid.p_mode   = 'dirichlet'
            source_elastic.s_mode = 'dirichlet'
        elif test_name == 'source.ux, additive' or test_name == 'source.uy, additive' or test_name == 'source.uz, additive':
            source_fluid.u_mode   = 'additive'
            source_elastic.u_mode = 'additive'
        elif test_name ==  'source.ux, dirichlet' or test_name == 'source.uy, dirichlet' or test_name == 'source.uz, dirichlet':
            source_fluid.u_mode   = 'dirichlet'
            source_elastic.u_mode = 'dirichlet'

        # options for writing to file, but not doing simulations
        input_filename_p = 'data_p_input.h5'
        output_filename_p = 'data_p_output.h5'
        DATA_CAST: str = 'single'
        DATA_PATH = '.'

        # set input args
        if not USE_PML:
            simulation_options_fluid = SimulationOptions(data_cast=DATA_CAST,
                                                         data_recast=True,
                                                         save_to_disk=True,
                                                         input_filename=input_filename_p,
                                                         output_filename=output_filename_p,
                                                         data_path=DATA_PATH,
                                                         use_kspace=False,
                                                         pml_alpha=0.0,
                                                         hdf_compression_level='lzf')
            simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                           pml_alpha=0.0,
                                                           kelvin_voigt_model=False)
        else:
            simulation_options_fluid = SimulationOptions(data_cast=DATA_CAST,
                                                         data_recast=True,
                                                         save_to_disk=True,
                                                         input_filename=input_filename_p,
                                                         output_filename=output_filename_p,
                                                         data_path=DATA_PATH,
                                                         use_kspace=False,
                                                         hdf_compression_level='lzf')
            simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                           kelvin_voigt_model=False)

        # options for executing simulations
        execution_options_fluid = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False)

        # run the fluid simulation
        sensor_data_fluid = kspaceFirstOrder3D(kgrid=deepcopy(kgrid),
                                               source=deepcopy(source_fluid),
                                               sensor=deepcopy(sensor_fluid),
                                               medium=deepcopy(medium_fluid),
                                               simulation_options=deepcopy(simulation_options_fluid),
                                               execution_options=deepcopy(execution_options_fluid))

        # run the elastic simulation
        sensor_data_elastic = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                              source=deepcopy(source_elastic),
                                              sensor=deepcopy(sensor_elastic),
                                              medium=deepcopy(medium_elastic),
                                              simulation_options=deepcopy(simulation_options_elastic))

        # reshape data to fit
        sensor_data_elastic['p_final'] = np.transpose(sensor_data_elastic['p_final'], (2, 1, 0))
        sensor_data_elastic['p_final'] = sensor_data_elastic['p_final'].reshape(sensor_data_elastic['p_final'].shape, order='F')

        sensor_data_elastic['ux_final'] = np.transpose(sensor_data_elastic['ux_final'], (2, 1, 0))
        sensor_data_elastic['ux_final'] = sensor_data_elastic['ux_final'].reshape(sensor_data_elastic['ux_final'].shape, order='F')

        sensor_data_elastic['uy_final'] = np.transpose(sensor_data_elastic['uy_final'], (2, 1, 0))
        sensor_data_elastic['uy_final'] = sensor_data_elastic['uy_final'].reshape(sensor_data_elastic['uy_final'].shape, order='F')

        sensor_data_elastic['uz_final'] = np.transpose(sensor_data_elastic['uz_final'], (2, 1, 0))
        sensor_data_elastic['uz_final'] = sensor_data_elastic['uz_final'].reshape(sensor_data_elastic['uz_final'].shape, order='F')

        # compute comparisons for time series
        L_inf_p = np.max(np.abs(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:] - sensor_data_fluid['p'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['p'][COMP_START_INDEX:]))
        L_inf_ux = np.max(np.abs(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:] - sensor_data_fluid['ux'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['ux'][COMP_START_INDEX:]))
        L_inf_uy = np.max(np.abs(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:] - sensor_data_fluid['uy'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['uy'][COMP_START_INDEX:]))
        L_inf_uz = np.max(np.abs(np.squeeze(sensor_data_elastic['uz'])[COMP_START_INDEX:] - sensor_data_fluid['uz'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['uz'][COMP_START_INDEX:]))

        # compuate comparisons for field
        L_inf_p_final = np.max(np.abs(sensor_data_elastic['p_final'] - sensor_data_fluid['p_final'])) / np.max(np.abs(sensor_data_fluid['p_final']))
        L_inf_ux_final = np.max(np.abs(sensor_data_elastic['ux_final'] - sensor_data_fluid['ux_final'])) / np.max(np.abs(sensor_data_fluid['ux_final']))
        L_inf_uy_final = np.max(np.abs(sensor_data_elastic['uy_final'] - sensor_data_fluid['uy_final'])) / np.max(np.abs(sensor_data_fluid['uy_final']))
        L_inf_uz_final = np.max(np.abs(sensor_data_elastic['uz_final'] - sensor_data_fluid['uz_final'])) / np.max(np.abs(sensor_data_fluid['uz_final']))

        # compute pass
        latest_test: bool = False
        if ((L_inf_p < COMPARISON_THRESH) and
            (L_inf_ux < COMPARISON_THRESH) and
            (L_inf_uy < COMPARISON_THRESH) and
            (L_inf_uz < COMPARISON_THRESH) and
            (L_inf_p_final < COMPARISON_THRESH) and
            (L_inf_ux_final < COMPARISON_THRESH) and
            (L_inf_uy_final < COMPARISON_THRESH) and
            (L_inf_uz_final < COMPARISON_THRESH)):
            # set test variable
            latest_test = True
        else:
            print('fails')

        if (L_inf_p < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_p =', L_inf_p)

        if (L_inf_ux < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_ux =', L_inf_ux)

        if (L_inf_uy < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_uy =', L_inf_uy)

        if (L_inf_uz < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_uz =', L_inf_uz)

        if (L_inf_p_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_p_final =', L_inf_p_final)

        if (L_inf_ux_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_ux_final =', L_inf_ux_final)

        if (L_inf_uy_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_uy_final =', L_inf_uy_final)

        if (L_inf_uz_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('\tfails at L_inf_uz_final =', L_inf_uz_final)

        test_pass = test_pass and latest_test


        fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2)
        fig1.suptitle(f"{test_name}: Comparisons")
        ax1a.plot(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['p'][COMP_START_INDEX:], 'b--*')
        ax1b.plot(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['ux'][COMP_START_INDEX:], 'b--*')
        ax1c.plot(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['uy'][COMP_START_INDEX:], 'b--*')
        ax1d.plot(np.squeeze(sensor_data_elastic['uz'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['uz'][COMP_START_INDEX:], 'b--*')

        fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2)
        fig2.suptitle(f"{test_name}: Errors")
        ax2a.plot(np.abs(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:] - sensor_data_fluid['p'][COMP_START_INDEX:]))
        ax2b.plot(np.abs(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:] - sensor_data_fluid['ux'][COMP_START_INDEX:]))
        ax2c.plot(np.abs(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:] - sensor_data_fluid['uy'][COMP_START_INDEX:]))
        ax2d.plot(np.abs(np.squeeze(sensor_data_elastic['uz'])[COMP_START_INDEX:] - sensor_data_fluid['uz'][COMP_START_INDEX:]))



        # clear structures
        del source_fluid
        del source_elastic
        del sensor_data_elastic
        del sensor_data_fluid

    plt.show()

    assert test_pass, "not working"

