"""
Unit test to compare that the elastic code with the shear wave speed set to
zero gives the same answers as the regular fluid code in k-Wave.
"""

import numpy as np
from copy import deepcopy
# import pytest
import matplotlib.pyplot as plt

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic2D import pstd_elastic_2d
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_disc

from scipy.io import loadmat


#@pytest.mark.skip(reason="2D not ready")
def test_pstd_elastic_2d_compare_with_kspaceFirstOrder2D():

    # set additional literals to give further permutations of the test
    HETEROGENEOUS: bool = True
    USE_PML: bool = False
    COMPARISON_THRESH = 5e-10

    # option to skip the first point in the time series (for p0 sources, there
    # is a strange bug where there is a high error for the first stored time
    # point)
    COMP_START_INDEX: int = 1

    # set pass variable
    test_pass: bool = True

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx: int = 96            # number of grid points in the x (row) direction
    Ny: int = 192           # number of grid points in the y (column) direction
    dx: float = 0.1e-3             # grid point spacing in the x direction [m]
    dy: float = 0.1e-3             # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the medium properties
    cp: float = 1500.0
    cs: float = 0.0
    rho: float = 1000.0

    # create the time array
    CFL: float = 0.1
    t_end: float = 7e-6
    kgrid.makeTime(cp, CFL, t_end)

    # create and assign the medium properties
    if HETEROGENEOUS:
        # elastic medium
        sound_speed_compression = cp * np.ones((Nx, Ny))
        sound_speed_compression[Nx // 2 - 1:, :] = 2.0 * cp
        sound_speed_shear = cs * np.ones((Nx, Ny))
        density = rho * np.ones((Nx, Ny))
        medium_elastic = kWaveMedium(sound_speed=sound_speed_compression,
                                     density=density,
                                     sound_speed_compression=sound_speed_compression,
                                     sound_speed_shear=sound_speed_shear)
        # fluid medium
        sound_speed = cp * np.ones((Nx, Ny))
        sound_speed[Nx // 2 - 1:, :] = 2.0 * cp
        medium_fluid = kWaveMedium(sound_speed,
                                   density=density)
    else:
        # elastic medium
        medium_elastic = kWaveMedium(sound_speed=cp,
                                     density=rho,
                                     sound_speed_compression=cp,
                                     sound_speed_shear=cs)
        # fluid medium
        medium_fluid = kWaveMedium(sound_speed=cp,
                                   density=rho)

    # test names
    test_names = ['source.p0',
                  'source.p, additive',
                  'source.p, dirichlet',
                  'source.ux, additive',
                  'source.ux, dirichlet',
                  'source.uy, additive',
                  'source.uy, dirichlet'
                  ]

    # define a single point sensor
    sensor_elastic = kSensor()
    sensor_fluid = kSensor()
    sensor_elastic.mask = np.zeros((Nx, Ny), dtype=bool)
    sensor_elastic.mask[3 * Nx // 4 - 1, 3 * Ny // 4 - 1] = True
    sensor_fluid.mask = np.zeros((Nx, Ny), dtype=bool)
    sensor_fluid.mask[3 * Nx // 4 - 1, 3 * Ny // 4 - 1] = True

    # set some things to record
    sensor_elastic.record = ['p', 'p_final', 'u', 'u_final']
    sensor_fluid.record = ['p', 'p_final', 'u', 'u_final']

    # loop through tests
    for test_num, test_name in enumerate(test_names):

        source_fluid = kSource()
        source_elastic = kSource()

        x_pos: int = 30           # [grid points]
        y_pos: int = Ny // 2      # [grid points]

        # update command line
        print('Running Number: ', test_num, ':', test_name)

        if test_name == 'source.p0':
            # create initial pressure distribution using makeDisc
            disc_magnitude: float = 5.0            # [Pa]
            disc_radius: int = 6          # [grid points]
            p0 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([x_pos, y_pos]), disc_radius).astype(float)
            source_fluid.p0 = p0
            # create equivalent elastic source
            source_elastic.p0 = p0

        elif test_name == 'source.p, additive' or test_name == 'source.p, dirichlet':
            # create pressure source
            source_fluid.p_mask = np.zeros((Nx, Ny), dtype=bool)
            freq: float = 2.0 * np.pi * 1e6
            magnitude: float = 5.0    # [Pa]
            source_fluid.p_mask[x_pos, y_pos] = bool
            p = magnitude * np.sin(freq * np.squeeze(kgrid.t_array))
            source_fluid.p = filter_time_series(deepcopy(kgrid), deepcopy(medium_fluid), p)
            # create equivalent elastic source
            source_elastic.s_mask = source_fluid.p_mask
            source_elastic.sxx = -deepcopy(source_fluid.p)
            source_elastic.syy = -deepcopy(source_fluid.p)

        elif test_name == 'source.ux, additive' or test_name == 'source.ux, dirichlet':
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny), dtype=bool)
            source_fluid.u_mask[x_pos, y_pos] = True
            ux = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.ux = filter_time_series(deepcopy(kgrid), deepcopy(medium_fluid), ux)
            # create equivalent elastic source
            source_elastic = deepcopy(source_fluid)

        elif test_name == 'source.uy, additive' or test_name == 'source.uy, dirichlet':
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny), dtype=bool)
            source_fluid.u_mask[x_pos, y_pos] = True
            uy = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.uy = filter_time_series(deepcopy(kgrid), deepcopy(medium_fluid), uy)
            # create equivalent elastic source
            source_elastic = deepcopy(source_fluid)

        # set source mode
        if test_name == 'source.p, additive':
            source_fluid.p_mode   = 'additive'
            source_elastic.s_mode = 'additive'
        elif test_name == 'source.p, dirichlet':
            source_fluid.p_mode   = 'dirichlet'
            source_elastic.s_mode = 'dirichlet'
        elif test_name == 'source.ux, additive' or test_name == 'source.uy, additive':
            source_fluid.u_mode   = 'additive'
            source_elastic.u_mode = 'additive'
        elif test_name == 'source.ux, dirichlet' or test_name == 'source.uy, dirichlet':
            source_fluid.u_mode   = 'dirichlet'
            source_elastic.u_mode = 'dirichlet'

        # options for writing to file, but not doing simulations
        input_filename_p = 'data_p_input.h5'
        output_filename_p = 'data_p_output.h5'
        DATA_CAST: str = 'single'
        DATA_PATH = '.'

        if not USE_PML:
            simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                           pml_alpha=0.0)
            simulation_options_fluid = SimulationOptions(simulation_type=SimulationType.FLUID,
                                                         data_cast=DATA_CAST,
                                                         data_recast=True,
                                                         save_to_disk=True,
                                                         input_filename=input_filename_p,
                                                         output_filename=output_filename_p,
                                                         data_path=DATA_PATH,
                                                         use_kspace=False,
                                                         pml_alpha=0.0,
                                                         hdf_compression_level='lzf')
        else:
            simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC)
            simulation_options_fluid = SimulationOptions(simulation_type=SimulationType.FLUID,
                                                         data_cast=DATA_CAST,
                                                         data_recast=True,
                                                         save_to_disk=True,
                                                         input_filename=input_filename_p,
                                                         output_filename=output_filename_p,
                                                         data_path=DATA_PATH,
                                                         use_kspace=False,
                                                         hdf_compression_level='lzf')

        # options for executing simulations
        execution_options_fluid = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=False)

        # run the fluid simulation
        sensor_data_fluid = kspace_first_order_2d_gpu(medium=deepcopy(medium_fluid),
                                                      kgrid=deepcopy(kgrid),
                                                      source=deepcopy(source_fluid),
                                                      sensor=deepcopy(sensor_fluid),
                                                      simulation_options=deepcopy(simulation_options_fluid),
                                                      execution_options=deepcopy(execution_options_fluid))

        # run the simulations
        sensor_data_elastic = pstd_elastic_2d(medium=deepcopy(medium_elastic),
                                              kgrid=deepcopy(kgrid),
                                              source=deepcopy(source_elastic),
                                              sensor=deepcopy(sensor_elastic),
                                              simulation_options=deepcopy(simulation_options_elastic))

        # reshape data to fit
        sensor_data_elastic['p_final'] = np.transpose(sensor_data_elastic['p_final'])
        sensor_data_elastic['p_final'] = sensor_data_elastic['p_final'].reshape(sensor_data_elastic['p_final'].shape, order='F')

        sensor_data_elastic['ux_final'] = np.transpose(sensor_data_elastic['ux_final'])
        sensor_data_elastic['ux_final'] = sensor_data_elastic['ux_final'].reshape(sensor_data_elastic['ux_final'].shape, order='F')

        sensor_data_elastic['uy_final'] = np.transpose(sensor_data_elastic['uy_final'])
        sensor_data_elastic['uy_final'] = sensor_data_elastic['uy_final'].reshape(sensor_data_elastic['uy_final'].shape, order='F')

        # compute comparisons for time series
        L_inf_p = np.max(np.abs(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:] - sensor_data_fluid['p'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['p'][COMP_START_INDEX:]))
        L_inf_ux = np.max(np.abs(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:] - sensor_data_fluid['ux'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['ux'][COMP_START_INDEX:]))
        L_inf_uy = np.max(np.abs(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:] - sensor_data_fluid['uy'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['uy'][COMP_START_INDEX:]))

        # compuate comparisons for field
        L_inf_p_final = np.max(np.abs(sensor_data_elastic['p_final'] - sensor_data_fluid['p_final'])) / np.max(np.abs(sensor_data_fluid['p_final']))
        L_inf_ux_final = np.max(np.abs(sensor_data_elastic['ux_final'] - sensor_data_fluid['ux_final'])) / np.max(np.abs(sensor_data_fluid['ux_final']))
        L_inf_uy_final = np.max(np.abs(sensor_data_elastic['uy_final'] - sensor_data_fluid['uy_final'])) / np.max(np.abs(sensor_data_fluid['uy_final']))

        # compute pass
        latest_test: bool = False
        if ((L_inf_p < COMPARISON_THRESH) and (L_inf_ux < COMPARISON_THRESH) and
            (L_inf_uy < COMPARISON_THRESH) and (L_inf_p_final < COMPARISON_THRESH) and
            (L_inf_ux_final < COMPARISON_THRESH) and (L_inf_uy_final < COMPARISON_THRESH)):
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

        test_pass = test_pass and latest_test

        ###########


        print('\t', np.squeeze(sensor_data_elastic['p'])[-4:], '\n\t', sensor_data_fluid['p'][-4:])
        print(str(np.squeeze(sensor_data_elastic['ux'])[-4:]) + '\n' + str(sensor_data_fluid['ux'][-4:]))
        print('\t', np.squeeze(sensor_data_elastic['uy'])[-4:], '\n\t', sensor_data_fluid['uy'][-4:])

        fig1, ((ax1a, ax1b, ax1c,)) = plt.subplots(3, 1)
        fig1.suptitle(f"{test_name}: Comparisons")
        # if test_num == 0:
        #     ax1a.plot(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['p'][COMP_START_INDEX:], 'b--*',
        #           np.squeeze(matlab_sensor_fluid_p), 'k--o', np.squeeze(matlab_sensor_elastic_p), 'k-+',)
        # else:
        ax1a.plot(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['p'][COMP_START_INDEX:], 'b--*')
        ax1b.plot(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['ux'][COMP_START_INDEX:], 'b--*')
        ax1c.plot(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:], 'r-o', sensor_data_fluid['uy'][COMP_START_INDEX:], 'b--*')

        # fig2, ((ax2a, ax2b, ax2c)) = plt.subplots(3, 1)
        # fig2.suptitle(f"{test_name}: Errors")
        # ax2a.plot(np.abs(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:] - sensor_data_fluid['p'][COMP_START_INDEX:]))
        # ax2b.plot(np.abs(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:] - sensor_data_fluid['ux'][COMP_START_INDEX:]))
        # ax2c.plot(np.abs(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:] - sensor_data_fluid['uy'][COMP_START_INDEX:]))

        # if test_num == 0:
        #     fig4, ((ax4a, ax4b, ax4c)) = plt.subplots(3, 1)
        #     ax4a.imshow(source_fluid.p0.astype(float))
        #     ax4b.imshow(matlab_source_fluid_p0.astype(float))
        #     ax4c.imshow(source_fluid.p0.astype(float) - matlab_source_fluid_p0.astype(float))

            # fig5, (ax5a, ax5b) = plt.subplots(1, 2)
            # ax5a.imshow(mat_sxx[:,0].reshape(p0.shape, order='F') )
            # ax5b.imshow(sxx[:,0].reshape(p0.shape, order='F') )

        # fig3, ((ax3a, ax3b, ax3c), (ax3d, ax3e, ax3f)) = plt.subplots(2, 3)
        # fig3.suptitle(f"{test_name}: Final Values")
        # ax3a.imshow(sensor_data_elastic['p_final'])
        # ax3b.imshow(sensor_data_elastic['ux_final'])
        # ax3c.imshow(sensor_data_elastic['uy_final'])
        # ax3d.imshow(sensor_data_fluid['p_final'])
        # ax3e.imshow(sensor_data_fluid['ux_final'])
        # ax3f.imshow(sensor_data_fluid['uy_final'])

        # clear structures
        del source_fluid
        del source_elastic
        del sensor_data_elastic
        del sensor_data_fluid

    plt.show()

    assert test_pass, "not working"
