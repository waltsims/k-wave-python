"""
Unit test to compare that the elastic code with the shear wave speed set to
zero gives the same answers as the regular fluid code in k-Wave.
"""

import numpy as np
from copy import deepcopy

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

def test_pstd_elastic_2d_compare_with_kspaceFirstOrder2D():

    # set additional literals to give further permutations of the test
    HETEROGENEOUS: bool = True
    USE_PML: bool = False
    COMPARISON_THRESH = 5e-13

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
    dx = 0.1e-3        # grid point spacing in the x direction [m]
    dy = 0.1e-3        # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the medium properties
    cp = 1500.0
    cs = 0.0
    rho = 1000.0

    # create the time array
    CFL = 0.1
    t_end = 7e-6
    kgrid.makeTime(cp, CFL, t_end)

    # create and assign the medium properties
    if HETEROGENEOUS:
        # elastic medium
        sound_speed_compression = cp * np.ones((Nx, Ny))
        sound_speed_shear = cs * np.ones((Nx, Ny))
        density = rho * np.ones((Nx, Ny))
        medium_elastic = kWaveMedium(sound_speed_compression,
                                    density=density,
                                    sound_speed_compression=sound_speed_compression,
                                    sound_speed_shear=sound_speed_shear)
        medium_elastic.sound_speed_compression[Nx // 2 - 1:, :] = 2.0 * cp

        # fluid medium
        sound_speed = cp * np.ones((Nx, Ny))
        density = rho * np.ones((Nx, Ny))
        medium_fluid = kWaveMedium(sound_speed, density=density)
        medium_fluid.sound_speed[Nx // 2 - 1:, :] = 2.0 * cp

    else:
        # elastic medium
        medium_elastic = kWaveMedium(cp, density=rho, sound_speed_compression=cp,
                                     sound_speed_shear=cs)
        # fluid medium
        medium_fluid = kWaveMedium(sound_speed=cp, density=rho)

    # test names
    test_names = ['source.p0', 'source.p, additive', 'source.p, dirichlet',
                  'source.ux, additive', 'source.ux, dirichlet',
                  'source.uy, additive', 'source.uy, dirichlet']

    # define a single point sensor
    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Ny))
    sensor.mask[3 * Nx // 4, 3 * Ny // 4] = 1

    # set some things to record
    sensor.record = ['p', 'p_final', 'u', 'u_final']

    if not USE_PML:
        simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                       pml_alpha=0.0, kelvin_voigt_model=False)
    else:
        simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                       kelvin_voigt_model=False)

    simulation_options_fluid = SimulationOptions(simulation_type=SimulationType.FLUID, use_kspace=False)

    # loop through tests
    for test_num in np.arange(7):

        source_fluid = kSource()
        source_elastic = kSource()

        # update command line
        print('Running Test: ', test_names[test_num])

        if test_num == 0:
            # create initial pressure distribution using makeDisc
            disc_magnitude = 5.0           # [Pa]
            disc_x_pos: int = 29           # [grid points]
            disc_y_pos: int = Ny // 2 - 1  # [grid points]
            disc_radius: int = 6           # [grid points]
            source_fluid.p0 = disc_magnitude * make_disc(Vector([Nx, Ny]),
                                                         Vector([disc_x_pos, disc_y_pos]), disc_radius)
            # create equivalent elastic source
            source_elastic = deepcopy(source_fluid)

        elif test_num == 1 or test_num == 2:
            # create pressure source
            source_fluid.p_mask = np.zeros((Nx, Ny))
            source_fluid.p_mask[29, Ny // 2 - 1] = 1
            source_fluid.p = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array))
            source_fluid.p = filter_time_series(deepcopy(kgrid),
                                                deepcopy(medium_fluid),
                                                deepcopy(source_fluid.p))
            # create equivalent elastic source
            source_elastic.s_mask = source_fluid.p_mask
            source_elastic.sxx = -source_fluid.p
            source_elastic.syy = -source_fluid.p

        elif test_num == 3 or test_num == 4:
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny))
            source_fluid.u_mask[29, Ny // 2 - 1] = 1
            source_fluid.ux = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array))
            source_fluid.ux = filter_time_series(kgrid, medium_fluid, source_fluid.ux)
            # create equivalent elastic source
            source_elastic = source_fluid

        elif test_num == 5 or test_num == 6:
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny))
            source_fluid.u_mask[29, Ny // 2 - 1] = 1
            source_fluid.uy = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.uy = filter_time_series(kgrid, medium_fluid, source_fluid.uy)
            # create equivalent elastic source
            source_elastic = source_fluid

        # set source mode
        if test_num == 1:
            source_fluid.p_mode   = 'additive'
            source_elastic.s_mode = 'additive'
        elif test_num == 2:
            source_fluid.p_mode   = 'dirichlet'
            source_elastic.s_mode = 'dirichlet'
        elif test_num == 3 or test_num == 5:
            source_fluid.u_mode   = 'additive'
            source_elastic.u_mode = 'additive'
        elif test_num == 4 or test_num == 6:
            source_fluid.u_mode   = 'dirichlet'
            source_elastic.u_mode = 'dirichlet'

        # options for writing to file, but not doing simulations
        input_filename_p = 'data_p_input.h5'
        output_filename_p = 'data_p_output.h5'
        DATA_CAST: str = 'single'
        DATA_PATH = '.'
        simulation_options_fluid = SimulationOptions(data_cast=DATA_CAST,
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
        print(kgrid)
        sensor_data_fluid = kspace_first_order_2d_gpu(medium=deepcopy(medium_fluid),
                                                      kgrid=deepcopy(kgrid),
                                                      source=deepcopy(source_fluid),
                                                      sensor=deepcopy(sensor),
                                                      simulation_options=deepcopy(simulation_options_fluid),
                                                      execution_options=deepcopy(execution_options_fluid))

        # run the simulations
        print(kgrid)
        sensor_data_elastic = pstd_elastic_2d(medium=deepcopy(medium_elastic),
                                              kgrid=deepcopy(kgrid),
                                              source=deepcopy(source_elastic),
                                              sensor=deepcopy(sensor),
                                              simulation_options=deepcopy(simulation_options_elastic))
        print(kgrid)
        # compute comparisons for time series
        L_inf_p = np.max(np.abs(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:] - sensor_data_fluid['p'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['p'][COMP_START_INDEX:]))
        L_inf_ux = np.max(np.abs(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:] - sensor_data_fluid['ux'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['ux'][COMP_START_INDEX:]))
        L_inf_uy = np.max(np.abs(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:] - sensor_data_fluid['uy'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['uy'][COMP_START_INDEX:]))

        # compuate comparisons for field
        L_inf_p_final = np.max(np.abs(sensor_data_elastic['p_final'].T - sensor_data_fluid['p_final'])) / np.max(np.abs(sensor_data_fluid['p_final']))
        L_inf_ux_final = np.max(np.abs(sensor_data_elastic['ux_final'].T  - sensor_data_fluid['ux_final'])) / np.max(np.abs(sensor_data_fluid['ux_final']))
        L_inf_uy_final = np.max(np.abs(sensor_data_elastic['uy_final'].T - sensor_data_fluid['uy_final'])) / np.max(np.abs(sensor_data_fluid['uy_final']))

        # compute pass
        latest_test: bool = False
        if ((L_inf_p < COMPARISON_THRESH) and (L_inf_ux < COMPARISON_THRESH) and
            (L_inf_uy < COMPARISON_THRESH) and (L_inf_p_final < COMPARISON_THRESH) and
            (L_inf_ux_final < COMPARISON_THRESH) and (L_inf_uy_final < COMPARISON_THRESH)):
            # set test variable
            latest_test = True
        else:
            print('fails')

        test_pass = test_pass and latest_test

        # clear structures
        del source_fluid
        del source_elastic
        del sensor_data_elastic
        del sensor_data_fluid

        print(kgrid)

    assert test_pass, "not working"
