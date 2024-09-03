import numpy as np
from copy import deepcopy

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

"""
Unit test to compare that the elastic code with the shear wave speed set to
zero gives the same answers as the regular fluid code in k-Wave.
"""

def test_pstd_elastic_3D_compare_with_kspaceFirstOrder3D():

    # set additional literals to give further permutations of the test
    HETEROGENEOUS: bool = True
    USE_PML: bool = False
    DATA_CAST = 'off'
    COMPARISON_THRESH: float = 5e-13

    # option to skip the first point in the time series (for p0 sources, there
    # is a strange bug where there is a high error for the first stored time
    # point)
    COMP_START_INDEX: int = 1

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx: int = 64
    Ny: int = 64
    Nz: int = 64
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
        sound_speed_shear = cs * np.ones((Nx, Ny, Nz))
        density = rho * np.ones((Nx, Ny, Nz))
        medium_elastic = kWaveMedium(sound_speed_compression,
                                    density=density,
                                    sound_speed_compression=sound_speed_compression,
                                    sound_speed_shear=sound_speed_shear)
        medium_elastic.sound_speed_compression[Nx // 2 - 1:, :, :] = 2 * cp
        # fluid medium
        sound_speed = cp * np.ones((Nx, Ny, Nz))
        density = rho * np.ones((Nx, Ny, Nz))
        medium_fluid = kWaveMedium(sound_speed, density=density)
        medium_fluid.sound_speed[Nx // 2 - 1:, :, :] = 2 * cp

    else:
        # elastic medium
        medium_elastic = kWaveMedium(cp, density=rho,
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
        'source.p, dirichlet',
        'source.ux, additive',
        'source.ux, dirichlet',
        'source.uy, additive',
        'source.uy, dirichlet',
        'source.uz, additive',
        'source.uz, dirichlet']

    # define a single point sensor
    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Ny, Nz))
    sensor.mask[3 * Nx // 4 - 1, 3 * Ny // 4 - 1, 3 * Nz // 4 - 1] = 1

    # set some things to record
    sensor.record = ['p', 'p_final', 'u', 'u_final']

    # set input args
    if not USE_PML:
        simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                       pml_alpha=0.0, kelvin_voigt_model=False)
    else:
        simulation_options_elastic = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                       kelvin_voigt_model=False)

    # loop through tests
    for test_num in np.arange(2,len(test_names)):

        # update command line
        print('Running Test: ', test_names[test_num])

        # set up sources
        source_fluid = kSource()
        source_elastic = kSource()

        if test_num == 0:
            # create initial pressure distribution using makeBall
            disc_magnitude: float = 5.0     # [Pa]
            disc_x_pos: int = Nx // 2 - 11  # [grid points]
            disc_y_pos: int = Ny // 2 - 1   # [grid points]
            disx_z_pos: int = Nz // 2 - 1   # [grid points]
            disc_radius: int = 3            # [grid points]
            source_fluid.p0 = disc_magnitude * make_ball(Vector([Nx, Ny, Nz]),
                                                         Vector([disc_x_pos, disc_y_pos, disx_z_pos]),
                                                         disc_radius)

            # assign to elastic source
            source_elastic = deepcopy(source_fluid)

        elif test_num == 1 or test_num == 2:
            # create pressure source
            source_fluid.p_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.p_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.p = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array))
            source_fluid.p = filter_time_series(deepcopy(kgrid),
                                                deepcopy(medium_fluid),
                                                deepcopy(source_fluid.p))
            # create equivalent elastic source
            source_elastic.s_mask = source_fluid.p_mask
            source_elastic.sxx = -source_fluid.p
            source_elastic.syy = -source_fluid.p
            source_elastic.szz = -source_fluid.p

        elif test_num == 3 or test_num == 4:
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.ux = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array))
            source_fluid.ux = filter_time_series(kgrid, medium_fluid, source_fluid.ux)
            # create equivalent elastic source
            source_elastic = source_fluid

        elif test_num == 5 or test_num == 6:
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.uy = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.uy = filter_time_series(kgrid, medium_fluid, source_fluid.uy)
            # create equivalent elastic source
            source_elastic = source_fluid

        elif test_num == 7 or test_num == 8:
            # create velocity source
            source_fluid.u_mask = np.zeros((Nx, Ny, Nz))
            source_fluid.u_mask[Nx // 2 - 11, Ny // 2 - 1, Nz // 2 - 1] = 1
            source_fluid.uz = 5.0 * np.sin(2.0 * np.pi * 1e6 * np.squeeze(kgrid.t_array)) / (cp * rho)
            source_fluid.uz = filter_time_series(kgrid, medium_fluid, source_fluid.uz)
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
        sensor_data_fluid = kspaceFirstOrder3D(kgrid=deepcopy(kgrid),
                                               source=deepcopy(source_fluid),
                                               sensor=deepcopy(sensor),
                                               medium=deepcopy(medium_fluid),
                                               simulation_options=deepcopy(simulation_options_fluid),
                                               execution_options=deepcopy(execution_options_fluid))

        # run the elastic simulation
        sensor_data_elastic = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                              source=deepcopy(source_elastic),
                                              sensor=deepcopy(sensor),
                                              medium=deepcopy(medium_elastic),
                                              simulation_options=deepcopy(simulation_options_elastic))

        # compute comparisons for time series
        L_inf_p = np.max(np.abs(np.squeeze(sensor_data_elastic['p'])[COMP_START_INDEX:] - sensor_data_fluid['p'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['p'][COMP_START_INDEX:]))
        L_inf_ux = np.max(np.abs(np.squeeze(sensor_data_elastic['ux'])[COMP_START_INDEX:] - sensor_data_fluid['ux'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['ux'][COMP_START_INDEX:]))
        L_inf_uy = np.max(np.abs(np.squeeze(sensor_data_elastic['uy'])[COMP_START_INDEX:] - sensor_data_fluid['uy'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['uy'][COMP_START_INDEX:]))
        L_inf_uz = np.max(np.abs(np.squeeze(sensor_data_elastic['uz'])[COMP_START_INDEX:] - sensor_data_fluid['uz'][COMP_START_INDEX:])) / np.max(np.abs(sensor_data_fluid['uz'][COMP_START_INDEX:]))

        # compuate comparisons for field
        L_inf_p_final = np.max(np.abs(sensor_data_elastic['p_final'].T - sensor_data_fluid['p_final'])) / np.max(np.abs(sensor_data_fluid['p_final']))
        L_inf_ux_final = np.max(np.abs(sensor_data_elastic['ux_final'].T  - sensor_data_fluid['ux_final'])) / np.max(np.abs(sensor_data_fluid['ux_final']))
        L_inf_uy_final = np.max(np.abs(sensor_data_elastic['uy_final'].T - sensor_data_fluid['uy_final'])) / np.max(np.abs(sensor_data_fluid['uy_final']))
        L_inf_uz_final = np.max(np.abs(sensor_data_elastic['uz_final'].T - sensor_data_fluid['uz_final'])) / np.max(np.abs(sensor_data_fluid['uz_final']))

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
            print('fails at L_inf_p =', L_inf_p)

        if (L_inf_ux < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_ux =', L_inf_ux)

        if (L_inf_uy < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_uy =', L_inf_uy)

        if (L_inf_uz < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_uz =', L_inf_uz)

        if (L_inf_p_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_p_final =', L_inf_p_final)

        if (L_inf_ux_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_ux_final =', L_inf_ux_final)

        if (L_inf_uy_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_uy_final =', L_inf_uy_final)

        if (L_inf_uz_final < COMPARISON_THRESH):
            latest_test = True
        else:
            print('fails at L_inf_uz_final =', L_inf_uz_final)

        test_pass = test_pass and latest_test

        # clear structures
        del source_fluid
        del source_elastic
        del sensor_data_elastic
        del sensor_data_fluid

    assert test_pass, "not working"

