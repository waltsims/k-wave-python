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
from kwave.pstdElastic3D import pstd_elastic_3d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.signals import tone_burst
from kwave.utils.mapgen import make_spherical_section

def test_pstd_elastic_2d_compare_with_kspaceFirstOrder2D():

    # set additional literals to give further permutations of the test
    HETEROGENEOUS       = True
    USE_PML             = False
    DATA_CAST           = 'off'
    COMPARISON_THRESH   = 5e-13

    # option to skip the first point in the time series (for p0 sources, there
    # is a strange bug where there is a high error for the first stored time
    # point)
    COMP_START_INDEX    = 2

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx = 96            # number of grid points in the x (row) direction
    Ny = 192           # number of grid points in the y (column) direction
    dx = 0.1e-3        # grid point spacing in the x direction [m]
    dy = 0.1e-3        # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the medium properties
    cp = 1500
    cs = 0
    rho = 1000

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
        medium_elastic.sound_speed_compression[Nx // 2 - 1:, :] = 2 * cp
        # fluid medium
        sound_speed = cp * np.ones((Nx, Ny))
        density = rho * np.ones((Nx, Ny))
        medium_fluid = kWaveMedium(sound_speed, density=density)
        medium_fluid.sound_speed[Nx // 2 - 1:, :] = 2 * cp

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
    test_pass = True

    # test names
    test_names = {...
        'source.p0', ...
        'source.p, additive', ...
        'source.p, dirichlet', ...
        'source.ux, additive', ...
        'source.ux, dirichlet', ...
        'source.uy, additive', ...
        'source.uy, dirichlet'}

    # define a single point sensor
    sensor.mask = np.zeros((Nx, Ny))
    sensor.mask[3 * Nx // 4, 3 * Ny // 4] = 1

    # set some things to record
    sensor.record = ['p', 'p_final', 'u', 'u_final']


    if not USE_PML:
        input_args = [input_args {'PMLAlpha', 0}]

    # loop through tests
    for test_num in np.arange(7):

        # clear structures
        del source_fluid
        del source_elastic

        # update command line
        print('Running Test: ' test_names{test_num})

        if test_num ==1:

                # create initial pressure distribution using makeDisc
                disc_magnitude = 5       # [Pa]
                disc_x_pos = 29          # [grid points]
                disc_y_pos = Ny // 2 -1  # [grid points]
                disc_radius = 6          # [grid points]
                source_fluid.p0 = disc_magnitude * make_disc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius)

                # create equivalent elastic source
                source_elastic = source_fluid

            case {2,3}

                # create pressure source
                source_fluid.p_mask = zeros(Nx, Ny)
                source_fluid.p_mask(30, Ny/2) = 1
                source_fluid.p = 5 * sin(2 * pi * 1e6 * kgrid.t_array)
                source_fluid.p = filterTimeSeries(kgrid, medium_fluid, source_fluid.p)

                # create equivalent elastic source
                source_elastic.s_mask = source_fluid.p_mask
                source_elastic.sxx = -source_fluid.p
                source_elastic.syy = -source_fluid.p

            case {4,5}

                # create velocity source
                source_fluid.u_mask = zeros(Nx, Ny)
                source_fluid.u_mask(30, Ny/2) = 1
                source_fluid.ux = 5 * sin(2 * pi * 1e6 * kgrid.t_array) ./ (cp * rho)
                source_fluid.ux = filterTimeSeries(kgrid, medium_fluid, source_fluid.ux)

                # create equivalent elastic source
                source_elastic = source_fluid

            case {6,7}

                # create velocity source
                source_fluid.u_mask = zeros(Nx, Ny)
                source_fluid.u_mask(30, Ny/2) = 1
                source_fluid.uy = 5 * sin(2 * pi * 1e6 * kgrid.t_array) ./ (cp * rho)
                source_fluid.uy = filterTimeSeries(kgrid, medium_fluid, source_fluid.uy)

                # create equivalent elastic source
                source_elastic = source_fluid

        end

        # set source mode
        switch test_num
            case 2
                source_fluid.p_mode   = 'additive'
                source_elastic.s_mode = 'additive'
            case 3
                source_fluid.p_mode   = 'dirichlet'
                source_elastic.s_mode = 'dirichlet'
            case {4, 6}
                source_fluid.u_mode   = 'additive'
                source_elastic.u_mode = 'additive'
            case {5, 7}
                source_fluid.u_mode   = 'dirichlet'
                source_elastic.u_mode = 'dirichlet'
        end

        # run the simulations
        sensor_data_elastic = pstd_elastic_2d(kgrid, medium_elastic, source_elastic, sensor, input_args{:})
        sensor_data_fluid   = kspaceFirstOrder2D(kgrid, medium_fluid,   source_fluid,   sensor, 'UsekSpace', false, input_args{:})

        # compute comparisons for time series
        L_inf_p         = np.max(np.abs(sensor_data_elastic.p[COMP_START_INDEX:] - sensor_data_fluid.p[COMP_START_INDEX:]))  / np.max(np.abssensor_data_fluid.p[COMP_START_INDEX:]))
        L_inf_ux        = np.max(np.abs(sensor_data_elastic.ux[COMP_START_INDEX:] - sensor_data_fluid.ux[COMP_START_INDEX:])) / np.max(np.abssensor_data_fluid.ux[COMP_START_INDEX:]))
        L_inf_uy        = np.max(np.abs(sensor_data_elastic.uy[COMP_START_INDEX:] - sensor_data_fluid.uy[COMP_START_INDEX:])) / np.max(np.abssensor_data_fluid.uy[COMP_START_INDEX:]))

        # compuate comparisons for field
        L_inf_p_final   = np.max(np.abs(sensor_data_elastic.p_final - sensor_data_fluid.p_final)) / np.max(np.abs(sensor_data_fluid.p_final))
        L_inf_ux_final  = np.max(np.abs(sensor_data_elastic.ux_final - sensor_data_fluid.ux_final)) / np.max(np.abs(sensor_data_fluid.ux_final))
        L_inf_uy_final  = np.max(np.abs(sensor_data_elastic.uy_final - sensor_data_fluid.uy_final)) / np.max(np.abs(sensor_data_fluid.uy_final))

        # compute pass
        if (L_inf_p > COMPARISON_THRESH) or (L_inf_ux > COMPARISON_THRESH) or (L_inf_uy > COMPARISON_THRESH) or (L_inf_p_final > COMPARISON_THRESH) or (L_inf_ux_final > COMPARISON_THRESH) or (L_inf_uy_final > COMPARISON_THRESH)
            # set test variable
            test_pass = False


