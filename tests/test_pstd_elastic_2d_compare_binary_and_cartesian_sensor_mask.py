"""
Unit test to compare cartesian and binary sensor masks.
"""

import numpy as np
from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic2D import pstd_elastic_2d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.conversion import cart2grid
from kwave.utils.mapgen import make_circle
from kwave.utils.signals import reorder_binary_sensor_data

@pytest.mark.skip(reason="2D not ready")
def test_pstd_elastic_2d_compare_binary_and_cartesian_sensor_mask():

    # set comparison threshold
    comparison_thresh = 1e-14

    # set pass variable
    test_pass = True

    # create the computational grid
    Nx: int = 128  # [grid points]
    Ny: int = 128  # [grid points]
    dx = 25e-3 / float(Nx)  # [m]
    dy = dx                 # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny))  # [m/s]
    sound_speed_shear = np.zeros((Nx, Ny))
    density = 1000.0 * np.ones((Nx, Ny))
    medium = kWaveMedium(sound_speed_compression,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear,
                         density=density)
    medium.sound_speed_shear[Nx // 2 - 1:, :] = 1200.0
    medium.sound_speed_compression[Nx // 2 - 1:, :] = 2000.0
    medium.density[Nx // 2 - 1:, :] = 1200.0

    # define source mask
    source = kSource()
    p0 = np.zeros((Nx, Ny))
    p0[21, Ny // 4 - 1:3 * Ny // 4] = 1
    source._p0 = p0

    # record all output variables
    sensor = kSensor()
    # sensor.record = ['p',
    #                  'p_max',
    #                  'p_min',
    #                  'p_rms',
    #                  'u',
    #                  'u_max',
    #                  'u_min',
    #                  'u_rms',
    #                  'u_non_staggered',
    #                  'I',
    #                  'I_avg']

    sensor.record = ['p',
                     'p_max',
                     'p_min',
                     'p_rms']

    # define Cartesian sensor points using points exactly on the grid
    circ_mask = make_circle(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), int(Nx // 2 - 10))
    x_points = kgrid.x[circ_mask == 1]
    y_points = kgrid.y[circ_mask == 1]
    sensor.mask = np.vstack((x_points, y_points))
    print(np.shape(x_points),
          np.shape(y_points),
          np.shape(np.hstack((x_points, y_points))),
          np.shape(np.vstack((x_points, y_points))) )

    # # run the simulation as normal
    # simulation_options_c_ln = SimulationOptions(simulation_type=SimulationType.ELASTIC,
    #                                             cart_interp='linear',
    #                                             kelvin_voigt_model=False)

    # sensor_data_c_ln = pstd_elastic_2d(kgrid=deepcopy(kgrid),
    #                                    medium=deepcopy(medium),
    #                                    sensor=deepcopy(sensor),
    #                                    source=deepcopy(source),
    #                                    simulation_options=deepcopy(simulation_options_c_ln))

    # # run the simulation using nearest-neighbour interpolation
    # simulation_options_c_nn = SimulationOptions(simulation_type=SimulationType.ELASTIC,
    #                                             cart_interp='nearest')
    # sensor_data_c_nn = pstd_elastic_2d(kgrid=deepcopy(kgrid),
    #                                    medium=deepcopy(medium),
    #                                    source=deepcopy(source),
    #                                    sensor=deepcopy(sensor),
    #                                    simulation_options=deepcopy(simulation_options_c_nn))

    # convert sensor mask
    sensor.mask, order_index, reorder_index = cart2grid(kgrid, sensor.mask)

    print(np.shape(order_index), np.shape(reorder_index), np.shape(sensor.mask), sensor.mask.ndim)

    # run the simulation again
    simulation_options_b = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                             cart_interp='linear',
                                             kelvin_voigt_model=False)
    sensor_data_b = pstd_elastic_2d(kgrid=deepcopy(kgrid),
                                    medium=deepcopy(medium),
                                    source=deepcopy(source),
                                    sensor=deepcopy(sensor),
                                    simulation_options=deepcopy(simulation_options_b))

    # reorder the binary sensor data
    print(Nx, Ny, kgrid.Nt)
    # sensor_data_b['p'] = reorder_binary_sensor_data(np.reshape(sensor_data_b['p'], (Nx,Ny)), reorder_index)
    sensor_data_b['p_max'] = reorder_binary_sensor_data(np.reshape(sensor_data_b['p_max'], np.shape(sensor.mask)), reorder_index)
    # sensor_data_b['p_min'] = reorder_binary_sensor_data(np.reshape(sensor_data_b['p_min'], (Nx,Ny)), reorder_index)
    # sensor_data_b['p_rms'] = reorder_binary_sensor_data(np.reshape(sensor_data_b['p_rms'], (Nx,Ny)), reorder_index)

    # sensor_data_b.ux                = reorder_binary_sensor_data(sensor_data_b.ux, reorder_index)
    # sensor_data_b.uy                = reorder_binary_sensor_data(sensor_data_b.uy, reorder_index)
    # sensor_data_b.ux_max            = reorder_binary_sensor_data(sensor_data_b.ux_max, reorder_index)
    # sensor_data_b.uy_max            = reorder_binary_sensor_data(sensor_data_b.uy_max, reorder_index)
    # sensor_data_b.ux_min            = reorder_binary_sensor_data(sensor_data_b.ux_min, reorder_index)
    # sensor_data_b.uy_min            = reorder_binary_sensor_data(sensor_data_b.uy_min, reorder_index)
    # sensor_data_b.ux_rms            = reorder_binary_sensor_data(sensor_data_b.ux_rms, reorder_index)
    # sensor_data_b.uy_rms            = reorder_binary_sensor_data(sensor_data_b.uy_rms, reorder_index)
    # sensor_data_b.ux_non_staggered	= reorder_binary_sensor_data(sensor_data_b.ux_non_staggered, reorder_index)
    # sensor_data_b.uy_non_staggered  = reorder_binary_sensor_data(sensor_data_b.uy_non_staggered, reorder_index)

    # sensor_data_b['Ix'] = reorder_binary_sensor_data(sensor_data_b['Ix'], reorder_index)
    # sensor_data_b['Iy'] = reorder_binary_sensor_data(sensor_data_b['Iy'], reorder_index)
    # sensor_data_b['Ix_avg'] = reorder_binary_sensor_data(sensor_data_b['Ix_avg'], reorder_index)
    # sensor_data_b['Iy_avg'] = reorder_binary_sensor_data(sensor_data_b['Iy_avg'], reorder_index)

    # compute errors
    err_p_nn = np.max(np.abs(sensor_data_c_nn['p'] - sensor_data_b['p'])) / np.max(np.abs(sensor_data_b['p']))
    if (err_p_nn > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_nn.p - sensor_data_b.p"

    err_p_ln = np.max(np.abs(sensor_data_c_ln['p'] - sensor_data_b['p'])) / np.max(np.abs(sensor_data_b['p']))
    if (err_p_ln > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p - sensor_data_b.p"

    err_p_max_nn = np.max(np.abs(sensor_data_c_nn['p_max'] - sensor_data_b['p_max'])) / np.max(np.abs(sensor_data_b['p_max']))
    if (err_p_max_nn > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p_max - sensor_data_b.pmax"

    err_p_max_ln = np.max(np.abs(sensor_data_c_ln['p_max'] - sensor_data_b['p_max'])) / np.max(np.abs(sensor_data_b['p_max']))
    if (err_p_max_ln > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p - sensor_data_b.p"

    err_p_min_nn = np.max(np.abs(sensor_data_c_nn['p_min'] - sensor_data_b['p_min'])) / np.max(np.abs(sensor_data_b['p_min']))
    if (err_p_min_nn > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p - sensor_data_b.p"

    err_p_min_ln = np.max(np.abs(sensor_data_c_ln['p_min']- sensor_data_b['p_min'])) / np.max(np.abs(sensor_data_b['p_min']))
    if (err_p_min_ln > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p - sensor_data_b.p"

    err_p_rms_nn = np.max(np.abs(sensor_data_c_nn['p_rms']- sensor_data_b['p_rms'])) / np.max(np.abs(sensor_data_b['p_rms']))
    if (err_p_rms_nn > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p - sensor_data_b.p"

    err_p_rms_ln = np.max(np.abs(sensor_data_c_ln['p_rms']- sensor_data_b['p_rms'])) / np.max(np.abs(sensor_data_b['p_rms']))
    if (err_p_rms_ln > comparison_thresh):
        test_pass = False
    assert test_pass, "failure with sensor_data_c_ln.p - sensor_data_b.p"

    # err_ux_nn = np.max(np.abs(sensor_data_c_nn.ux- sensor_data_b.ux)) / np.max(np.abs(sensor_data_b.ux))
    # err_ux_ln = np.max(np.abs(sensor_data_c_ln.ux- sensor_data_b.ux)) / np.max(np.abs(sensor_data_b.ux))

    # err_uy_nn = np.max(np.abs(sensor_data_c_nn.uy- sensor_data_b.uy)) / np.max(np.abs(sensor_data_b.uy))
    # err_uy_ln = np.max(np.abs(sensor_data_c_ln.uy- sensor_data_b.uy)) / np.max(np.abs(sensor_data_b.uy))

    # err_ux_max_nn = np.max(np.abs(sensor_data_c_nn.ux_max- sensor_data_b.ux_max)) / np.max(np.abs(sensor_data_b.ux_max))
    # err_ux_max_ln = np.max(np.abs(sensor_data_c_ln.ux_max- sensor_data_b.ux_max)) / np.max(np.abs(sensor_data_b.ux_max))

    # err_uy_max_nn = np.max(np.abs(sensor_data_c_nn.uy_max- sensor_data_b.uy_max)) / np.max(np.abs(sensor_data_b.uy_max))
    # err_uy_max_ln = np.max(np.abs(sensor_data_c_ln.uy_max- sensor_data_b.uy_max)) / np.max(np.abs(sensor_data_b.uy_max))

    # err_ux_min_nn = np.max(np.abs(sensor_data_c_nn.ux_min- sensor_data_b.ux_min)) / np.max(np.abs(sensor_data_b.ux_min))
    # err_ux_min_ln = np.max(np.abs(sensor_data_c_ln.ux_min- sensor_data_b.ux_min)) / np.max(np.abs(sensor_data_b.ux_min))

    # err_uy_min_nn = np.max(np.abs(sensor_data_c_nn.uy_min- sensor_data_b.uy_min)) / np.max(np.abs(sensor_data_b.uy_min))
    # err_uy_min_ln = np.max(np.abs(sensor_data_c_ln.uy_min- sensor_data_b.uy_min)) / np.max(np.abs(sensor_data_b.uy_min))

    # err_ux_rms_nn = np.max(np.abs(sensor_data_c_nn.ux_rms- sensor_data_b.ux_rms)) / np.max(np.abs(sensor_data_b.ux_rms))
    # err_ux_rms_ln = np.max(np.abs(sensor_data_c_ln.ux_rms- sensor_data_b.ux_rms)) / np.max(np.abs(sensor_data_b.ux_rms))

    # err_uy_rms_nn = np.max(np.abs(sensor_data_c_nn.uy_rms- sensor_data_b.uy_rms)) / np.max(np.abs(sensor_data_b.uy_rms))
    # err_uy_rms_ln = np.max(np.abs(sensor_data_c_ln.uy_rms- sensor_data_b.uy_rms)) / np.max(np.abs(sensor_data_b.uy_rms))

    # err_ux_non_staggered_nn = np.max(np.abs(sensor_data_c_nn.ux_non_staggered- sensor_data_b.ux_non_staggered)) / np.max(np.abs(sensor_data_b.ux_non_staggered))
    # err_ux_non_staggered_ln = np.max(np.abs(sensor_data_c_ln.ux_non_staggered- sensor_data_b.ux_non_staggered)) / np.max(np.abs(sensor_data_b.ux_non_staggered))

    # err_uy_non_staggered_nn = np.max(np.abs(sensor_data_c_nn.uy_non_staggered- sensor_data_b.uy_non_staggered)) / np.max(np.abs(sensor_data_b.uy_non_staggered))
    # err_uy_non_staggered_ln = np.max(np.abs(sensor_data_c_ln.uy_non_staggered- sensor_data_b.uy_non_staggered)) / np.max(np.abs(sensor_data_b.uy_non_staggered))

    # err_Ix_nn = np.max(np.abs(sensor_data_c_nn['Ix']- sensor_data_b['Ix'])) / np.max(np.abs(sensor_data_b['Ix']))
    # err_Ix_ln = np.max(np.abs(sensor_data_c_ln['Ix']- sensor_data_b['Ix'])) / np.max(np.abs(sensor_data_b['Ix']))

    # err_Iy_nn = np.max(np.abs(sensor_data_c_nn['Iy']- sensor_data_b['Iy'])) / np.max(np.abs(sensor_data_b['Iy']))
    # err_Iy_ln = np.max(np.abs(sensor_data_c_ln['Iy']- sensor_data_b['Iy'])) / np.max(np.abs(sensor_data_b['Iy']))

    # err_Ix_avg_nn = np.max(np.abs(sensor_data_c_nn['Ix_avg']- sensor_data_b['Ix_avg'])) / np.max(np.abs(sensor_data_b['Ix_avg']))
    # err_Ix_avg_ln = np.max(np.abs(sensor_data_c_ln['Ix_avg']- sensor_data_b['Ix_avg'])) / np.max(np.abs(sensor_data_b['Ix_avg']))

    # err_Iy_avg_nn = np.max(np.abs(sensor_data_c_nn['Iy_avg']- sensor_data_b['Iy_avg'])) / np.max(np.abs(sensor_data_b['Iy_avg']))
    # err_Iy_avg_ln = np.max(np.abs(sensor_data_c_ln['Iy_avg']- sensor_data_b['Iy_avg'])) / np.max(np.abs(sensor_data_b['Iy_avg']))

    # # check for test pass
    # if (err_p_nn > comparison_thresh) || ...
    #         (err_p_ln > comparison_thresh) || ...
    #         (err_p_max_nn > comparison_thresh) || ...
    #         (err_p_max_ln > comparison_thresh) || ...
    #         (err_p_min_nn > comparison_thresh) || ...
    #         (err_p_min_ln > comparison_thresh) || ...
    #         (err_p_rms_nn > comparison_thresh) || ...
    #         (err_p_rms_ln > comparison_thresh) || ...
    #         (err_ux_nn > comparison_thresh) || ...
    #         (err_ux_ln > comparison_thresh) || ...
    #         (err_ux_ln > comparison_thresh) || ...
    #         (err_ux_max_nn > comparison_thresh) || ...
    #         (err_ux_max_ln > comparison_thresh) || ...
    #         (err_ux_min_nn > comparison_thresh) || ...
    #         (err_ux_min_ln > comparison_thresh) || ...
    #         (err_ux_rms_nn > comparison_thresh) || ...
    #         (err_ux_rms_ln > comparison_thresh) || ...
    #         (err_ux_non_staggered_nn > comparison_thresh) || ...
    #         (err_ux_non_staggered_ln > comparison_thresh) || ...
    #         (err_uy_nn > comparison_thresh) || ...
    #         (err_uy_ln > comparison_thresh) || ...
    #         (err_uy_max_nn > comparison_thresh) || ...
    #         (err_uy_max_ln > comparison_thresh) || ...
    #         (err_uy_min_nn > comparison_thresh) || ...
    #         (err_uy_min_ln > comparison_thresh) || ...
    #         (err_uy_rms_nn > comparison_thresh) || ...
    #         (err_uy_rms_ln > comparison_thresh) || ...
    #         (err_uy_non_staggered_nn > comparison_thresh) || ...
    #         (err_uy_non_staggered_ln > comparison_thresh) || ...
    #         (err_Ix_nn > comparison_thresh) || ...
    #         (err_Ix_ln > comparison_thresh) || ...
    #         (err_Ix_avg_nn > comparison_thresh) || ...
    #         (err_Ix_avg_ln > comparison_thresh) || ...
    #         (err_Iy_nn > comparison_thresh) || ...
    #         (err_Iy_ln > comparison_thresh) || ...
    #         (err_Iy_avg_nn > comparison_thresh) || ...
    #         (err_Iy_avg_ln > comparison_thresh)
    #     test_pass = False




