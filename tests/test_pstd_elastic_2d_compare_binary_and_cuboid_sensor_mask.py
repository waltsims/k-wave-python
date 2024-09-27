"""
Unit test to compare the simulation results using a labelled and binary source mask
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
from kwave.utils.mapgen import make_disc

@pytest.mark.skip(reason="2D not ready")
def test_pstd_elastic_2d_compare_binary_and_cuboid_sensor_mask():

    # set pass variable
    test_pass: bool = True

    # set additional literals to give further permutations of the test
    comparison_threshold: float = 1e-15
    pml_inside: bool = True

    # create the computational grid
    Nx: int = 128           # number of grid points in the x direction
    Ny: int = 128           # number of grid points in the y direction
    dx: float = 0.1e-3             # grid point spacing in the x direction [m]
    dy: float = 0.1e-3             # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the upper layer of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny))  # [m/s]
    sound_speed_shear = np.zeros((Nx, Ny))                # [m/s]
    density  = 1000.0 * np.ones((Nx, Ny))                 # [kg/m^3]

    medium = kWaveMedium(sound_speed_compression,
                         density=density,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear)

    # define the properties of the lower layer of the propagation medium
    medium.sound_speed_compression[Nx // 2 - 1:, :] = 2000.0  # [m/s]
    medium.sound_speed_shear[Nx // 2 - 1:, :] = 800.0         # [m/s]
    medium.density[Nx // 2 - 1:, :] = 1200.0                  # [kg/m^3]

    # create initial pressure distribution using makeDisc
    disc_magnitude = 5.0 # [Pa]
    disc_x_pos: int = 30    # [grid points]
    disc_y_pos: int = 64    # [grid points]
    disc_radius: int = 5    # [grid points]

    source = kSource()
    source.p0 = disc_magnitude * make_disc(Vector([Nx, Ny]), Vector([disc_x_pos, disc_y_pos]), disc_radius)

    # define list of cuboid corners using two intersecting cuboids
    cuboid_corners = [[40, 10, 90, 65], [10, 60, 50, 70]]

    sensor = kSensor()
    sensor.mask = cuboid_corners

    # set the variables to record
    sensor.record = ['p', 'p_max', 'p_min', 'p_rms', 'p_max_all', 'p_min_all', 'p_final',
                     'u', 'u_max', 'u_min', 'u_rms', 'u_max_all', 'u_min_all', 'u_final',
                     'u_non_staggered', 'I', 'I_avg']

    # run the simulation as normal
    simulation_options_cuboids = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                   pml_inside=pml_inside,
                                                   kelvin_voigt_model=False)

    sensor_data_cuboids = pstd_elastic_2d(deepcopy(kgrid),
                                          deepcopy(medium),
                                          deepcopy(source),
                                          deepcopy(sensor),
                                          deepcopy(simulation_options_cuboids))

    # create a binary mask for display from the list of corners
    sensor.mask = np.zeros(np.shape(kgrid.k))

    cuboid_index: int = 0
    sensor.mask[cuboid_corners[0, cuboid_index]:cuboid_corners[2, cuboid_index] + 1,
                cuboid_corners[1, cuboid_index]:cuboid_corners[3, cuboid_index] + 1] = 1

    # run the simulation
    simulation_options_comp1 = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                 pml_inside=pml_inside,
                                                 kelvin_voigt_model=False)

    sensor_data_comp1 = pstd_elastic_2d(deepcopy(kgrid),
                                        deepcopy(medium),
                                        deepcopy(source),
                                        deepcopy(sensor),
                                        deepcopy(simulation_options_comp1))

    # compute the error from the first cuboid
    L_inf_p = np.max(np.abs(sensor_data_cuboids[cuboid_index].p - sensor_data_comp1.p)) / np.max(np.abs(sensor_data_comp1.p))
    L_inf_p_max = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_max - sensor_data_comp1.p_max)) / np.max(np.abs(sensor_data_comp1.p_max))
    L_inf_p_min = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_min - sensor_data_comp1.p_min)) / np.max(np.abs(sensor_data_comp1.p_min))
    L_inf_p_rms = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_rms - sensor_data_comp1.p_rms)) / np.max(np.abs(sensor_data_comp1.p_rms))

    L_inf_ux = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux - sensor_data_comp1.ux)) / np.max(np.abs(sensor_data_comp1.ux))
    L_inf_ux_max = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_max - sensor_data_comp1.ux_max)) / np.max(np.abs(sensor_data_comp1.ux_max))
    L_inf_ux_min = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_min - sensor_data_comp1.ux_min)) / np.max(np.abs(sensor_data_comp1.ux_min))
    L_inf_ux_rms = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_rms - sensor_data_comp1.ux_rms)) / np.max(np.abs(sensor_data_comp1.ux_rms))

    L_inf_uy = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy - sensor_data_comp1.uy)) / np.max(np.abs(sensor_data_comp1.uy))
    L_inf_uy_max = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_max - sensor_data_comp1.uy_max)) / np.max(np.abs(sensor_data_comp1.uy_max))
    L_inf_uy_min = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_min - sensor_data_comp1.uy_min)) / np.max(np.abs(sensor_data_comp1.uy_min))
    L_inf_uy_rms = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_rms - sensor_data_comp1.uy_rms)) / np.max(np.abs(sensor_data_comp1.uy_rms))

    # compute the error from the total variables
    L_inf_p_max_all  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_max_all - sensor_data_comp1.p_max_all))  / np.max(np.abs(sensor_data_comp1.p_max_all))
    L_inf_ux_max_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_max_all - sensor_data_comp1.ux_max_all)) / np.max(np.abs(sensor_data_comp1.ux_max_all))
    L_inf_uy_max_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_max_all - sensor_data_comp1.uy_max_all)) / np.max(np.abs(sensor_data_comp1.uy_max_all))

    L_inf_p_min_all  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_min_all  - sensor_data_comp1.p_min_all))  / np.max(np.abs(sensor_data_comp1.p_min_all))
    L_inf_ux_min_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_min_all - sensor_data_comp1.ux_min_all)) / np.max(np.abs(sensor_data_comp1.ux_min_all))
    L_inf_uy_min_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_min_all - sensor_data_comp1.uy_min_all)) / np.max(np.abs(sensor_data_comp1.uy_min_all))

    L_inf_p_final  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_final  - sensor_data_comp1.p_final))  / np.max(np.abs(sensor_data_comp1.p_final))
    L_inf_ux_final = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_final - sensor_data_comp1.ux_final)) / np.max(np.abs(sensor_data_comp1.ux_final))
    L_inf_uy_final = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_final - sensor_data_comp1.uy_final)) / np.max(np.abs(sensor_data_comp1.uy_final))

    # get maximum error
    L_inf_max = np.max([L_inf_p, L_inf_p_max, L_inf_p_min, L_inf_p_rms, L_inf_ux,
                        L_inf_ux_max, L_inf_ux_min, L_inf_ux_rms, L_inf_uy, L_inf_uy_max,
                        L_inf_uy_min, L_inf_uy_rms, L_inf_p_max_all, L_inf_ux_max_all,
                        L_inf_uy_max_all, L_inf_p_min_all, L_inf_ux_min_all, L_inf_uy_min_all,
                        L_inf_p_final, L_inf_ux_final, L_inf_uy_final])

    # compute pass
    if (L_inf_max > comparison_threshold):
        test_pass = False
    assert test_pass, "fails here"

    # ------------------------

    # create a binary mask for display from the list of corners
    sensor.mask = np.zeros(np.shape(kgrid.k))

    cuboid_index = 1
    sensor.mask[cuboid_corners[0, cuboid_index]:cuboid_corners[2, cuboid_index],
                cuboid_corners[1, cuboid_index]:cuboid_corners[3, cuboid_index]] = 1

    # run the simulation
    simulation_options_comp2 = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                 pml_inside=pml_inside,
                                                 kelvin_voigt_model=False)

    sensor_data_comp2 = pstd_elastic_2d(deepcopy(kgrid),
                                        deepcopy(medium),
                                        deepcopy(source),
                                        deepcopy(sensor),
                                        deepcopy(simulation_options_comp2))

    # compute the error from the second cuboid
    L_inf_p = np.max(np.abs(sensor_data_cuboids[cuboid_index].p - sensor_data_comp2.p)) / np.max(np.abs(sensor_data_comp2.p))
    L_inf_p_max = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_max - sensor_data_comp2.p_max)) / np.max(np.abs(sensor_data_comp2.p_max))
    L_inf_p_min = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_min - sensor_data_comp2.p_min)) / np.max(np.abs(sensor_data_comp2.p_min))
    L_inf_p_rms = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_rms - sensor_data_comp2.p_rms)) / np.max(np.abs(sensor_data_comp2.p_rms))

    L_inf_ux = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux - sensor_data_comp2.ux)) / np.max(np.abs(sensor_data_comp2.ux))
    L_inf_ux_max = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_max - sensor_data_comp2.ux_max)) / np.max(np.abs(sensor_data_comp2.ux_max))
    L_inf_ux_min = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_min - sensor_data_comp2.ux_min)) / np.max(np.abs(sensor_data_comp2.ux_min))
    L_inf_ux_rms = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_rms - sensor_data_comp2.ux_rms)) / np.max(np.abs(sensor_data_comp2.ux_rms))

    L_inf_uy = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy - sensor_data_comp2.uy)) / np.max(np.abs(sensor_data_comp2.uy))
    L_inf_uy_max = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_max - sensor_data_comp2.uy_max)) / np.max(np.abs(sensor_data_comp2.uy_max))
    L_inf_uy_min = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_min - sensor_data_comp2.uy_min)) / np.max(np.abs(sensor_data_comp2.uy_min))
    L_inf_uy_rms = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_rms - sensor_data_comp2.uy_rms)) / np.max(np.abs(sensor_data_comp2.uy_rms))

    # get maximum error
    L_inf_max = np.max([L_inf_p, L_inf_p_max, L_inf_p_min, L_inf_p_rms,
                        L_inf_ux, L_inf_ux_max, L_inf_ux_min, L_inf_ux_rms,
                        L_inf_uy, L_inf_uy_max, L_inf_uy_min, L_inf_uy_rms])

    # compute pass
    if (L_inf_max > comparison_threshold):
        test_pass = False

    assert test_pass, "fails at this point"
