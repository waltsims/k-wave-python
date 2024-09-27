"""
Unit test to compare the simulation results using a binary and cuboid sensor mask
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
from kwave.reconstruction.beamform import focus

def test_pstd_elastic_3d_compare_binary_and_cuboid_sensor_mask():

    # set pass variable
    test_pass: bool = True

    # set additional literals to give further permutations of the test
    COMPARISON_THRESH: float = 1e-15
    PML_INSIDE: bool = True

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx: int = 64              # number of grid points in the x direction
    Ny: int = 64              # number of grid points in the y direction
    Nz: int = 64              # number of grid points in the z direction
    dx: float = 0.1e-3        # grid point spacing in the x direction [m]
    dy: float = 0.1e-3        # grid point spacing in the y direction [m]
    dz: float = 0.1e-3        # grid point spacing in the z direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the upper layer of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny, Nz))  # [m/s]
    sound_speed_shear = np.zeros((Nx, Ny, Nz))                # [m/s]
    density = 1000.0 * np.ones((Nx, Ny, Nz))                  # [kg/m^3]

    # define the properties of the lower layer of the propagation medium
    sound_speed_compression[Nx // 2 - 1:, :, :] = 2000.0  # [m/s]
    sound_speed_shear[Nx // 2 - 1:, :, :] = 800.0         # [m/s]
    density[Nx // 2 - 1:, :, :] = 1200.0                  # [kg/m^3]

    medium = kWaveMedium(sound_speed=sound_speed_compression,
                         density=density,
                         sound_speed_shear=sound_speed_shear,
                         sound_speed_compression=sound_speed_compression)

    # create the time array
    cfl = 0.1
    t_end = 5e-6
    kgrid.makeTime(np.max(medium.sound_speed_compression), cfl, t_end)

    source = kSource()

    # define source mask to be a square piston
    source_x_pos: int = 10      # [grid points]
    source_radius: int = 15     # [grid points]
    source.u_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    source.u_mask[source_x_pos,
                  Ny // 2 - source_radius:Ny // 2 + source_radius,
                  Nz // 2 - source_radius:Nz // 2 + source_radius] = True

    # define source to be a velocity source
    source_freq = 2e6      # [Hz]
    source_cycles = 3
    source_mag = 1e-6
    fs = 1.0 / kgrid.dt
    source.ux = source_mag * tone_burst(fs, source_freq, source_cycles)

    # set source focus
    source.ux = focus(kgrid, deepcopy(source.ux), deepcopy(source.u_mask), Vector([0.0, 0.0, 0.0]), 1500.0)

    # define list of cuboid corners using two intersecting cuboids

    # cuboid_corners = np.array([[20, 10],
    #                            [40, 35],
    #                            [30, 30],
    #                            [30, 25],
    #                            [50, 42],
    #                            [40, 40]], dtype=int)

    cuboid_corners = np.transpose(np.array([[20, 40, 30, 30, 50, 40], [10, 35, 30, 25, 42, 40]], dtype=int)) - int(1)

    # create sensor
    sensor = kSensor()

    #create sensor mask
    sensor.mask = cuboid_corners

    # set the variables to record
    sensor.record = ['p', 'p_max', 'p_min', 'p_rms', 'p_max_all', 'p_min_all', 'p_final',
                     'u', 'u_max', 'u_min', 'u_rms', 'u_max_all', 'u_min_all', 'u_final',
                     'I', 'I_avg']

    # run the simulation as normal
    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                   pml_inside=PML_INSIDE,
                                                   kelvin_voigt_model=False)
    # run the simulation
    sensor_data_cuboids = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                          medium=deepcopy(medium),
                                          source=deepcopy(source),
                                          sensor=deepcopy(sensor),
                                          simulation_options=deepcopy(simulation_options))

    # create a binary mask for display from the list of corners
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    cuboid_index: int = 0
    sensor.mask[cuboid_corners[0, cuboid_index]:cuboid_corners[3, cuboid_index] + 1,
                cuboid_corners[1, cuboid_index]:cuboid_corners[4, cuboid_index] + 1,
                cuboid_corners[2, cuboid_index]:cuboid_corners[5, cuboid_index] + 1] = True

    # run the simulation
    sensor_data_comp1 = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                        medium=deepcopy(medium),
                                        source=deepcopy(source),
                                        sensor=deepcopy(sensor),
                                        simulation_options=deepcopy(simulation_options))

    # print(np.shape(sensor_data_comp1.p))

    # compute the error from the first cuboid
    L_inf_p      = np.max(np.abs(sensor_data_cuboids[cuboid_index].p -
                          np.reshape(sensor_data_comp1.p, np.shape(sensor_data_cuboids[cuboid_index].p), order='F') ))     / np.max(np.abs(sensor_data_comp1.p))
    # L_inf_p_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_max  - sensor_data_comp1.p_max)) / np.max(np.abs(sensor_data_comp1.p_max))
    # L_inf_p_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_min  - sensor_data_comp1.p_min)) / np.max(np.abs(sensor_data_comp1.p_min))
    # L_inf_p_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_rms  - sensor_data_comp1.p_rms)) / np.max(np.abs(sensor_data_comp1.p_rms))

    # L_inf_ux      = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux      - sensor_data_comp1.ux))     / np.max(np.abs(sensor_data_comp1.ux))
    # L_inf_ux_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_max  - sensor_data_comp1.ux_max)) / np.max(np.abs(sensor_data_comp1.ux_max))
    # L_inf_ux_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_min  - sensor_data_comp1.ux_min)) / np.max(np.abs(sensor_data_comp1.ux_min))
    # L_inf_ux_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_rms  - sensor_data_comp1.ux_rms)) / np.max(np.abs(sensor_data_comp1.ux_rms))

    # L_inf_uy      = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy      - sensor_data_comp1.uy))     / np.max(np.abs(sensor_data_comp1.uy))
    # L_inf_uy_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_max  - sensor_data_comp1.uy_max)) / np.max(np.abs(sensor_data_comp1.uy_max))
    # L_inf_uy_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_min  - sensor_data_comp1.uy_min)) / np.max(np.abs(sensor_data_comp1.uy_min))
    # L_inf_uy_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_rms  - sensor_data_comp1.uy_rms)) / np.max(np.abs(sensor_data_comp1.uy_rms))

    # L_inf_uz      = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz      - sensor_data_comp1.uz))     / np.max(np.abs(sensor_data_comp1.uz))
    # L_inf_uz_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_max  - sensor_data_comp1.uz_max)) / np.max(np.abs(sensor_data_comp1.uz_max))
    # L_inf_uz_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_min  - sensor_data_comp1.uz_min)) / np.max(np.abs(sensor_data_comp1.uz_min))
    # L_inf_uz_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_rms  - sensor_data_comp1.uz_rms)) / np.max(np.abs(sensor_data_comp1.uz_rms))

    # # compute the error from the total variables
    # L_inf_p_max_all  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_max_all  - sensor_data_comp1.p_max_all))  / np.max(np.abs(sensor_data_comp1.p_max_all))
    # L_inf_ux_max_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_max_all - sensor_data_comp1.ux_max_all)) / np.max(np.abs(sensor_data_comp1.ux_max_all))
    # L_inf_uy_max_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_max_all - sensor_data_comp1.uy_max_all)) / np.max(np.abs(sensor_data_comp1.uy_max_all))
    # L_inf_uz_max_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_max_all - sensor_data_comp1.uz_max_all)) / np.max(np.abs(sensor_data_comp1.uz_max_all))

    # L_inf_p_min_all  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_min_all  - sensor_data_comp1.p_min_all))  / np.max(np.abs(sensor_data_comp1.p_min_all))
    # L_inf_ux_min_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_min_all - sensor_data_comp1.ux_min_all)) / np.max(np.abs(sensor_data_comp1.ux_min_all))
    # L_inf_uy_min_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_min_all - sensor_data_comp1.uy_min_all)) / np.max(np.abs(sensor_data_comp1.uy_min_all))
    # L_inf_uz_min_all = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_min_all - sensor_data_comp1.uz_min_all)) / np.max(np.abs(sensor_data_comp1.uz_min_all))

    # L_inf_p_final  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_final  - sensor_data_comp1.p_final))  / np.max(np.abs(sensor_data_comp1.p_final))
    # L_inf_ux_final = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_final - sensor_data_comp1.ux_final)) / np.max(np.abs(sensor_data_comp1.ux_final))
    # L_inf_uy_final = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_final - sensor_data_comp1.uy_final)) / np.max(np.abs(sensor_data_comp1.uy_final))
    # L_inf_uz_final = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_final - sensor_data_comp1.uz_final)) / np.max(np.abs(sensor_data_comp1.uz_final))

    # get maximum error
    L_inf_max = np.max([L_inf_p, #L_inf_p_max, L_inf_p_min, L_inf_p_rms,
        # L_inf_ux, L_inf_ux_max, L_inf_ux_min, L_inf_ux_rms,
        # L_inf_uy, L_inf_uy_max, L_inf_uy_min, L_inf_uy_rms,
        # L_inf_uz, L_inf_uz_max, L_inf_uz_min, L_inf_uz_rms,
        # L_inf_p_max_all, L_inf_ux_max_all, L_inf_uy_max_all, L_inf_uz_max_all,
        # L_inf_p_min_all, L_inf_ux_min_all, L_inf_uy_min_all, L_inf_uz_min_all,
        # L_inf_p_final, L_inf_ux_final, L_inf_uy_final, L_inf_uz_final
        ])

    # compute pass
    if (L_inf_max > COMPARISON_THRESH):
        test_pass = False

    assert test_pass, "fails on first cuboids"

    # create a binary mask for display from the list of corners

    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    cuboid_index: int = 1
    sensor.mask[cuboid_corners[0, cuboid_index]:cuboid_corners[3, cuboid_index] + 1,
                cuboid_corners[1, cuboid_index]:cuboid_corners[4, cuboid_index] + 1,
                cuboid_corners[2, cuboid_index]:cuboid_corners[5, cuboid_index] + 1] = True

    # run the simulation
    sensor_data_comp2 = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                        medium=deepcopy(medium),
                                        source=deepcopy(source),
                                        sensor=deepcopy(sensor),
                                        simulation_options=deepcopy(simulation_options))

    # compute the error from the second cuboid
    L_inf_p      = np.max(np.abs(sensor_data_cuboids[cuboid_index].p -
                          np.reshape(sensor_data_comp2.p, np.shape(sensor_data_cuboids[cuboid_index].p), order='F') )) / np.max(np.abs(sensor_data_comp2.p))

    # L_inf_p_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_max  - sensor_data_comp2.p_max)) / np.max(np.abs(sensor_data_comp2.p_max))
    # L_inf_p_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_min  - sensor_data_comp2.p_min)) / np.max(np.abs(sensor_data_comp2.p_min))
    # L_inf_p_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].p_rms  - sensor_data_comp2.p_rms)) / np.max(np.abs(sensor_data_comp2.p_rms))

    # L_inf_ux      = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux      - sensor_data_comp2.ux))     / np.max(np.abs(sensor_data_comp2.ux))
    # L_inf_ux_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_max  - sensor_data_comp2.ux_max)) / np.max(np.abs(sensor_data_comp2.ux_max))
    # L_inf_ux_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_min  - sensor_data_comp2.ux_min)) / np.max(np.abs(sensor_data_comp2.ux_min))
    # L_inf_ux_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].ux_rms  - sensor_data_comp2.ux_rms)) / np.max(np.abs(sensor_data_comp2.ux_rms))

    # L_inf_uy      = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy      - sensor_data_comp2.uy))     / np.max(np.abs(sensor_data_comp2.uy))
    # L_inf_uy_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_max  - sensor_data_comp2.uy_max)) / np.max(np.abs(sensor_data_comp2.uy_max))
    # L_inf_uy_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_min  - sensor_data_comp2.uy_min)) / np.max(np.abs(sensor_data_comp2.uy_min))
    # L_inf_uy_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uy_rms  - sensor_data_comp2.uy_rms)) / np.max(np.abs(sensor_data_comp2.uy_rms))

    # L_inf_uz      = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz      - sensor_data_comp2.uz))     / np.max(np.abs(sensor_data_comp2.uz))
    # L_inf_uz_max  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_max  - sensor_data_comp2.uz_max)) / np.max(np.abs(sensor_data_comp2.uz_max))
    # L_inf_uz_min  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_min  - sensor_data_comp2.uz_min)) / np.max(np.abs(sensor_data_comp2.uz_min))
    # L_inf_uz_rms  = np.max(np.abs(sensor_data_cuboids[cuboid_index].uz_rms  - sensor_data_comp2.uz_rms)) / np.max(np.abs(sensor_data_comp2.uz_rms))

    # get maximum error
    L_inf_max = np.max([L_inf_p#, L_inf_p_max, L_inf_p_min, L_inf_p_rms,
        # L_inf_ux, L_inf_ux_max, L_inf_ux_min, L_inf_ux_rms,
        # L_inf_uy, L_inf_uy_max, L_inf_uy_min, L_inf_uy_rms,
        # L_inf_uz, L_inf_uz_max, L_inf_uz_min, L_inf_uz_rms
        ])

    # compute pass
    if (L_inf_max > COMPARISON_THRESH):
        test_pass = False

    assert test_pass, "fails on second cuboids"

# # =========================================================================
# # PLOT COMPARISONS
# # =========================================================================

# if plot_comparisons

#     # plot the simulated sensor data
#     figure;
#     subplot(3, 2, 1);
#     imagesc(reshape(sensor_data_cuboids(1).p, [], size(sensor_data_comp1.p, 2)), [-1, 1]);
#     colormap(getColorMap);
#     ylabel('Sensor Position');
#     xlabel('Time Step');
#     title('Cuboid 1');
#     colorbar;

#     subplot(3, 2, 2);
#     imagesc(reshape(sensor_data_cuboids(2).p, [], size(sensor_data_comp1.p, 2)), [-1, 1]);
#     colormap(getColorMap);
#     ylabel('Sensor Position');
#     xlabel('Time Step');
#     title('Cuboid 2');
#     colorbar;

#     subplot(3, 2, 3);
#     imagesc(sensor_data_comp1.p, [-1, 1]);
#     colormap(getColorMap);
#     ylabel('Sensor Position');
#     xlabel('Time Step');
#     title('Cuboid 1 - Comparison');
#     colorbar;

#     subplot(3, 2, 4);
#     imagesc(sensor_data_comp2.p, [-1, 1]);
#     colormap(getColorMap);
#     ylabel('Sensor Position');
#     xlabel('Time Step');
#     title('Cuboid 2 - Comparison');
#     colorbar;

#     subplot(3, 2, 5);
#     imagesc(reshape(sensor_data_cuboids(1).p, [], size(sensor_data_comp1.p, 2)) - sensor_data_comp1.p, [-1, 1]);
#     colormap(getColorMap);
#     ylabel('Sensor Position');
#     xlabel('Time Step');
#     title('Cuboid 1 - Difference');
#     colorbar;

#     subplot(3, 2, 6);
#     imagesc(reshape(sensor_data_cuboids(2).p, [], size(sensor_data_comp1.p, 2)) - sensor_data_comp2.p, [-1, 1]);
#     colormap(getColorMap);
#     ylabel('Sensor Position');
#     xlabel('Time Step');
#     title('Cuboid 2 - Difference');
#     colorbar;

# end