"""
Unit test to compare cartesian and binary sensor masks
"""

import numpy as np
from copy import deepcopy
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic3D import pstd_elastic_3d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.conversion import cart2grid
from kwave.utils.mapgen import make_sphere, make_multi_bowl
from kwave.utils.signals import reorder_binary_sensor_data
from kwave.utils.filters import filter_time_series

@pytest.mark.skip(reason="not ready")
def test_pstd_elastic_3d_compare_binary_and_cartesian_sensor_mask():

    # set comparison threshold
    comparison_thresh: float = 1e-14

    # set pass variable
    test_pass: bool = True

    # create the computational grid
    Nx: int = 48
    Ny: int = 48
    Nz: int = 48
    dx: float = 25e-3 / float(Nx)
    dy: float = 25e-3 / float(Ny)
    dz: float = 25e-3 / float(Nz)
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))



    # define the properties of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny, Nz))  # [m/s]
    sound_speed_compression[Nx // 2 - 1:, :, :] = 2000.0

    sound_speed_shear = np.zeros((Nx, Ny, Nz))   # [m/s]
    sound_speed_shear[Nx // 2 - 1:, :, :] = 1400

    density = 1000.0 * np.ones((Nx, Ny, Nz))
    density[Nx // 2 - 1:, :, :] = 1200.0

    medium = kWaveMedium(sound_speed_compression,
                         density=density,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear)

    # create the time array using default CFL condition
    cfl: float = 0.1
    kgrid.makeTime(medium.sound_speed_compression, cfl=cfl)

    # define source mask
    # source = kSource()
    # p0 = np.zeros((Nx, Ny, Nz), dtype=bool)
    # p0[7,
    #    Ny // 4 - 1:3 * Ny // 4,
    #    Nz // 4 - 1:3 * Nz // 4] = True
    # source.p0 = p0

    source_freq_0 = 1e6          # [Hz]
    source_mag_0 = 0.5           # [Pa]
    source_0 = source_mag_0 * np.sin(2.0 * np.pi * source_freq_0 * np.squeeze(kgrid.t_array))
    source_0 = filter_time_series(kgrid, medium, deepcopy(source_0))

    source_freq_1 = 3e6          # [Hz]
    source_mag_1 = 0.8           # [Pa]
    source_1 = source_mag_1 * np.sin(2.0 * np.pi * source_freq_1 * np.squeeze(kgrid.t_array))
    source_1 = filter_time_series(kgrid, medium, deepcopy(source_1))

    # assemble sources
    labelled_sources = np.zeros((2, kgrid.Nt))
    labelled_sources[0, :] = np.squeeze(source_0)
    labelled_sources[1, :] = np.squeeze(source_1)

   # define multiple curved transducer elements
    bowl_pos = np.array([(19.0, 19.0, Nz / 2.0 - 1.0), (48.0, 48.0, Nz / 2.0 - 1.0)])
    bowl_radius = np.array([20.0, 15.0])
    bowl_diameter = np.array([int(15), int(21)], dtype=np.uint8)
    bowl_focus = np.array([(int(31), int(31), int(31))], dtype=np.uint8)

    binary_mask, labelled_mask = make_multi_bowl(Vector([Nx, Ny, Nz]), bowl_pos, bowl_radius, bowl_diameter, bowl_focus)

    # create ksource object
    source = kSource()

    # source mask is from the labelled mask
    source.s_mask = deepcopy(labelled_mask)

    # assign sources from labelled source
    source.sxx = deepcopy(labelled_sources)
    source.syy = deepcopy(labelled_sources)
    source.szz = deepcopy(labelled_sources)
    source.sxy = deepcopy(labelled_sources)
    source.sxz = deepcopy(labelled_sources)
    source.syz = deepcopy(labelled_sources)


    sensor = kSensor()

    # define Cartesian sensor points using points exactly on the grid
    sphere_mask = make_sphere(Vector([Nx, Ny, Nz]), radius=10)
    x_points = kgrid.x[sphere_mask == 1]
    y_points = kgrid.y[sphere_mask == 1]
    z_points = kgrid.z[sphere_mask == 1]
    sensor.mask = np.vstack((x_points, y_points, z_points))

    # record all output variables
    sensor.record = ['p', 'p_max', 'p_min', 'p_rms', 'u', 'u_max', 'u_min', 'u_rms',
                     'u_non_staggered', 'I', 'I_avg']

    # run the simulation as normal
    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           pml_size=6,
                                           kelvin_voigt_model=False)

    sensor_data_c_ln = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                       source=deepcopy(source),
                                       sensor=deepcopy(sensor),
                                       medium=deepcopy(medium),
                                       simulation_options=deepcopy(simulation_options))

    # run the simulation using nearest-neighbour interpolation
    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           pml_size=6,
                                           cart_interp='nearest',
                                           kelvin_voigt_model=False)

    sensor_data_c_nn = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                       source=deepcopy(source),
                                       sensor=deepcopy(sensor),
                                       medium=deepcopy(medium),
                                       simulation_options=deepcopy(simulation_options))

    # convert sensor mask
    _, _, reorder_index  = cart2grid(kgrid, sensor.mask)

    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           pml_size=int(6),
                                           kelvin_voigt_model=False)

    # run the simulation again
    sensor_data_b = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                    source=deepcopy(source),
                                    sensor=deepcopy(sensor),
                                    medium=deepcopy(medium),
                                    simulation_options=deepcopy(simulation_options))

    # reorder the binary sensor data
    sensor_data_b.p                 = reorder_binary_sensor_data(sensor_data_b.p, reorder_index)
    # sensor_data_b.p_max             = reorder_binary_sensor_data(sensor_data_b.p_max, reorder_index)
    # sensor_data_b.p_min             = reorder_binary_sensor_data(sensor_data_b.p_min, reorder_index)
    # sensor_data_b.p_rms             = reorder_binary_sensor_data(sensor_data_b.p_rms, reorder_index)
    # sensor_data_b.ux                = reorder_binary_sensor_data(sensor_data_b.ux, reorder_index)
    # sensor_data_b.uy                = reorder_binary_sensor_data(sensor_data_b.uy, reorder_index)
    # sensor_data_b.uz                = reorder_binary_sensor_data(sensor_data_b.uz, reorder_index)
    # sensor_data_b.ux_max            = reorder_binary_sensor_data(sensor_data_b.ux_max, reorder_index)
    # sensor_data_b.uy_max            = reorder_binary_sensor_data(sensor_data_b.uy_max, reorder_index)
    # sensor_data_b.uz_max            = reorder_binary_sensor_data(sensor_data_b.uz_max, reorder_index)
    # sensor_data_b.ux_min            = reorder_binary_sensor_data(sensor_data_b.ux_min, reorder_index)
    # sensor_data_b.uy_min            = reorder_binary_sensor_data(sensor_data_b.uy_min, reorder_index)
    # sensor_data_b.uz_min            = reorder_binary_sensor_data(sensor_data_b.uz_min, reorder_index)
    # sensor_data_b.ux_rms            = reorder_binary_sensor_data(sensor_data_b.ux_rms, reorder_index)
    # sensor_data_b.uy_rms            = reorder_binary_sensor_data(sensor_data_b.uy_rms, reorder_index)
    # sensor_data_b.uz_rms            = reorder_binary_sensor_data(sensor_data_b.uz_rms, reorder_index)
    # sensor_data_b.ux_non_staggered	= reorder_binary_sensor_data(sensor_data_b.ux_non_staggered, reorder_index)
    # sensor_data_b.uy_non_staggered  = reorder_binary_sensor_data(sensor_data_b.uy_non_staggered, reorder_index)
    # sensor_data_b.uz_non_staggered  = reorder_binary_sensor_data(sensor_data_b.uz_non_staggered, reorder_index)
    # sensor_data_b.Ix                = reorder_binary_sensor_data(sensor_data_b.Ix, reorder_index)
    # sensor_data_b.Iy                = reorder_binary_sensor_data(sensor_data_b.Iy, reorder_index)
    # sensor_data_b.Iz                = reorder_binary_sensor_data(sensor_data_b.Iz, reorder_index)
    # sensor_data_b.Ix_avg            = reorder_binary_sensor_data(sensor_data_b.Ix_avg, reorder_index)
    # sensor_data_b.Iy_avg            = reorder_binary_sensor_data(sensor_data_b.Iy_avg, reorder_index)
    # sensor_data_b.Iz_avg            = reorder_binary_sensor_data(sensor_data_b.Iz_avg, reorder_index)

    # compute errors
    err_p_nn = np.max(np.abs(sensor_data_c_nn.p - sensor_data_b.p)) / np.max(np.abs(sensor_data_b.p))
    err_p_ln = np.max(np.abs(sensor_data_c_ln.p - sensor_data_b.p)) / np.max(np.abs(sensor_data_b.p))

    # err_p_max_nn = np.max(np.abs(sensor_data_c_nn.p_max - sensor_data_b.p_max)) / np.max(np.abs(sensor_data_b.p_max))
    # err_p_max_ln = np.max(np.abs(sensor_data_c_ln.p_max - sensor_data_b.p_max)) / np.max(np.abs(sensor_data_b.p_max))

    # err_p_min_nn = np.max(np.abs(sensor_data_c_nn.p_min - sensor_data_b.p_min)) / np.max(np.abs(sensor_data_b.p_min))
    # err_p_min_ln = np.max(np.abs(sensor_data_c_ln.p_min - sensor_data_b.p_min)) / np.max(np.abs(sensor_data_b.p_min))

    # err_p_rms_nn = np.max(np.abs(sensor_data_c_nn.p_rms - sensor_data_b.p_rms)) / np.max(np.abs(sensor_data_b.p_rms))
    # err_p_rms_ln = np.max(np.abs(sensor_data_c_ln.p_rms - sensor_data_b.p_rms)) / np.max(np.abs(sensor_data_b.p_rms))

    # err_ux_nn = np.max(np.abs(sensor_data_c_nn.ux - sensor_data_b.ux)) / np.max(np.abs(sensor_data_b.ux))
    # err_ux_ln = np.max(np.abs(sensor_data_c_ln.ux - sensor_data_b.ux)) / np.max(np.abs(sensor_data_b.ux))

    # err_uy_nn = np.max(np.abs(sensor_data_c_nn.uy - sensor_data_b.uy)) / np.max(np.abs(sensor_data_b.uy))
    # err_uy_ln = np.max(np.abs(sensor_data_c_ln.uy - sensor_data_b.uy)) / np.max(np.abs(sensor_data_b.uy))

    # err_uz_nn = np.max(np.abs(sensor_data_c_nn.uz - sensor_data_b.uz)) / np.max(np.abs(sensor_data_b.uz))
    # err_uz_ln = np.max(np.abs(sensor_data_c_ln.uz - sensor_data_b.uz)) / np.max(np.abs(sensor_data_b.uz))

    # err_ux_max_nn = np.max(np.abs(sensor_data_c_nn.ux_max - sensor_data_b.ux_max)) / np.max(np.abs(sensor_data_b.ux_max))
    # err_ux_max_ln = np.max(np.abs(sensor_data_c_ln.ux_max - sensor_data_b.ux_max)) / np.max(np.abs(sensor_data_b.ux_max))

    # err_uy_max_nn = np.max(np.abs(sensor_data_c_nn.uy_max - sensor_data_b.uy_max)) / np.max(np.abs(sensor_data_b.uy_max))
    # err_uy_max_ln = np.max(np.abs(sensor_data_c_ln.uy_max - sensor_data_b.uy_max)) / np.max(np.abs(sensor_data_b.uy_max))

    # err_uz_max_nn = np.max(np.abs(sensor_data_c_nn.uz_max - sensor_data_b.uz_max)) / np.max(np.abs(sensor_data_b.uz_max))
    # err_uz_max_ln = np.max(np.abs(sensor_data_c_ln.uz_max - sensor_data_b.uz_max)) / np.max(np.abs(sensor_data_b.uz_max))

    # err_ux_min_nn = np.max(np.abs(sensor_data_c_nn.ux_min - sensor_data_b.ux_min)) / np.max(np.abs(sensor_data_b.ux_min))
    # err_ux_min_ln = np.max(np.abs(sensor_data_c_ln.ux_min - sensor_data_b.ux_min)) / np.max(np.abs(sensor_data_b.ux_min))

    # err_uy_min_nn = np.max(np.abs(sensor_data_c_nn.uy_min - sensor_data_b.uy_min)) / np.max(np.abs(sensor_data_b.uy_min))
    # err_uy_min_ln = np.max(np.abs(sensor_data_c_ln.uy_min - sensor_data_b.uy_min)) / np.max(np.abs(sensor_data_b.uy_min))

    # err_uz_min_nn = np.max(np.abs(sensor_data_c_nn.uz_min - sensor_data_b.uz_min)) / np.max(np.abs(sensor_data_b.uz_min))
    # err_uz_min_ln = np.max(np.abs(sensor_data_c_ln.uz_min - sensor_data_b.uz_min)) / np.max(np.abs(sensor_data_b.uz_min))

    # err_ux_rms_nn = np.max(np.abs(sensor_data_c_nn.ux_rms - sensor_data_b.ux_rms)) / np.max(np.abs(sensor_data_b.ux_rms))
    # err_ux_rms_ln = np.max(np.abs(sensor_data_c_ln.ux_rms - sensor_data_b.ux_rms)) / np.max(np.abs(sensor_data_b.ux_rms))

    # err_uy_rms_nn = np.max(np.abs(sensor_data_c_nn.uy_rms - sensor_data_b.uy_rms)) / np.max(np.abs(sensor_data_b.uy_rms))
    # err_uy_rms_ln = np.max(np.abs(sensor_data_c_ln.uy_rms - sensor_data_b.uy_rms)) / np.max(np.abs(sensor_data_b.uy_rms))

    # err_uz_rms_nn = np.max(np.abs(sensor_data_c_nn.uz_rms - sensor_data_b.uz_rms)) / np.max(np.abs(sensor_data_b.uz_rms))
    # err_uz_rms_ln = np.max(np.abs(sensor_data_c_ln.uz_rms - sensor_data_b.uz_rms)) / np.max(np.abs(sensor_data_b.uz_rms))

    # err_ux_non_staggered_nn = np.max(np.abs(sensor_data_c_nn.ux_non_staggered - sensor_data_b.ux_non_staggered)) / np.max(np.abs(sensor_data_b.ux_non_staggered))
    # err_ux_non_staggered_ln = np.max(np.abs(sensor_data_c_ln.ux_non_staggered - sensor_data_b.ux_non_staggered)) / np.max(np.abs(sensor_data_b.ux_non_staggered))

    # err_uy_non_staggered_nn = np.max(np.abs(sensor_data_c_nn.uy_non_staggered - sensor_data_b.uy_non_staggered)) / np.max(np.abs(sensor_data_b.uy_non_staggered))
    # err_uy_non_staggered_ln = np.max(np.abs(sensor_data_c_ln.uy_non_staggered - sensor_data_b.uy_non_staggered)) / np.max(np.abs(sensor_data_b.uy_non_staggered))

    # err_uz_non_staggered_nn = np.max(np.abs(sensor_data_c_nn.uz_non_staggered - sensor_data_b.uz_non_staggered)) / np.max(np.abs(sensor_data_b.uz_non_staggered))
    # err_uz_non_staggered_ln = np.max(np.abs(sensor_data_c_ln.uz_non_staggered - sensor_data_b.uz_non_staggered)) / np.max(np.abs(sensor_data_b.uz_non_staggered))

    # err_Ix_nn = np.max(np.abs(sensor_data_c_nn.Ix - sensor_data_b.Ix)) / np.max(np.abs(sensor_data_b.Ix))
    # err_Ix_ln = np.max(np.abs(sensor_data_c_ln.Ix - sensor_data_b.Ix)) / np.max(np.abs(sensor_data_b.Ix))

    # err_Iy_nn = np.max(np.abs(sensor_data_c_nn.Iy - sensor_data_b.Iy)) / np.max(np.abs(sensor_data_b.Iy))
    # err_Iy_ln = np.max(np.abs(sensor_data_c_ln.Iy - sensor_data_b.Iy)) / np.max(np.abs(sensor_data_b.Iy))

    # err_Iz_nn = np.max(np.abs(sensor_data_c_nn.Iz - sensor_data_b.Iz)) / np.max(np.abs(sensor_data_b.Iz))
    # err_Iz_ln = np.max(np.abs(sensor_data_c_ln.Iz - sensor_data_b.Iz)) / np.max(np.abs(sensor_data_b.Iz))

    # err_Ix_avg_nn = np.max(np.abs(sensor_data_c_nn.Ix_avg - sensor_data_b.Ix_avg)) / np.max(np.abs(sensor_data_b.Ix_avg))
    # err_Ix_avg_ln = np.max(np.abs(sensor_data_c_ln.Ix_avg - sensor_data_b.Ix_avg)) / np.max(np.abs(sensor_data_b.Ix_avg))

    # err_Iy_avg_nn = np.max(np.abs(sensor_data_c_nn.Iy_avg - sensor_data_b.Iy_avg)) / np.max(np.abs(sensor_data_b.Iy_avg))
    # err_Iy_avg_ln = np.max(np.abs(sensor_data_c_ln.Iy_avg - sensor_data_b.Iy_avg)) / np.max(np.abs(sensor_data_b.Iy_avg))

    # err_Iz_avg_nn = np.max(np.abs(sensor_data_c_nn.Iz_avg - sensor_data_b.Iz_avg)) / np.max(np.abs(sensor_data_b.Iz_avg))
    # err_Iz_avg_ln = np.max(np.abs(sensor_data_c_ln.Iz_avg - sensor_data_b.Iz_avg)) / np.max(np.abs(sensor_data_b.Iz_avg))

    # check for test pass
    if ((err_p_nn > comparison_thresh)
        or (err_p_ln > comparison_thresh)
        # or (err_p_max_nn > comparison_thresh) or
        # (err_p_max_ln > comparison_thresh) or
        # (err_p_min_nn > comparison_thresh) or
        # (err_p_min_ln > comparison_thresh) or
        # (err_p_rms_nn > comparison_thresh) or
        # (err_p_rms_ln > comparison_thresh) or
        # (err_ux_nn > comparison_thresh) or
        # (err_ux_ln > comparison_thresh) or
        # (err_ux_max_nn > comparison_thresh) or
        # (err_ux_max_ln > comparison_thresh) or
        # (err_ux_min_nn > comparison_thresh) or
        # (err_ux_min_ln > comparison_thresh) or
        # (err_ux_rms_nn > comparison_thresh) or
        # (err_ux_rms_ln > comparison_thresh) or
        # (err_ux_non_staggered_nn > comparison_thresh) or
        # (err_ux_non_staggered_ln > comparison_thresh) or
        # (err_uy_nn > comparison_thresh) or
        # (err_uy_ln > comparison_thresh) or
        # (err_uy_max_nn > comparison_thresh) or
        # (err_uy_max_ln > comparison_thresh) or
        # (err_uy_min_nn > comparison_thresh) or
        # (err_uy_min_ln > comparison_thresh) or
        # (err_uy_rms_nn > comparison_thresh) or
        # (err_uy_rms_ln > comparison_thresh) or
        # (err_uy_non_staggered_nn > comparison_thresh) or
        # (err_uy_non_staggered_ln > comparison_thresh) or
        # (err_uz_nn > comparison_thresh) or
        # (err_uz_ln > comparison_thresh) or
        # (err_uz_max_nn > comparison_thresh) or
        # (err_uz_max_ln > comparison_thresh) or
        # (err_uz_min_nn > comparison_thresh) or
        # (err_uz_min_ln > comparison_thresh) or
        # (err_uz_rms_nn > comparison_thresh) or
        # (err_uz_rms_ln > comparison_thresh) or
        # (err_uz_non_staggered_nn > comparison_thresh) or
        # (err_uz_non_staggered_ln > comparison_thresh) or
        # (err_Ix_nn > comparison_thresh) or
        # (err_Ix_ln > comparison_thresh) or
        # (err_Ix_avg_nn > comparison_thresh) or
        # (err_Ix_avg_ln > comparison_thresh) or
        # (err_Iy_nn > comparison_thresh) or
        # (err_Iy_ln > comparison_thresh) or
        # (err_Iy_avg_nn > comparison_thresh) or
        # (err_Iy_avg_ln > comparison_thresh) or
        # (err_Iz_nn > comparison_thresh) or
        # (err_Iz_ln > comparison_thresh) or
        # (err_Iz_avg_nn > comparison_thresh) or
        # (err_Iz_avg_ln > comparison_thresh)
        ):
        test_pass = False

    assert test_pass, "Fails"


    # # plot
    # if plot_comparisons

    #     figure;
    #     subplot(5, 1, 1);
    #     imagesc(sensor_data_c_ln.p);
    #     colorbar;
    #     title('Cartesian - Linear');

    #     subplot(5, 1, 2);
    #     imagesc(sensor_data_c_nn.p);
    #     colorbar;
    #     title('Cartesian - Nearest Neighbour');

    #     subplot(5, 1, 3);
    #     imagesc(sensor_data_b.p);
    #     colorbar;
    #     title('Binary');

    #     subplot(5, 1, 4);
    #     imagesc(abs(sensor_data_b.p - sensor_data_c_ln.p))
    #     colorbar;
    #     title('Diff (Linear - Binary)');

    #     subplot(5, 1, 5);
    #     imagesc(abs(sensor_data_b.p - sensor_data_c_nn.p))
    #     colorbar;
    #     title('Diff (Nearest Neighbour - Binary)');

    # end