"""
Unit test to compare the simulation results using a labelled and binary source mask.
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
from kwave.utils.mapgen import make_multi_bowl
from kwave.utils.filters import filter_time_series

def test_pstd_elastic_3d_compare_labelled_and_binary_source_mask():

    # set pass variable
    test_pass: bool = True

    # set additional literals to give further permutations of the test.
    COMPARISON_THRESH: float = 1e-14
    pml_inside: bool = True

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # create the computational grid
    Nx: int = 64            # number of grid points in the x direction
    Ny: int = 64            # number of grid points in the y direction
    Nz: int = 64            # number of grid points in the z direction
    dx: float = 0.1e-3        # grid point spacing in the x direction [m]
    dy: float = 0.1e-3        # grid point spacing in the y direction [m]
    dz: float = 0.1e-3        # grid point spacing in the z direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the propagation medium
    sound_speed_compression = 1500.0 # [m/s]
    sound_speed_shear = 1000.0       # [m/s]
    density = 1000.0                 # [kg/m^3]
    medium = kWaveMedium(sound_speed_compression,
                         density=density,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear)

    # create the time array using default CFL condition
    t_end: float = 3e-6
    kgrid.makeTime(medium.sound_speed_compression, t_end=t_end)

    # define multiple curved transducer elements
    bowl_pos = np.array([(19.0, 19.0, Nz / 2.0 - 1.0), (48.0, 48.0, Nz / 2.0 - 1.0)])
    bowl_radius = np.array([20.0, 15.0])
    bowl_diameter = np.array([int(15), int(21)], dtype=np.uint8)
    bowl_focus = np.array([(int(31), int(31), int(31))], dtype=np.uint8)

    binary_mask, labelled_mask = make_multi_bowl(Vector([Nx, Ny, Nz]), bowl_pos, bowl_radius, bowl_diameter, bowl_focus)

    # create sensor object
    sensor = kSensor()

    # create a sensor mask covering the entire computational domain using the
    # opposing corners of a cuboid. These means cuboid corners will be used
    sensor.mask = np.array([[0, 0, 0, Nx - 1, Ny - 1, Nz - 1]], dtype=int).T

    # set the record mode capture the final wave-field and the statistics at
    # each sensor point
    sensor.record = ['p_final', 'p_max']

    # assign the input options
    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           pml_inside=pml_inside)

    # define a time varying sinusoidal source
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

    # run the simulation using the labelled source mask
    sensor_data_labelled_s = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                             source=deepcopy(source),
                                             sensor=deepcopy(sensor),
                                             medium=deepcopy(medium),
                                             simulation_options=deepcopy(simulation_options))

    # assign the source using a binary source mask
    del source
    source = kSource()

    # source mask is from **binary source mask**
    source.s_mask = binary_mask

    index_mask = labelled_mask.flatten('F')[labelled_mask.flatten('F') != 0].astype(int) - int(1)

    source.sxx = deepcopy(labelled_sources[index_mask, :])
    source.syy = deepcopy(source.sxx)
    source.szz = deepcopy(source.sxx)
    source.sxy = deepcopy(source.sxx)
    source.sxz = deepcopy(source.sxx)
    source.syz = deepcopy(source.sxx)

    # run the simulation using the a binary source mask
    sensor_data_binary_s = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                           source=deepcopy(source),
                                           sensor=deepcopy(sensor),
                                           medium=deepcopy(medium),
                                           simulation_options=deepcopy(simulation_options))

    # compute the error from the first cuboid
    L_inf_final_stress_s = np.max(np.abs(sensor_data_labelled_s[1].p_final - sensor_data_binary_s[1].p_final)) / np.max(np.abs(sensor_data_binary_s[1].p_final))
    L_inf_max_stress_s = np.max(np.abs(sensor_data_labelled_s[0].p_max - sensor_data_binary_s[0].p_max)) / np.max(np.abs(sensor_data_binary_s[0].p_max))

    # ----------------------------------------
    # repeat for a velocity source
    # ----------------------------------------

    del source
    source = kSource()

    # assign the source using a **labelled** source mask
    source.u_mask = deepcopy(labelled_mask)
    source.ux = 1e-6 * deepcopy(labelled_sources)
    source.uy = 1e-6 * deepcopy(labelled_sources)
    source.uz = 1e-6 * deepcopy(labelled_sources)

    # run the simulation using the labelled source mask
    sensor_data_labelled_v = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                             source=deepcopy(source),
                                             sensor=deepcopy(sensor),
                                             medium=deepcopy(medium),
                                             simulation_options=deepcopy(simulation_options))

    # assign the source using a **binary** source mask
    del source
    source = kSource()
    source.u_mask = binary_mask
    index_mask = labelled_mask.flatten('F')[labelled_mask.flatten('F') != 0].astype(int) - int(1)
    source.ux = 1e-6 * labelled_sources[index_mask, :]
    source.uy = deepcopy(source.ux)
    source.uz = deepcopy(source.ux)

    # run the simulation using the a binary source mask
    sensor_data_binary_v = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                           source=deepcopy(source),
                                           sensor=deepcopy(sensor),
                                           medium=deepcopy(medium),
                                           simulation_options=deepcopy(simulation_options))

    # compute the error from the first cuboid
    L_inf_final_v = np.max(np.abs(sensor_data_labelled_v[1].p_final - sensor_data_binary_v[1].p_final)) / np.max(np.abs(sensor_data_binary_v[1].p_final))
    L_inf_max_v = np.max(np.abs(sensor_data_labelled_v[0].p_max - sensor_data_binary_v[0].p_max)) / np.max(np.abs(sensor_data_binary_v[0].p_max))

    # compute pass
    if (L_inf_max_stress_s > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "cuboid to binary sensor mask using a stress source " + str(L_inf_max_stress_s)

    # compute pass
    if (L_inf_final_stress_s > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "cuboid to binary sensor mask using a stress source " + str(L_inf_final_stress_s)

    # compute pass
    if (L_inf_final_v > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "cuboid to binary sensor mask using a velocity source " + str(L_inf_final_v)

    if (L_inf_max_v > COMPARISON_THRESH):
        test_pass = False
    assert test_pass, "cuboid to binary sensor mask using a velocity source " + str(L_inf_max_v)
