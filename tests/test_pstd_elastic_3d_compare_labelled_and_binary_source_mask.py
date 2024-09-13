"""
Unit test to compare the simulation results using a labelled andbinary source mask.
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

    # set additional literals to give further permutations of the test
    COMPARISON_THRESH: float = 1e-15
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
    bowl_pos = [(19, 19, Nz // 2 - 1), (48, 48, Nz // 2 - 1)]
    bowl_pos = np.array([[19, 19, Nz // 2 - 1], [48, 48, Nz // 2 - 1]], dtype=int)
    bowl_radius = [int(20), int(15)]
    bowl_diameter = [int(15), int(21)]
    bowl_focus = [(int(31), int(31), int(31))]
    binary_mask, labelled_mask = make_multi_bowl(Vector([Nx, Ny, Nz]), bowl_pos, bowl_radius, bowl_diameter, bowl_focus)

    # define a time varying sinusoidal source
    source_freq = 1e6          # [Hz]
    source_mag = 0.5           # [Pa]
    source_0 = source_mag * np.sin(2.0 * np.pi * source_freq * np.squeeze(kgrid.t_array))
    source_0 = filter_time_series(kgrid, medium, source_0)

    source_freq = 3e6          # [Hz]
    source_mag = 0.8           # [Pa]
    source_1 = source_mag * np.sin(2.0 * np.pi * source_freq * np.squeeze(kgrid.t_array))
    source_1 = filter_time_series(kgrid, medium, source_1)

    # assemble sources
    labelled_sources = np.zeros((2, kgrid.Nt))
    labelled_sources[0, :] = np.squeeze(source_0)
    labelled_sources[1, :] = np.squeeze(source_1)

    # create ksource object
    source = kSource()

    # assign sources for labelled source mask
    source.s_mask = deepcopy(labelled_mask)
    source.sxx = deepcopy(labelled_sources)
    source.syy = deepcopy(labelled_sources)
    source.szz = deepcopy(labelled_sources)
    source.sxy = deepcopy(labelled_sources)
    source.sxz = deepcopy(labelled_sources)
    source.syz = deepcopy(labelled_sources)

    # create sensor object
    sensor = kSensor()

    # create a sensor mask covering the entire computational domain using the
    # opposing corners of a cuboid
    sensor.mask = np.array([[0, 0, 0, Nx-1, Ny-1, Nz-1]], dtype=int).T

    # set the record mode capture the final wave-field and the statistics at
    # each sensor point
    sensor.record = ['p_final', 'p_max']

    # assign the input options
    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                           pml_inside=pml_inside)

    # run the simulation using the labelled source mask
    sensor_data_labelled = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                           source=deepcopy(source),
                                           sensor=deepcopy(sensor),
                                           medium=deepcopy(medium),
                                           simulation_options=deepcopy(simulation_options))


    # reassign the source using a binary source mask
    del source
    source = kSource()
    source.s_mask = binary_mask
    index_mask = labelled_mask[labelled_mask != 0].astype(int) - int(1)
    source.sxx = labelled_sources[index_mask, :]
    source.syy = deepcopy(source.sxx)
    source.szz = deepcopy(source.sxx)
    source.sxy = deepcopy(source.sxx)
    source.sxz = deepcopy(source.sxx)
    source.syz = deepcopy(source.sxx)

    # run the simulation using the a binary source mask
    sensor_data_binary = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                         source=deepcopy(source),
                                         sensor=deepcopy(sensor),
                                         medium=deepcopy(medium),
                                         simulation_options=deepcopy(simulation_options))

    # compute the error from the first cuboid
    L_inf_final = np.max(np.abs(sensor_data_labelled[0].p_final - sensor_data_binary[0].p_final)) / np.max(np.abs(sensor_data_binary[0].p_final))
    L_inf_max   = np.max(np.abs(sensor_data_labelled[0].p_max - sensor_data_binary[0].p_max)) / np.max(np.abs(sensor_data_binary[0].p_max))

    # compute pass
    if (L_inf_max > COMPARISON_THRESH) or (L_inf_final > COMPARISON_THRESH):
        test_pass = False

    assert test_pass, "cuboid to binary sensor mask using a stress source"

    # ----------------------------------------
    # repeat for a velocity source
    # ----------------------------------------

    del source
    source = kSource()
    source.u_mask = labelled_mask.astype(int)
    source.ux = 1e6 * labelled_sources
    source.uy = deepcopy(source.ux)
    source.uz = deepcopy(source.ux)

    # run the simulation using the labelled source mask
    sensor_data_labelled = pstd_elastic_3d(kgrid=deepcopy(kgrid),
                                           source=deepcopy(source),
                                           sensor=deepcopy(sensor),
                                           medium=deepcopy(medium),
                                           simulation_options=deepcopy(simulation_options))

    # reassign the source using a binary source mask
    del source
    source = kSource()
    source.u_mask = binary_mask
    index_mask = labelled_mask[labelled_mask != 0].astype(int) - int(1)
    source.ux = 1e6 * labelled_sources[index_mask, :]
    source.uy = deepcopy(source.ux)
    source.uz = deepcopy(source.ux)

    # run the simulation using the a binary source mask
    sensor_data_binary = pstd_elastic_3d(deepcopy(kgrid),
                                         deepcopy(medium),
                                         deepcopy(source),
                                         deepcopy(sensor),
                                         deepcopy(simulation_options))

    # compute the error from the first cuboid
    L_inf_final = np.max(np.abs(sensor_data_labelled[0].p_final - sensor_data_binary[0].p_final)) / np.max(np.abs(sensor_data_binary[0].p_final))
    L_inf_max   = np.max(np.abs(sensor_data_labelled[0].p_max - sensor_data_binary[0].p_max)) / np.max(np.abs(sensor_data_binary[0].p_max))

    # compute pass
    if (L_inf_max > COMPARISON_THRESH) or (L_inf_final > COMPARISON_THRESH):
        test_pass = False

    assert test_pass, "cuboid to binary sensor mask using a velocity source"