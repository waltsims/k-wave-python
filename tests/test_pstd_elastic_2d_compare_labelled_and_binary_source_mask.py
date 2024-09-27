"""
Unit test to compare the simulation results using a labelled and binary source mask
"""

import numpy as np
from copy import deepcopy
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic2D import pstd_elastic_2d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.mapgen import make_multi_arc

@pytest.mark.skip(reason="2D not ready")
def pstd_elastic_2d_compare_labelled_and_binary_source_mask():

    # set pass variable
    test_pass: bool = True

    # set additional literals to give further permutations of the test
    comparison_threshold: float = 1e-15
    pml_inside: bool = False

    # create the computational grid
    Nx: int = 216           # number of grid points in the x direction
    Ny: int = 216           # number of grid points in the y direction
    dx = 50e-3 / float(Nx)  # grid point spacing in the x direction [m]
    dy = dx                 # grid point spacing in the y direction [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the properties of the upper layer of the propagation medium
    sound_speed_compression = 1500.0 * np.ones((Nx, Ny))  # [m/s]
    sound_speed_shear = np.zeros((Nx, Ny))                # [m/s]
    density  = 1000.0 * np.ones((Nx, Ny))                 # [kg/m^3]

    medium = kWaveMedium(sound_speed_compression,
                         density=density,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear)

    t_end = 20e-6
    kgrid.makeTime(medium.sound_speed_compression, t_end=t_end)

    # define a curved transducer element
    arc_pos = np.array([[30, 30], [150, 30], [150, 200]], dtype=int)
    radius = np.array([20, 30, 40], dtype=int)
    diameter = np.array([21, 15, 31], dtype=int)
    focus_pos = np.arary([Nx // 2, Ny // 2], dtype=int)
    binary_mask, labelled_mask = make_multi_arc(Vector([Nx, Ny]), arc_pos, radius, diameter, focus_pos)

    # define a time varying sinusoidal source
    source_freq = 0.25e6       # [Hz]
    source_mag = 0.5           # [Pa]
    source_1 = source_mag * np.sin(2.0 * np.pi * source_freq * kgrid.t_array)

    source_freq = 1e6          # [Hz]
    source_mag = 0.8           # [Pa]
    source_2 = source_mag * np.sin(2.0 * np.pi * source_freq * kgrid.t_array)

    source_freq = 0.05e6       # [Hz]
    source_mag = 0.2           # [Pa]
    source_3 = source_mag * np.sin(2.0 * np.pi * source_freq * kgrid.t_array)

    # assemble sources
    labelled_sources = np.empty((3, kgrid.Nt))
    labelled_sources[0, :] = np.squeeze(source_1)
    labelled_sources[1, :] = np.squeeze(source_2)
    labelled_sources[2, :] = np.squeeze(source_3)

    # assign sources for labelled source mask
    source = kSource()
    source.s_mask = labelled_mask
    source.sxx = labelled_sources
    source.syy = labelled_sources
    source.sxy = labelled_sources

    # create a sensor mask covering the entire computational domain using the
    # opposing corners of a rectangle
    sensor = kSensor()
    sensor.mask = [1, 1, Nx, Ny]

    # set the record mode capture the final wave-field and the statistics at
    # each sensor point
    sensor.record = ['p_final', 'p_max']

    simulation_options_labelled = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                    pml_inside=pml_inside)

    sensor_data_labelled = pstd_elastic_2d(deepcopy(kgrid),
                                           deepcopy(medium),
                                           deepcopy(source),
                                           deepcopy(sensor),
                                           deepcopy(simulation_options_labelled))

    # reassign the source using a binary source mask
    source.s_mask = binary_mask
    index_mask = labelled_mask[labelled_mask != 0]
    source.sxx = labelled_sources[index_mask, :]
    source.syy = source.sxx
    source.sxy = source.sxx

    # run the simulation using the a binary source mask
    simulation_options_binary = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                  pml_inside=pml_inside)

    sensor_data_binary = pstd_elastic_2d(deepcopy(kgrid),
                                         deepcopy(medium),
                                         deepcopy(source),
                                         deepcopy(sensor),
                                         deepcopy(simulation_options_binary))

    # # compute the error from the first cuboid
    # L_inf_final = np.max(np.abs(sensor_data_labelled.p_final - sensor_data_binary.p_final)) / np. max(np.abs(sensor_data_binary.p_final))
    # L_inf_max   = np.max(np.abs(sensor_data_labelled.p_max - sensor_data_binary.p_max)) / np. max(np.abs(sensor_data_binary.p_max))

    # # compute pass
    # if (L_inf_max > comparison_threshold) or (L_inf_final > comparison_threshold):
    #     test_pass = False

    L_inf_max = np.max(np.abs(sensor_data_labelled['p_max'] - sensor_data_binary['p_max'])) / np.max(np.abs(sensor_data_binary['p_max']))
    if (L_inf_max > comparison_threshold):
        test_pass = False
    assert test_pass, "L_inf_max, stress source"

    L_inf_final = np.max(np.abs(sensor_data_labelled['p_final'] - sensor_data_binary['p_final'])) / np.max(np.abs(sensor_data_binary['p_final']))
    if (L_inf_final > comparison_threshold):
        test_pass = False
    assert test_pass, "L_inf_final, stress source"


    # ----------------------------------------

    # repeat for velocity source
    del source
    source = kSource()
    source.u_mask = labelled_mask
    source.ux = labelled_sources * 1e-6
    source.uy = source.ux

    # run the simulation using the labelled source mask
    simulation_options_labelled = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                    pml_inside=pml_inside)

    sensor_data_labelled = pstd_elastic_2d(deepcopy(kgrid),
                                           deepcopy(medium),
                                           deepcopy(source),
                                           deepcopy(sensor),
                                           deepcopy(simulation_options_labelled))

    # reassign the source using a binary source mask
    del source
    source = kSource()
    source.u_mask = binary_mask
    index_mask = labelled_mask[labelled_mask != 0]
    source.ux = labelled_sources[index_mask, :] * 1e-6
    source.uy = source.ux

    # run the simulation using the a binary source mask
    simulation_options_binary = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                                  pml_inside=pml_inside)

    sensor_data_binary = pstd_elastic_2d(deepcopy(kgrid),
                                         deepcopy(medium),
                                         deepcopy(source),
                                         deepcopy(sensor),
                                         deepcopy(simulation_options_binary))

    # compute the error from the first cuboid
    L_inf_max = np.max(np.abs(sensor_data_labelled['p_max'] - sensor_data_binary['p_max'])) / np.max(np.abs(sensor_data_binary['p_max']))
    if (L_inf_max > comparison_threshold):
        test_pass = False
    assert test_pass, "L_inf_max, velocity source"

    L_inf_final = np.max(np.abs(sensor_data_labelled['p_final'] - sensor_data_binary['p_final'])) / np.max(np.abs(sensor_data_binary['p_final']))
    if (L_inf_final > comparison_threshold):
        test_pass = False
    assert test_pass, "L_inf_final, velocity source"

