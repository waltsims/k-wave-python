"""
    Using An Ultrasound Transducer As A Sensor Example

    This example shows how an ultrasound transducer can be used as a detector
    by substituting a transducer object for the normal sensor input
    structure. It builds on the Defining An Ultrasound Transducer and
    Simulating Ultrasound Beam Patterns examples.
"""
import os
from copy import deepcopy
from tempfile import gettempdir

import numpy as np

# noinspection PyUnresolvedReferences
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_circle
from kwave.utils.matlab import matlab_find, matlab_mask, unflatten_matlab_mask
from tests.diff_utils import compare_against_ref


def test_sd_directional_array_elements():
    # create the computational grid
    grid_size = Vector([180, 180])  # [grid points]
    grid_spacing = Vector([0.1e-3, 0.1e-3])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)

    # define the array of time points [s]
    t_end = 12e-6  # [s]
    kgrid.makeTime(medium.sound_speed, t_end=t_end)

    # =========================================================================
    # DEFINE A FOCUSSED ARRAY OF DIRECTIONAL ELEMENTS
    # =========================================================================

    # define a semicircular sensor centred on the grid
    semicircle_radius = 65  # [grid points]
    arc = make_circle(grid_size, grid_size // 2, semicircle_radius, np.pi)

    # find total number and indices of the grid points constituting the semicircle
    arc_indices = matlab_find(arc, val=1, mode="eq")
    Nv = len(arc_indices)

    # calculate angles between grid points in the arc and the centre of the grid
    arc_angles = np.arctan(matlab_mask(kgrid.y, arc_indices) / matlab_mask(kgrid.x, arc_indices - 1))  # -1 compatibility

    # sort the angles into ascending order, and adjust the indices accordingly
    sorted_index = arc_angles.ravel().argsort()
    sorted_arc_indices = arc_indices[sorted_index]

    # divide the semicircle into Ne separate sensor elements
    Ne = 13
    sensor_mask = np.zeros(grid_size)
    for loop in range(1, Ne + 1):
        # get the indices of the grid points belonging to the current element
        # (there is a two grid point gap between the elements)
        voxel_indices = sorted_arc_indices[((loop - 1) * Nv // Ne) + 1 : (loop * Nv // Ne) - 1] - 1  # -1 compatibility

        # add the element to the sensor.mask
        sensor_mask[unflatten_matlab_mask(sensor_mask, voxel_indices)] = 1
    sensor = kSensor(sensor_mask)

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # define an infinitely wide plane wave source (this is achieved by turning off the PML)
    source = kSource()
    source.p_mask = np.zeros(grid_size)
    source.p_mask[139, :] = 1

    # define a time varying sinusoidal source
    source_freq = 1e6  # [Hz]
    source_mag = 0.5  # [Pa]
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)
    source.p = filter_time_series(kgrid, medium, source.p)

    # run the simulation with the PML switched off on the sides, so that the
    # source continues up to the edge of the domain (and from there infinitely,
    # because of the periodic assumption implicit in pseudospectral methods)
    # input arguments
    input_filename = "example_sd_direct_input.h5"
    pathname = gettempdir()
    input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(
        pml_alpha=np.array([2, 0]),
        smooth_p0=False,
        save_to_disk=True,
        input_filename=input_filename,
        data_path=pathname,
        save_to_disk_exit=True,
    )
    # run the simulation
    kspaceFirstOrder2DC(
        medium=medium,
        kgrid=kgrid,
        source=deepcopy(source),
        sensor=deepcopy(sensor),
        simulation_options=simulation_options,
        execution_options=SimulationExecutionOptions(),
    )

    assert compare_against_ref("out_sd_directional_array_elements", input_file_full_path), "Files do not match!"
