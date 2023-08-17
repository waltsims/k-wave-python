import os

import numpy as np
import matplotlib.pyplot as plt
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import make_cart_circle, make_disc
#from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def main():

    # Create empty array
    karray = kWaveArray()

    # Set the properties for the arc shaped elements and the ring geometry in
    # which they're placed
    radius_of_curv = 100e-3
    diameter = 8e-3
    ring_radius = 50e-3
    num_elements = 20

    # Orient all elements towards the centre of the grid
    focus_pos = Vector([0, 0])

    # Generate the centre position for each element in Cartesian space using
    # makeCartCircle (these positions could also be defined manually, etc)
    elem_pos = make_cart_circle(ring_radius, num_elements, focus_pos)

    # Add elements to the array
    for ind in range(num_elements):
        karray.add_arc_element(elem_pos[:, ind].tolist(), radius_of_curv, diameter, focus_pos.tolist())

    # =========================================================================
    # DEFINE GRID PROPERTIES
    # =========================================================================

    # grid properties
    Nx = 256
    dx = 0.5e-3
    Ny = 256
    dy = 0.5e-3
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # medium properties
    medium = kWaveMedium(sound_speed=1500)

    # time array
    kgrid.makeTime(medium.sound_speed)

    # # =========================================================================
    # # CONVENTIONAL SIMULATION
    # # =========================================================================

    # # Set source as a square (directional) and a disc
    source = kSource()
    source.p0 = make_disc(Vector([Nx, Ny]), Vector([Nx // 2, Ny // 2]), 3)
    source.p0[100:120, 50:200] = 1

    # Assign Cartesian points
    sensor = kSensor()
    sensor.mask = elem_pos

    # # Run the k-Wave simulation using point detectors in the normal way
    # simulation_options = SimulationOptions()
    # execution_options = SimulationExecutionOptions()
    # sensor_data_point = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)
    # exit(0)

    # test_record_path = os.path.join('/Users/farid/PycharmProjects/k-wave-python/tests/'
    #                                 'matlab_test_data_collectors/matlab_collectors',
    #                                 'collectedValues/trimCartPoints.mat')
    # reader = TestRecordReader(test_record_path)
    # expected_sensor_data_point = reader.expected_value_of('sensor_data_point')
    # assert np.allclose(sensor_data_point, expected_sensor_data_point, atol=1e-6)
    # exit(0)

    # =========================================================================
    # KWAVEARRAY SIMULATION
    # =========================================================================

    # Assign binary mask from karray to the sensor mask
    sensor.mask = karray.get_array_binary_mask(kgrid)

    # Run k-Wave simulation
    simulation_options = SimulationOptions()
    execution_options = SimulationExecutionOptions()
    sensor_data = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)

    # Combine data to give one trace per physical array element
    combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data)

    # =========================================================================
    # VISUALISATION
    # =========================================================================

    # Create pml mask (default size in 2D is 20 grid points)
    pml_size = 20
    pml_mask = np.zeros((Nx, Ny), dtype=bool)
    pml_mask[1:pml_size, :] = True
    pml_mask[:, 1:pml_size] = True
    pml_mask[-pml_size+1:, :] = True
    pml_mask[:, -pml_size+1:] = True

    # Plot source, sensor, and pml masks
    fig = plt.figure()
    plt.imshow(np.logical_or(sensor.mask, source.p0, pml_mask), cmap='gray')
    plt.axis('image')

    # Overlay the physical source positions
    # karray.plotArray(False)

    # Plot recorded sensor data
    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    plt.imshow(sensor_data_point / np.max(sensor_data_point), cmap='gray', aspect='auto')
    plt.xlabel('Time [\mus]')
    plt.ylabel('Detector Number')
    plt.title('Arc detectors')

    # Plot a trace from the recorded sensor data
    fig = plt.figure()
    plt.plot(kgrid.t_array * 1e6, sensor_data_point[0, :], label='Cartesian point detectors')
    plt.plot(kgrid.t_array * 1e6, combined_sensor_data[0, :], label='Arc detectors')
    plt.xlabel('Time [\mus]')
    plt.ylabel('Pressure [pa]')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()