import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import make_cart_circle, make_disc


def main():
    # create empty array
    karray = kWaveArray()

    # define arc properties
    radius = 100e-3  # [m]
    diameter = 8e-3  # [m]
    ring_radius = 50e-3  # [m]
    num_elements = 20

    # orient all elements towards the center of the grid
    focus_pos = Vector([0, 0])  # [m]

    element_pos = make_cart_circle(ring_radius, num_elements, focus_pos)

    for idx in range(num_elements):
        karray.add_arc_element(element_pos[:, idx], radius, diameter, focus_pos)

    # grid properties
    Nx = 256
    dx = 0.5e-3
    Ny = 256
    dy = 0.5e-3
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # medium properties
    medium = kWaveMedium(sound_speed=1500)

    # time array
    kgrid.makeTime(medium.sound_speed)

    source = kSource()
    source.p0 = make_disc(Vector([Nx, Ny]), Vector([Nx / 4 + 20, Ny / 4]), 4)

    sensor = kSensor()
    sensor.mask = element_pos
    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
    )

    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    output = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor_data_point = output['p'].T

    # assign binary mask from karray to the source mask
    sensor.mask = karray.get_array_binary_mask(kgrid)

    output = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor_data = output['p'].T
    combined_sensor_data = karray.combine_sensor_data(kgrid, sensor_data)

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    # create pml mask (default size in 2D is 20 grid points)
    pml_size = 20
    pml_mask = np.zeros((Nx, Ny), dtype=bool)
    pml_mask[:pml_size, :] = 1
    pml_mask[:, :pml_size] = 1
    pml_mask[-pml_size:, :] = 1
    pml_mask[:, -pml_size:] = 1
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
    ax1 = fig.add_subplot(211)  # noqa
    plt.imshow(sensor_data_point / np.max(sensor_data_point), cmap='gray', aspect='auto')
    plt.xlabel(r'Time [$\mu$s]')
    plt.ylabel('Detector Number')
    plt.title('Arc detectors')

    # Plot a trace from the recorded sensor data
    fig = plt.figure()
    plt.plot(kgrid.t_array.squeeze() * 1e6, sensor_data_point[0, :], label='Cartesian point detectors')
    plt.plot(kgrid.t_array.squeeze() * 1e6, combined_sensor_data[0, :], label='Arc detectors')
    plt.xlabel(r'Time [$\mu$s]')
    plt.ylabel('Pressure [pa]')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()