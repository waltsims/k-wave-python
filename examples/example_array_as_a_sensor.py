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
from kwave.utils.conversion import cart2grid
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import make_cart_circle, make_disc
from kwave.utils.signals import reorder_binary_sensor_data


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
    Nxy = Vector([256, 256])
    dxy = Vector([0.5e-3, 0.5e-3])
    kgrid = kWaveGrid(Nxy, dxy)

    # medium properties
    medium = kWaveMedium(sound_speed=1500)

    # time array
    kgrid.makeTime(medium.sound_speed)

    source = kSource()
    source.p0 = make_disc(Nxy, Vector([Nx / 4 + 20, Ny / 4]), 4)
    source.p0[99:119, 59:199] = 1
    logical_p0 = source.p0.astype(bool)
    sensor = kSensor()
    sensor.mask = element_pos
    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
    )

    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    output = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)
        # TODO (walter): This should be done by kspaceFirstOrder
    _, _, reorder_index = cart2grid(kgrid, element_pos)
    sensor_data_point = reorder_binary_sensor_data(output['p'].T, reorder_index=reorder_index)
    
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

    # Plot source, sensor, and pml masks
    fig = plt.figure()
    plt.imshow(sensor.mask | logical_p0 | pml_mask, cmap='gray')
    plt.axis('image')

    # Overlay the physical source positions
    # karray.plotArray(False)

    # Plot recorded sensor data
    fig, (ax1, ax2)= plt.subplots(ncols=1, nrows=2)
    ax1.imshow(sensor_data_point, aspect='auto')
    ax1.set_xlabel(r'Time [$\mu$s]')
    ax1.set_ylabel('Detector Number')
    ax1.set_title('Cartesian point detectors')
    
    ax2.imshow(combined_sensor_data, aspect='auto')
    ax2.set_xlabel(r'Time [$\mu$s]')
    ax2.set_ylabel('Detector Number')
    ax2.set_title('Arc detectors')

    fig.subplots_adjust(hspace=0.5)

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