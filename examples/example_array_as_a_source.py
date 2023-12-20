import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import tone_burst


def main():
    # =========================================================================
    # DEFINE KWAVEARRAY
    # =========================================================================

    # create empty array
    karray = kWaveArray()

    # define arc properties
    radius = 50e-3  # [m]
    diameter = 30e-3  # [m]
    focus_pos = [-20e-3, 0]  # [m]

    # add arc-shaped element
    elem_pos = [10e-3, -40e-3]  # [m]
    karray.add_arc_element(elem_pos, radius, diameter, focus_pos)

    # add arc-shaped element
    elem_pos = [20e-3, 0]  # [m]
    karray.add_arc_element(elem_pos, radius, diameter, focus_pos)

    # add arc-shaped element
    elem_pos = [10e-3, 40e-3]  # [m]
    karray.add_arc_element(elem_pos, radius, diameter, focus_pos)

    # move the array down 10 mm, and rotate by 10 degrees (this moves all the
    # elements together)
    karray.set_array_position([10e-3, 0], 10)

    # =========================================================================
    # DEFINE GRID PROPERTIES
    # =========================================================================

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

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # assign binary mask from karray to the source mask
    source_p_mask = karray.get_array_binary_mask(kgrid)

    # set source signals, one for each physical array element
    f1 = 100e3
    f2 = 200e3
    f3 = 500e3
    sig1 = tone_burst(1 / kgrid.dt, f1, 3).squeeze()
    sig2 = tone_burst(1 / kgrid.dt, f2, 5).squeeze()
    sig3 = tone_burst(1 / kgrid.dt, f3, 5).squeeze()

    # combine source signals into one array
    source_signal = np.zeros((3, max(len(sig1), len(sig2))))
    source_signal[0, :len(sig1)] = sig1
    source_signal[1, :len(sig2)] = sig2
    source_signal[2, :len(sig3)] = sig3

    # get distributed source signals (this automatically returns a weighted
    # source signal for each grid point that forms part of the source)
    source_p = karray.get_distributed_source_signal(kgrid, source_signal)

    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast='single',
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    # run k-Wave simulation (no sensor is used for this example)
    # TODO: I would say proper behaviour would be to return the entire pressure field if sensor is None
    sensor = kSensor()
    sensor.mask = np.ones((Nx, Ny), dtype=bool)

    source = kSource()
    source.p_mask = source_p_mask
    source.p = source_p

    p = kspace_first_order_2d_gpu(kgrid, source, sensor, medium, simulation_options, execution_options)

    p_field = np.reshape(p['p'], (kgrid.Nt, Nx, Ny))
    p_field = np.transpose(p_field, (0, 2, 1))
    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    # Normalize frames based on the maximum value over all frames
    max_value = np.max(p_field)
    normalized_frames = p_field / max_value

    # Create a custom colormap (replace 'viridis' with your preferred colormap)
    cmap = plt.get_cmap('viridis')

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create an empty image with the first normalized frame
    image = ax.imshow(normalized_frames[0], cmap=cmap, norm=colors.Normalize(vmin=0, vmax=1))

    # Function to update the image for each frame
    def update(frame):
        image.set_data(normalized_frames[frame])
        ax.set_title(f'Frame {frame + 1}/{kgrid.Nt}')
        return [image]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=kgrid.Nt, interval=100)  # Adjust interval as needed (in milliseconds)

    # Save the animation as a video file (e.g., MP4)
    video_filename = 'output_video1.mp4'
    ani.save('/tmp/' + video_filename, writer='ffmpeg', fps=30)  # Adjust FPS as needed

    # Show the animation (optional)
    plt.show()

    # Show the animation (optional)
    plt.show()

    # create pml mask (default size in 2D is 20 grid points)
    pml_size = 20
    pml_mask = np.zeros((Nx, Ny), dtype=bool)
    pml_mask[:pml_size, :] = 1
    pml_mask[:, :pml_size] = 1
    pml_mask[-pml_size:, :] = 1
    pml_mask[:, -pml_size:] = 1

    # plot source and pml masks
    plt.figure()
    plt.imshow(np.logical_not(np.squeeze(source.p_mask | pml_mask)), aspect='auto', cmap='gray')
    plt.xlabel('x-position [m]')
    plt.ylabel('y-position [m]')
    plt.title('Source and PML Masks')
    plt.show()

    # overlay the physical source positions
    plt.figure()
    # TODO: missing karray.plot_array(show=True)
    # karray.plot_array(show=True)


if __name__ == "__main__":
    main()
