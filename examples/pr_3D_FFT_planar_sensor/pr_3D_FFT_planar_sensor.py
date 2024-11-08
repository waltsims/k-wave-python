from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.kspacePlaneRecon import kspacePlaneRecon
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.plot import voxel_plot
from kwave.utils.colormap import get_color_map
from kwave.utils.mapgen import make_ball
from kwave.utils.filters import smooth

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# 3D FFT Reconstruction For A Planar Sensor Example

# This example demonstrates the use of k-Wave for the reconstruction of a
# three-dimensional photoacoustic wave-field recorded over a planar array
# of sensor elements. The sensor data is simulated using kspaceFirstOrder3D
# and reconstructed using kspacePlaneRecon. It builds on the Simulations In
# Three Dimensions and 2D FFT Reconstruction For A Line Sensor examples.


def main():

    # --------------------
    # SIMULATION
    # --------------------

    scale = 1

    # create the computational grid
    PML_size = 10  # size of the PML in grid points
    N = Vector([32, 64, 64]) * scale - 2 * PML_size  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3]) / scale  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using makeBall
    ball_magnitude = 10  # [Pa]
    ball_radius = 3 * scale  # [grid points]
    p0 = ball_magnitude * make_ball(N, N / 2, ball_radius)
    p0 = smooth(p0, restore_max=True)

    source = kSource()
    source.p0 = p0

    # define a binary planar sensor
    sensor = kSensor()
    sensor_mask = np.zeros(N)
    sensor_mask[0] = 1
    sensor.mask = sensor_mask

    # set the input arguments
    simulation_options = SimulationOptions(
        save_to_disk=True,
        pml_size=PML_size,
        pml_inside=False,
        smooth_p0=False,
        data_cast='single'
    )

    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

    # run the simulation
    sensor_data = kspaceFirstOrder3D(kgrid, source, sensor, medium, simulation_options, execution_options)
    sensor_data = sensor_data['p'].T

    # reshape sensor data to y, z, t
    sensor_data_rs = sensor_data.reshape(N[1], N[2], kgrid.Nt)

    # reconstruct the initial pressure
    p_xyz = kspacePlaneRecon(sensor_data_rs, kgrid.dy, kgrid.dz, kgrid.dt.item(),
                             medium.sound_speed.item(), data_order='yzt', pos_cond=True)

    # define a k-space grid using the dimensions of p_xyz
    N_recon = Vector(p_xyz.shape)
    d_recon = Vector([kgrid.dt.item() * medium.sound_speed.item(), kgrid.dy, kgrid.dz])
    kgrid_recon = kWaveGrid(N_recon, d_recon)

    # define a k-space grid with the same z-spacing as p0
    kgrid_interp = kWaveGrid(N, d)

    # resample the p_xyz to be the same size as p0
    interp_func = RegularGridInterpolator(
        (kgrid_recon.x_vec[:, 0] - kgrid_recon.x_vec[:, 0].min(),
         kgrid_recon.y_vec[:, 0] - kgrid_recon.y_vec[:, 0].min(),
         kgrid_recon.z_vec[:, 0] - kgrid_recon.z_vec[:, 0].min()),
        p_xyz, method='linear'
    )
    query_points = np.stack((kgrid_interp.x - kgrid_interp.x.min(),
                             kgrid_interp.y - kgrid_interp.y.min(),
                             kgrid_interp.z - kgrid_interp.z.min()),
                            axis=-1)
    p_xyz_rs = interp_func(query_points)


    # --------------------
    # VISUALIZATION
    # --------------------

    # plot the initial pressure and sensor surface in voxel form
    # voxel_plot(np.single((p0 + sensor_mask) > 0))  # todo: needs unsmoothed po + plot not working

    # plot the initial pressure
    plot_scale = [-10, 10]
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(p0[:, :, N[2] // 2],
                     extent=[kgrid_interp.y_vec.min() * 1e3, kgrid_interp.y_vec.max() * 1e3,
                             kgrid_interp.x_vec.max() * 1e3, kgrid_interp.x_vec.min() * 1e3],
                     vmin=plot_scale[0], vmax=plot_scale[1], cmap=get_color_map())
    axs[0, 0].set_title('x-y plane')

    axs[0, 1].imshow(p0[:, N[1] // 2, :],
                     extent=[kgrid_interp.z_vec.min() * 1e3, kgrid_interp.z_vec.max() * 1e3,
                             kgrid_interp.x_vec.max() * 1e3, kgrid_interp.x_vec.min() * 1e3],
                     vmin=plot_scale[0], vmax=plot_scale[1], cmap=get_color_map())
    axs[0, 1].set_title('x-z plane')

    axs[1, 0].imshow(p0[N[0] // 2, :, :],
                     extent=[kgrid_interp.z_vec.min() * 1e3, kgrid_interp.z_vec.max() * 1e3,
                             kgrid_interp.y_vec.max() * 1e3, kgrid_interp.y_vec.min() * 1e3],
                     vmin=plot_scale[0], vmax=plot_scale[1], cmap=get_color_map())
    axs[1, 0].set_title('y-z plane')

    axs[1, 1].axis('off')
    axs[1, 1].set_title('(All axes in mm)')
    plt.show()

    # plot the reconstructed initial pressure
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(p_xyz_rs[:, :, N[2] // 2],
                     extent=[kgrid_interp.y_vec.min() * 1e3, kgrid_interp.y_vec.max() * 1e3,
                             kgrid_interp.x_vec.max() * 1e3, kgrid_interp.x_vec.min() * 1e3],
                     vmin=plot_scale[0], vmax=plot_scale[1], cmap=get_color_map())
    axs[0, 0].set_title('x-y plane')

    axs[0, 1].imshow(p_xyz_rs[:, N[1] // 2, :],
                     extent=[kgrid_interp.z_vec.min() * 1e3, kgrid_interp.z_vec.max() * 1e3,
                             kgrid_interp.x_vec.max() * 1e3, kgrid_interp.x_vec.min() * 1e3],
                     vmin=plot_scale[0], vmax=plot_scale[1], cmap=get_color_map())
    axs[0, 1].set_title('x-z plane')

    axs[1, 0].imshow(p_xyz_rs[N[0] // 2, :, :],
                     extent=[kgrid_interp.z_vec.min() * 1e3, kgrid_interp.z_vec.max() * 1e3,
                             kgrid_interp.y_vec.max() * 1e3, kgrid_interp.y_vec.min() * 1e3],
                     vmin=plot_scale[0], vmax=plot_scale[1], cmap=get_color_map())
    axs[1, 0].set_title('y-z plane')

    axs[1, 1].axis('off')
    axs[1, 1].set_title('(All axes in mm)')
    plt.show()


if __name__ == '__main__':
    main()