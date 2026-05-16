"""
Modelling Sensor Directivity In 3D Example

Ported from: k-Wave/examples/example_sd_directivity_modelling_3D.m

3D version of the 2D sensor directivity example.  Eleven equally-spaced
point sources on a semicircular arc (radius 20 grid points, lying in the
x-y plane) are fired one at a time.  A 17 x 17 grid-point square detector
face records each arrival, and the summed signal is compared across source
angles to show the expected directivity roll-off.

Grid: 64 x 64 x 64, dx = dy = dz = 100e-3/64 m.
Medium: c = 1500 m/s (lossless).

Note: This example does NOT use sensor.directivity_angle.  The directivity
arises purely from spatial averaging across the detector surface.
The 3D loop (11 simulations on a 64^3 grid) can take a few minutes.

author: Ben Cox and Bradley Treeby
date: 29th October 2010
"""
# %%
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.conversion import cart2grid
from kwave.utils.filters import filter_time_series
from kwave.utils.mapgen import make_cart_circle


# %%
def setup():
    """Set up simulation physics (grid, medium, source template).

    Returns:
        tuple: (kgrid, medium, source) -- source.p is the waveform,
               source.p_mask is not yet set.
    """
    # create the computational grid
    Nx = 64
    Ny = 64
    Nz = 64
    dx = 100e-3 / Nx
    dy = dx
    dz = dx
    kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dy, dz]))

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    # define a time-varying sinusoidal source
    source_freq = 0.25e6  # [Hz]
    source_mag = 1.0  # [Pa]
    source = kSource()
    source.p = source_mag * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    # filter the source to remove high frequencies not supported by the grid
    source.p = filter_time_series(kgrid, medium, source.p)

    return kgrid, medium, source


# %%
def run(backend="python", device="cpu", quiet=True):
    """Run 11 simulations (one per source angle) and return directivity data.

    Sensor: 17 x 17 square face at x = Nx//2 (0-based: 32), centred in y-z.
    Sources: 11 points on a semicircular arc in the x-y plane (z = 0),
             radius 20 grid points, snapped to grid via cart2grid.

    Returns:
        dict with keys:
            'single_element_data': (Nt, 11) summed sensor output per source
            'source_positions': linear indices of the 11 source grid points
    """
    Nx, Ny, Nz = 64, 64, 64
    dx = 100e-3 / Nx

    kgrid_for_cart = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

    # define a large area detector (17 x 17 face at x = Nx//2)
    # MATLAB: sensor.mask(Nx/2+1, (Ny/2-sz/2+1):(Ny/2+sz/2+1),
    #                              (Nz/2-sz/2+1):(Nz/2+sz/2+1)) = 1
    # 1-based: row 33, cols 25..41, slices 25..41
    # 0-based: row 32, cols 24..40, slices 24..40
    sz = 16
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=float)
    sensor_mask[
        Nx // 2,
        (Ny // 2 - sz // 2) : (Ny // 2 + sz // 2 + 1),
        (Nz // 2 - sz // 2) : (Nz // 2 + sz // 2 + 1),
    ] = 1

    # define equally spaced point sources on a semicircle in x-y plane
    radius = 20  # [grid points]
    points = 11
    circle_2d = make_cart_circle(radius * dx, points, center_pos=Vector([0, 0]), arc_angle=np.pi)
    # extend to 3D: z = 0 for all points
    circle_3d = np.vstack([circle_2d, np.zeros((1, points))])

    # snap Cartesian coordinates to nearest grid points
    circle_grid, _, _ = cart2grid(kgrid_for_cart, circle_3d, order="C")

    # find the linear indices of the source points
    source_positions = np.flatnonzero(circle_grid)

    # pre-allocate output
    kgrid0, _, _ = setup()
    Nt = kgrid0.Nt
    single_element_data = np.zeros((Nt, points))

    # run a simulation for each source position
    for i in range(points):
        kgrid, medium, source = setup()

        # set source mask for this point source
        p_mask = np.zeros((Nx, Ny, Nz), dtype=float)
        p_mask.flat[source_positions[i]] = 1
        source.p_mask = p_mask

        sensor = kSensor(mask=sensor_mask.astype(bool))

        result = kspaceFirstOrder(
            kgrid,
            medium,
            source,
            sensor,
            backend=backend,
            device=device,
            quiet=quiet,
            pml_inside=True,
            pml_size=10,
        )
        p = np.asarray(result["p"])

        # sum over all sensor grid points to simulate large-aperture detector
        single_element_data[:, i] = p.sum(axis=0)

    return {
        "single_element_data": single_element_data,
        "source_positions": source_positions,
    }


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run(quiet=False)
    kgrid, _, _ = setup()

    single_element_data = result["single_element_data"]
    source_positions = result["source_positions"]

    Nx, Ny, Nz = 64, 64, 64
    t_array = np.asarray(kgrid.t_array).ravel()
    t_us = t_array * 1e6

    # -- Figure 1: time series for each source direction --
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_us, single_element_data)
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [au]")
    ax1.set_title("Time Series For Each Source Direction (3D)")

    # -- Figure 2: directivity pattern --
    dx = 100e-3 / Nx
    kgrid_full = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))
    x_vec = np.asarray(kgrid_full.x_vec).ravel()
    y_vec = np.asarray(kgrid_full.y_vec).ravel()
    # unravel flat indices into (ix, iy, iz)
    ix, iy, iz = np.unravel_index(source_positions, (Nx, Ny, Nz))
    x_src = x_vec[ix]
    y_src = y_vec[iy]
    angles = np.arctan(y_src / x_src)  # MATLAB uses atan(y/x), not atan2

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(angles, np.max(single_element_data, axis=0), "o")
    ax2.set_xlabel("Angle Between Source and Centre of Detector Face [rad]")
    ax2.set_ylabel("Maximum Detected Pressure [au]")
    ax2.set_title("Directivity Pattern (3D)")

    plt.tight_layout()
    plt.show()
