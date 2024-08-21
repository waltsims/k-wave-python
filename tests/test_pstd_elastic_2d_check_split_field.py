

"""
Unit test to check that the split field components sum to give the correct field, e.g., ux = ux^p + ux^s.
"""

import numpy as np
from copy import deepcopy

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.pstdElastic2D import pstd_elastic_2d
from kwave.ksensor import kSensor
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.utils.signals import tone_burst
from kwave.utils.mapgen import make_arc

def test_pstd_elastic_2d_check_split_field():

    # set comparison threshold
    COMPARISON_THRESH   = 1e-15

    # set pass variable
    test_pass = True

    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================

    # change scale to 2 to reproduce higher resolution figures in help file
    scale: int = 1

    # create the computational grid
    PML_size: int = 10                          # [grid points]
    Nx: int = 128 * scale - 2 * PML_size       # [grid points]
    Ny: int = 192 * scale - 2 * PML_size       # [grid points]
    dx = 0.5e-3 / float(scale)                 # [m]
    dy = 0.5e-3 / float(scale)                 # [m]
    kgrid = kWaveGrid(Vector([Nx, Ny]), Vector([dx, dy]))

    # define the medium properties for the top layer
    cp1                 = 1540     # compressional wave speed [m/s]
    cs1                 = 0        # shear wave speed [m/s]
    rho1                = 1000     # density [kg/m^3]
    alpha0_p1           = 0.1      # compressional absorption [dB/(MHz^2 cm)]
    alpha0_s1           = 0.1      # shear absorption [dB/(MHz^2 cm)]

    # define the medium properties for the bottom layer
    cp2                 = 3000     # compressional wave speed [m/s]
    cs2                 = 1400     # shear wave speed [m/s]
    rho2                = 1850     # density [kg/m^3]
    alpha0_p2           = 1        # compressional absorption [dB/(MHz^2 cm)]
    alpha0_s2           = 1        # shear absorption [dB/(MHz^2 cm)]

    # create the time array
    cfl                 = 0.1
    t_end               = 60e-6
    kgrid.makeTime(cp1, cfl, t_end)

    # define position of heterogeneous slab
    slab = np.zeros((Nx, Ny))
    slab[Nx // 2 - 1:, :] = 1

    # define the source geometry in SI units (where 0, 0 is the grid center)
    arc_pos             = [-15e-3, -25e-3] # [m]
    focus_pos           = [5e-3, 5e-3]     # [m]
    radius              = 25e-3            # [m]
    diameter            = 20e-3            # [m]

    # define the driving signal
    source_freq         = 500e3    # [Hz]
    source_strength     = 1e6      # [Pa]
    source_cycles       = 3        # number of tone burst cycles

    # define the sensor to record the maximum particle velocity everywhere
    sensor = kSensor()
    sensor.record = ['u_split_field', 'u_non_staggered']
    sensor.mask = np.ones((Nx, Ny))

    # =========================================================================
    # SIMULATION
    # =========================================================================

    # convert the source parameters to grid points
    arc_pos     = np.round(np.asarray(arc_pos) / dx).astype(int) + np.asarray([Nx // 2, Ny // 2])
    focus_pos   = np.round(np.asarray(focus_pos) / dx).astype(int) + np.asarray([Nx // 2, Ny // 2])
    radius      = round(radius / dx)
    diameter    = round(diameter / dx)

    # force the diameter to be odd
    if diameter % 2 == 0:
        diameter: int = diameter + int(1)

    # define the medium properties
    sound_speed_compression = cp1 * np.ones((Nx, Ny))
    sound_speed_shear = cs1 * np.ones((Nx, Ny))
    density = rho1 * np.ones((Nx, Ny))
    alpha_coeff_compression = alpha0_p1 * np.ones((Nx, Ny))
    alpha_coeff_shear = alpha0_s1 * np.ones((Nx, Ny))

    medium = kWaveMedium(sound_speed_compression,
                         sound_speed_compression=sound_speed_compression,
                         sound_speed_shear=sound_speed_shear,
                         density=density,
                         alpha_coeff_compression=alpha_coeff_compression,
                         alpha_coeff_shear=alpha_coeff_shear)

    medium.sound_speed_compression[slab == 1] = cp2
    medium.sound_speed_shear[slab == 1]       = cs2
    medium.density[slab == 1]                 = rho2
    medium.alpha_coeff_compression[slab == 1] = alpha0_p2
    medium.alpha_coeff_shear[slab == 1]       = alpha0_s2

    # generate the source geometry
    source_mask = make_arc(Vector([Nx, Ny]), arc_pos, radius, diameter, Vector(focus_pos))

    # assign the source
    source = kSource()
    source.s_mask = source_mask
    fs = 1.0 / kgrid.dt
    source.sxx = -source_strength * tone_burst(fs, source_freq, source_cycles)
    source.syy = source.sxx

    simulation_options = SimulationOptions(simulation_type=SimulationType.ELASTIC,
                                               pml_inside=False,
                                               pml_size=PML_size)

    # run the elastic simulation
    sensor_data_elastic = pstd_elastic_2d(deepcopy(kgrid),
                                          medium=deepcopy(medium),
                                          source=deepcopy(source),
                                          sensor=deepcopy(sensor),
                                          simulation_options=deepcopy(simulation_options))


    # compute errors
    diff_ux = np.max(np.abs(sensor_data_elastic.ux_non_staggered -
                            sensor_data_elastic.ux_split_p -
                            sensor_data_elastic.ux_split_s)) / np.max(np.abs(sensor_data_elastic.ux_non_staggered))

    diff_uy = np.max(np.abs(sensor_data_elastic.uy_non_staggered -
                            sensor_data_elastic.uy_split_p -
                            sensor_data_elastic.uy_split_s)) / max(abs(sensor_data_elastic.uy_non_staggered))

    # check for test pass
    if (diff_ux > COMPARISON_THRESH) or (diff_uy > COMPARISON_THRESH):
        test_pass = False

    return test_pass


