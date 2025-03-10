from tempfile import gettempdir

import numpy as np

# noinspection PyUnresolvedReferences
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrderAS import kspaceFirstOrderASC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import create_cw_signals

# from kwave.utils.mapgen import make_disc
# from tests.diff_utils import compare_against_ref


def test_ivp_axisymmetric_karray_simulation():
    # create the computational grid
    Nx = 128
    Ny = 64
    grid_size = Vector([Nx, Ny])  # [grid points]
    dx = 0.1e-3
    grid_spacing = Vector([dx, dx])  # [m]
    kgrid = kWaveGrid(grid_size, grid_spacing)

    # define the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=1500.0 * np.ones(grid_size),  # [m/s]
        density=1000.0 * np.ones(grid_size),  # [kg/m^3]
    )
    medium.sound_speed[grid_size.x // 2 - 1 :, :] = 1800.0  # [m/s]
    medium.density[grid_size.x // 2 - 1 :, :] = 1200.0  # [kg/m^3]

    # piston diameter [m]
    source_diam = 10e-3
    # source frequency [Hz]
    source_f0 = 1e6
    # source pressure [Pa]
    source_mag = np.array([1e6])
    # phase [rad]
    source_phase = np.array([0.0])

    ppw = 4  # number of points per wavelength
    t_end = 40e-6  # total compute time [s] (this must be long enough to reach steady state)
    record_periods = 1  # number of periods to record
    cfl = 0.05  # CFL number
    bli_tolerance = 0.05  # tolerance for truncation of the off-grid source points
    upsampling_rate = 10  # density of integration points relative to grid

    # compute points per period
    ppp = round(ppw / cfl)
    # compute time step
    dt = 1.0 / (ppp * source_f0)
    # create the time array using an integer number of points per period
    Nt = round(t_end / dt)
    kgrid.setTime(Nt, dt)

    # create time varying continuous wave source
    source_sig = create_cw_signals(np.squeeze(kgrid.t_array), source_f0, source_mag, source_phase)

    # create empty kWaveArray this specfies the transducer properties in
    # axisymmetric coordinate system
    karray = kWaveArray(axisymmetric=True, bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate, single_precision=True)

    # add line shaped element for transducer
    karray.add_line_element([kgrid.x_vec[0].item(), -source_diam / 2.0], [kgrid.x_vec[0].item(), source_diam / 2.0])

    # make a source object
    source = kSource()
    # assign binary mask using the karray
    source.p_mask = karray.get_array_binary_mask(kgrid)
    # assign source pressure output in time
    source.p = karray.get_distributed_source_signal(kgrid, source_sig)

    sensor = kSensor()
    # set sensor mask to record central plane, not including the source point
    sensor.mask = np.zeros((Nx, Ny), dtype=bool)
    sensor.mask[1:, :] = True
    # set the record type: record the pressure waveform
    sensor.record = ["p"]
    # record only the final few periods when the field is in steady state
    sensor.record_start_index = kgrid.Nt - record_periods * ppp + 1

    # set the input settings
    input_filename = "example_ivp_axisymmetric_karray_input.h5"
    pathname = gettempdir()
    # input_file_full_path = os.path.join(pathname, input_filename)
    simulation_options = SimulationOptions(save_to_disk=True, input_filename=input_filename, save_to_disk_exit=True, data_path=pathname)
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False, delete_data=False, verbose_level=2)

    # run the simulation
    kspaceFirstOrderASC(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )

    assert True
