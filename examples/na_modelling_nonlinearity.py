"""
Modelling Nonlinear Wave Propagation Example

Ported from: k-Wave/examples/example_na_modelling_nonlinearity.m

Simulates nonlinear wave propagation in a 1D medium with power-law
absorption. A high-amplitude sinusoidal pressure source (10 MPa) drives
nonlinear steepening governed by the parameter of nonlinearity B/A. The
shock parameter sigma controls the degree of harmonic generation at the
sensor location.

Key physics:
  - Nonlinearity coefficient B/A is computed from the desired shock
    parameter sigma, the source Mach number, wavenumber, and propagation
    distance.
  - The time array is set explicitly via kgrid.setTime() using an integer
    number of points per temporal period (derived from CFL = 0.25) rather
    than the automatic makeTime() routine.
  - sensor.record_start_index is used to record only the last 3 periods,
    capturing the steady-state nonlinear waveform.

The MATLAB original also compares against the Mendousse analytical solution
for nonlinear plane waves — that comparison is omitted here; only the k-Wave
simulation is ported.

Builds on: example_tvsp_homogeneous_medium_monopole (time-varying pressure
source), but uses a 1D grid with nonlinear medium properties.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder


def setup():
    """Set up the nonlinear 1D simulation physics (grid, medium, source).

    Grid: Nx=2048 (1D), dx = c0 / (points_per_wavelength * f0).
    Medium: c = 1500 m/s, rho = 1000 kg/m^3, alpha_coeff = 0.25,
            alpha_power = 2, BonA computed from shock parameter sigma = 2.
    Source: single point at index 10 (1-based), sinusoidal p = 10 MPa.
    Time array: set explicitly with Nt, dt derived from CFL = 0.25 and
                100 points per wavelength at 1 MHz.

    Returns:
        tuple: (kgrid, medium, source)
    """

    # =========================================================================
    # DEFINE SIMULATION PROPERTIES
    # =========================================================================

    p0 = 10e6  # source pressure [Pa]
    c0 = 1500  # sound speed [m/s]
    rho0 = 1000  # density [kg/m^3]
    alpha_0 = 0.25  # absorption coefficient [dB/(MHz^2 cm)]
    sigma = 2  # shock parameter
    source_freq = 1e6  # frequency [Hz]
    points_per_wavelength = 100  # number of grid points per wavelength at f0
    wavelength_separation = 15  # separation between source and detector
    CFL = 0.25  # CFL number

    # =========================================================================
    # COMPUTE GRID
    # =========================================================================

    # compute grid spacing
    dx = c0 / (points_per_wavelength * source_freq)  # [m]

    # compute grid size
    Nx = wavelength_separation * points_per_wavelength + 20  # [grid points]

    # create the computational grid
    kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))

    # =========================================================================
    # MEDIUM PROPERTIES
    # =========================================================================

    # assign the properties of the propagation medium
    medium = kWaveMedium(
        sound_speed=c0,  # [m/s]
        density=rho0,  # [kg/m^3]
        alpha_power=2,
        alpha_coeff=alpha_0,  # [dB/(MHz^2 cm)]
    )

    # compute the sensor position (integer number of wavelengths from source)
    source_pos = 10  # [grid points, 1-based]
    x_px = wavelength_separation * points_per_wavelength  # distance in grid points
    x = x_px * dx  # physical distance [m]

    # compute the nonlinearity coefficient B/A from the desired shock parameter
    mach_num = p0 / (rho0 * c0**2)
    k = 2 * np.pi * source_freq / c0
    BonA = 2 * (sigma / (mach_num * k * x) - 1)
    medium.BonA = BonA

    # =========================================================================
    # TIME ARRAY
    # =========================================================================

    # compute points per temporal period
    # MATLAB: round(points_per_wavelength / CFL) — round half away from zero
    points_per_period = np.round(points_per_wavelength / CFL).astype(int)

    # compute corresponding time spacing
    dt = 1 / (points_per_period * source_freq)

    # create the time array using an integer number of points per period
    t_end = 25e-6  # [s]
    Nt = np.round(t_end / dt).astype(int)
    kgrid.setTime(int(Nt), float(dt))

    # =========================================================================
    # SOURCE
    # =========================================================================

    # define a single source element at position 10 (1-based)
    source = kSource()
    p_mask = np.zeros((Nx, 1), dtype=float)
    p_mask[source_pos - 1, 0] = 1  # convert 1-based to 0-based
    source.p_mask = p_mask

    # create the source term: sinusoidal pressure
    source.p = p0 * np.sin(2 * np.pi * source_freq * kgrid.t_array)

    return kgrid, medium, source


def run(backend="python", device="cpu", quiet=True):
    """Run with a full-grid binary sensor recording p.

    Uses PML outside the domain (pml_inside=False) with pml_size=80 and
    pml_alpha=1.5, matching the MATLAB example's simulation options.

    The sensor records only the last 3 temporal periods via
    sensor.record_start_index.

    Returns:
        dict: Simulation results with key 'p' (n_sensor x Nt_recorded).
    """
    kgrid, medium, source = setup()

    Nx = int(kgrid.N[0])

    # full-grid binary sensor
    sensor = kSensor(
        mask=np.ones((Nx, 1), dtype=float),
        record=["p", "p_final"],
    )

    # set the start time to only record the last three periods
    # MATLAB: points_per_period = round(100 / 0.25) = 400
    points_per_period = np.round(100 / 0.25).astype(int)
    sensor.record_start_index = int(kgrid.Nt - 3 * points_per_period + 1)

    # PML settings matching the MATLAB example
    pml_size = 80
    pml_alpha = 1.5

    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        pml_inside=False,
        pml_size=pml_size,
        pml_alpha=pml_alpha,
        backend=backend,
        device=device,
        quiet=quiet,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kgrid, medium, source = setup()
    result = run(quiet=False)
    p = np.asarray(result["p"])

    # extract the sensor point at the expected detector location
    # source_pos=10 (1-based), x_px = 15*100 = 1500, detector at index 1509 (0-based)
    source_pos_0 = 9  # 0-based
    x_px = 15 * 100
    detector_idx = source_pos_0 + x_px  # 0-based index in the full grid

    # MATLAB: points_per_period = round(100/0.25) = 400
    points_per_period = np.round(100 / 0.25).astype(int)
    dt = 1 / (points_per_period * 1e6)
    t_axis = np.arange(p.shape[1]) * dt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis * 1e6, p[detector_idx, :] * 1e-6, "r-", label="k-Wave")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Nonlinear Wave at Detector (sigma = 2)")
    ax.legend()
    fig.tight_layout()
    plt.show()
