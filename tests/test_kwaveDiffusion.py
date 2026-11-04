import numpy as np

from kwave.kgrid import kWaveGrid
from kwaveDiffusion import DiffusionMedium, DiffusionSource, kWaveDiffusion


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------

Nx, Ny, Nz = 64, 64, 64
dx = 0.5e-3

kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])


# ----------------------------------------------------------------------
# Initial temperature and heat source
# ----------------------------------------------------------------------

T0 = np.full((Nx, Ny, Nz), 37.0, dtype=np.float64)

S = np.zeros_like(T0)

# Small spherical-ish heating region
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
radius = 5

x, y, z = np.indices(T0.shape)

mask = (
    (x - cx)**2
    + (y - cy)**2
    + (z - cz)**2
    <= radius**2
)

S[mask] = 1.0  # K/s

# The solver accepts volumetric heat deposition Q [W/m^3].
# Convert the desired temperature source S [K/s] using rho * cp.
rho = 1000.0       # kg/m^3
cp = 4000.0        # J/(kg K)
Q = S * rho * cp


# ----------------------------------------------------------------------
# Material properties
# ----------------------------------------------------------------------

D = 1.4e-7         # m^2/s
P = 0.01           # 1/s
Ta = 37.0          # degC

medium = DiffusionMedium(
    diffusion_coeff=D,
    density=rho,
    specific_heat=cp,
    perfusion_coeff=P,
    blood_ambient_temperature=Ta,
)

source = DiffusionSource(
    T0=T0,
    Q=Q,
)


# ----------------------------------------------------------------------
# CPU / CUDA selection
# ----------------------------------------------------------------------

# Uses CUDA when CuPy and a CUDA device are available; otherwise CPU.
try:
    import cupy as cp

    device = "gpu" if cp.cuda.runtime.getDeviceCount() > 0 else "cpu"
except (ImportError, RuntimeError):
    device = "cpu"

print(f"Using device: {device}")


# ----------------------------------------------------------------------
# Run simulation
# ----------------------------------------------------------------------

t = 20.0  # total simulation time [s]

solver = kWaveDiffusion(
    kgrid,
    medium,
    source,
    use_kspace=True,
    dtype=np.float64,
    display_updates=True,
    device=device,
)

# A single k-space-corrected time step advances the homogeneous
# solution over the full interval.
T = solver.take_time_step(
    Nt=1,
    dt=t,
)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

# Convert to NumPy for reporting/plotting when running on CUDA.
T = cp.asnumpy(T)

print("Temperature shape:", T.shape)
print("Minimum temperature:", T.min(), "degC")
print("Maximum temperature:", T.max(), "degC")
print("Maximum temperature rise:", T.max() - T0.max(), "degC")
print("Maximum CEM43:", cp.asnumpy(solver.cem43).max())
print("Lesion volume:", cp.asnumpy(solver.lesion_size), "m^3")
