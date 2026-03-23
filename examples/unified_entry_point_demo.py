# %% [markdown]
# # Unified Entry Point Demo
# Demonstrates the new single-function kspaceFirstOrder() API across backends and PML options.

# %%
"""
Unified Entry Point Demo — kspaceFirstOrder()

Demonstrates the new single-function API for k-Wave simulations.
Compare with the legacy approach that required separate options classes.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

# %% Common setup
grid_size = Vector([128, 128])
kgrid = kWaveGrid(grid_size, Vector([0.1e-3, 0.1e-3]))
kgrid.makeTime(1500)

medium = kWaveMedium(sound_speed=1500, density=1000)

source = kSource()
p0 = np.zeros((128, 128))
p0[64, 64] = 1.0
source.p0 = p0

sensor = kSensor(mask=np.ones((128, 128), dtype=bool))

# %% Example 1: Native CPU (default)
print("Running native CPU simulation...")
result = kspaceFirstOrder(kgrid, medium, source, sensor)
print(f"  Sensor data: {result['p'].shape}")

# %% Example 2: Custom PML
print("\nRunning with custom PML (10, 15)...")
result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=(10, 15))
print(f"  Sensor data: {result['p'].shape}")

# %% Example 3: Auto PML
print("\nRunning with auto PML...")
result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size="auto")
print(f"  Sensor data: {result['p'].shape}")

# %% Example 4: C++ save_only (writes HDF5 for cluster submission)
import tempfile

print("\nSaving HDF5 for C++ binary...")
result = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    backend="cpp",
    save_only=True,
    data_path=tempfile.mkdtemp(),
)
print(f"  Input file: {result['input_file']}")

# %% Example 5: Migrating from legacy options
print("\nMigrating from legacy options...")
from kwave.compat import options_to_kwargs
from kwave.options.simulation_options import SimulationOptions

sim_opts = SimulationOptions(smooth_p0=False)
kwargs = options_to_kwargs(simulation_options=sim_opts)
# Remove backend-specific kwargs that would require C++ binary
kwargs.pop("data_path", None)
kwargs.pop("backend", None)
result = kspaceFirstOrder(kgrid, medium, source, sensor, **kwargs)
print(f"  Sensor data: {result['p'].shape}")

print("\nAll examples completed successfully!")
