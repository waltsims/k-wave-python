# %% [markdown]
# # 2D Initial Value Problem (New API)
# A disc-shaped initial pressure propagates outward in a homogeneous medium.

# %%
"""
2D Initial Value Problem using the new unified API.

A disc-shaped initial pressure distribution propagates outward in a
homogeneous medium. Demonstrates the simplest usage of kspaceFirstOrder().
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.mapgen import make_disc

# %% Setup and run
# Grid
grid_size = Vector([128, 128])
grid_spacing = Vector([0.1e-3, 0.1e-3])
kgrid = kWaveGrid(grid_size, grid_spacing)
kgrid.makeTime(1500)

# Medium
medium = kWaveMedium(sound_speed=1500, density=1000)

# Source: disc-shaped initial pressure
source = kSource()
source.p0 = make_disc(grid_size, Vector([64, 64]), 5).astype(float)

# Sensor: record everywhere
sensor = kSensor(mask=np.ones((128, 128), dtype=bool))

# Run
result = kspaceFirstOrder(kgrid, medium, source, sensor)

print(f"Sensor data shape: {result['p'].shape}")
print(f"Final pressure shape: {result['p_final'].shape}")
print(f"Max recorded pressure: {np.max(np.abs(result['p'])):.6f}")
print("2D IVP example completed successfully!")
