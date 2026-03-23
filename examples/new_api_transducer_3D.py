"""
3D simulation using the new unified API.

Demonstrates 3D wave propagation with the unified kspaceFirstOrder() function.
For auto PML sizing, use larger grids (>128) where the auto-selected PML
leaves sufficient interior points.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

# 3D grid for demonstration
grid_size = Vector([64, 64, 64])
grid_spacing = Vector([0.1e-3, 0.1e-3, 0.1e-3])
kgrid = kWaveGrid(grid_size, grid_spacing)
kgrid.makeTime(1500)

# Homogeneous medium
medium = kWaveMedium(sound_speed=1500, density=1000)

# Point source at center
source = kSource()
p0 = np.zeros((64, 64, 64))
p0[32, 32, 32] = 1.0
source.p0 = p0

# Record on a plane
sensor_mask = np.zeros((64, 64, 64), dtype=bool)
sensor_mask[:, :, 32] = True
sensor = kSensor(mask=sensor_mask)

# Run with custom PML size
result = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    pml_size=10,
)

print(f"Sensor data shape: {result['p'].shape}")
print(f"Final pressure shape: {result['p_final'].shape}")
print("3D transducer example with auto PML completed successfully!")
