"""
3D Transducer simulation with pml_size="auto" using the new unified API.

Demonstrates auto PML sizing for optimal FFT performance, and shows
how the unified API handles 3D simulations.
"""
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

# Small 3D grid for demonstration
grid_size = Vector([32, 32, 32])
grid_spacing = Vector([0.1e-3, 0.1e-3, 0.1e-3])
kgrid = kWaveGrid(grid_size, grid_spacing)
kgrid.makeTime(1500)

# Homogeneous medium
medium = kWaveMedium(sound_speed=1500, density=1000)

# Point source at center
source = kSource()
p0 = np.zeros((32, 32, 32))
p0[16, 16, 16] = 1.0
source.p0 = p0

# Record on a plane
sensor_mask = np.zeros((32, 32, 32), dtype=bool)
sensor_mask[:, :, 16] = True
sensor = kSensor(mask=sensor_mask)

# Run with auto PML — finds PML sizes giving smallest prime factors
result = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    pml_size="auto",
)

print(f"Sensor data shape: {result['p'].shape}")
print(f"Final pressure shape: {result['p_final'].shape}")
print("3D transducer example with auto PML completed successfully!")
