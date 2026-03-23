# %% [markdown]
# # 1D Initial Value Problem
# Heterogeneous medium with reflections at impedance boundaries.

# %%
"""
1D Initial Value Problem — heterogeneous medium with reflections.

A smooth pressure pulse propagates through a medium with varying
sound speed and density. Reflections at impedance boundaries and
sensor time-series are recorded and plotted.
"""
import matplotlib.pyplot as plt
import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

# %% Grid and medium
Nx = 512
dx = 0.05e-3  # [m]
kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))

# -- Heterogeneous medium --
sound_speed = 1500 * np.ones(Nx)
sound_speed[: Nx // 3] = 2000  # faster region (bone-like)

density = 1000 * np.ones(Nx)
density[4 * Nx // 5 :] = 1500  # denser region

medium = kWaveMedium(sound_speed=sound_speed, density=density)

# -- Time stepping --
kgrid.makeTime(sound_speed, cfl=0.3)

# %% Source and sensor
source = kSource()
source.p0 = np.zeros(Nx)
x0, width = 280, 100
pulse = 0.5 * (np.sin(np.arange(width + 1) * np.pi / width - np.pi / 2) + 1)
source.p0[x0 : x0 + width + 1] = pulse

# -- Sensor: two points --
sensor_mask = np.zeros(Nx)
sensor_mask[Nx // 4] = 1  # left sensor
sensor_mask[3 * Nx // 4] = 1  # right sensor
sensor = kSensor(mask=sensor_mask)

# %% Run simulation
result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")

print(f"Sensor data shape: {result['p'].shape}")  # (2, Nt)

# %% Visualization
t_us = np.arange(kgrid.Nt) * float(kgrid.dt) * 1e6
x_mm = np.arange(Nx) * dx * 1e3

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(x_mm, source.p0)
axes[0].set_xlabel("x (mm)")
axes[0].set_ylabel("Pressure (Pa)")
axes[0].set_title("Initial pressure")

axes[1].plot(x_mm, sound_speed, label="c₀")
ax2 = axes[1].twinx()
ax2.plot(x_mm, density, color="tab:red", label="ρ₀")
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Sound speed (m/s)")
ax2.set_ylabel("Density (kg/m³)")
axes[1].set_title("Medium properties")

axes[2].plot(t_us, result["p"][0], label=f"x = {Nx // 4 * dx * 1e3:.1f} mm")
axes[2].plot(t_us, result["p"][1], label=f"x = {3 * Nx // 4 * dx * 1e3:.1f} mm")
axes[2].set_xlabel("Time (μs)")
axes[2].set_ylabel("Pressure (Pa)")
axes[2].set_title("Sensor signals")
axes[2].legend()

plt.tight_layout()
plt.savefig("example_1d_ivp.png", dpi=150)
plt.show()
