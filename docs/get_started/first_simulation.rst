Your First k-Wave Simulation
============================

This guide walks you through creating your first k-Wave simulation, introducing the four core components step by step.

Understanding the Four Components
---------------------------------

Every k-Wave simulation requires exactly four components:

1. **Grid**: Defines the computational domain
2. **Medium**: Specifies material properties  
3. **Source**: Introduces acoustic energy
4. **Sensor**: Records simulation data

The Four Components of Every Simulation
========================================

Every k-Wave simulation is built from four core components that work together to define the acoustic problem:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :doc:`Grid <get_started/grid_overview>`
      :text-align: center

      🕸️

      Computational foundation defining spatial and temporal discretization

   .. grid-item-card:: :doc:`Medium <get_started/medium_overview>`
      :text-align: center

      🌊

      Material properties through which acoustic waves propagate

   .. grid-item-card:: :doc:`Source <get_started/source_overview>`
      :text-align: center

      📡

      How acoustic energy is introduced into the simulation

   .. grid-item-card:: :doc:`Sensor <get_started/sensor_overview>`
      :text-align: center

      🎙️

      Where and what acoustic data is recorded

Understanding these four components is key to mastering k-Wave simulations. Each component page provides conceptual explanations, practical examples, and API reference.


Let's build a simple 2D simulation to see how these work together.

Step 1: Create the Grid
-----------------------

The grid defines where and when we compute the acoustic fields:

.. code-block:: python

   from kwave import kWaveGrid
   
   # Create a 2D grid: 2cm × 2cm domain with 100μm resolution
   Nx, Ny = 200, 200           # 200 × 200 grid points
   dx, dy = 100e-6, 100e-6     # 100 μm spacing
   
   grid = kWaveGrid([Nx, Ny], [dx, dy])

This grid (~100 μm spacing) is suitable up to about 7.5 MHz (≈2 points per wavelength in water). For higher frequencies, reduce dx to maintain ≥2–3 PPW.
Step 2: Define the Medium
-------------------------

The medium specifies how waves propagate through the material:

.. code-block:: python

   from kwave import kWaveMedium
   
   # Simple water-like medium
   medium = kWaveMedium(
       sound_speed=1500,  # m/s - speed of sound in water
       density=1000       # kg/m³ - density of water
   )

For a homogeneous medium, we specify properties as single values that apply everywhere.

Step 3: Create a Source
-----------------------

The source defines how acoustic energy enters the simulation:

.. code-block:: python

   import numpy as np
   from kwave import kSource
   
   # Create initial pressure distribution (photoacoustic-style)
   source = kSource()
   
   # Simple circular initial pressure in the center
   x_pos = grid.x_vec
   y_pos = grid.y_vec
   X, Y = np.meshgrid(x_pos, y_pos, indexing='ij')
   
   # 2mm radius circle with 10 kPa initial pressure
   radius = 2e-3  # 2 mm
   center_x, center_y = 0, 0  # Center of domain
   
   initial_pressure = np.zeros(grid.N)
   mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
   initial_pressure[mask] = 10e3  # 10 kPa
   
   source.p0 = initial_pressure

This creates an initial pressure distribution that will propagate outward as acoustic waves.

Step 4: Define Sensors
----------------------

Sensors specify where we record the acoustic data:

.. code-block:: python

   from kwave import kSensor
   
   # Create sensors around the edge to capture the expanding wave
   sensor_mask = np.zeros(grid.N)
   sensor_mask[0, :] = 1     # Top edge
   sensor_mask[-1, :] = 1    # Bottom edge  
   sensor_mask[:, 0] = 1     # Left edge
   sensor_mask[:, -1] = 1    # Right edge
   
   sensor = kSensor(mask=sensor_mask, record=['p'])

Step 5: Run the Simulation
--------------------------

Now we combine all four components and run:

.. code-block:: python

   from kwave.kspaceFirstOrder import kspaceFirstOrder

   # Run with the NumPy/CuPy backend in Python
   sensor_data = kspaceFirstOrder(
       grid, medium, source, sensor,
       pml_inside=False,
       quiet=True,
   )

.. note::

   ``kspaceFirstOrder()`` is the unified entry point introduced in v0.6.0.
   It auto-detects dimensionality from the grid and replaces the legacy
   ``kspaceFirstOrder2D`` / ``3D`` functions. See :doc:`/get_started/new_api`
   for details.

   The legacy functions still work but emit warnings.

Step 6: Visualize Results
-------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot the recorded pressure at sensors vs time
   plt.figure(figsize=(10, 6))
   plt.imshow(sensor_data['p'], aspect='auto', extent=[
       0, grid.Nt * grid.dt * 1e6,  # Time in μs
       0, sensor_data['p'].shape[0]  # Sensor number
   ])
   plt.xlabel('Time (μs)')
   plt.ylabel('Sensor Number')
   plt.title('Recorded Pressure at Boundary Sensors')
   plt.colorbar(label='Pressure (Pa)')
   plt.show()

What Just Happened?
-------------------

1. The initial pressure distribution creates acoustic waves
2. These waves propagate outward through the medium at 1500 m/s
3. When waves reach the boundary sensors, pressure is recorded
4. The result shows the acoustic wavefront arriving at different sensors over time

Next Steps
----------

Explore the :doc:`/examples_guide` to see the four-component framework applied to different problems — from heterogeneous media to beam steering, Doppler effects, and image reconstruction.