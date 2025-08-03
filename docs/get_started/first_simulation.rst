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

   .. grid-item-card:: :doc:`Grid <kwave.kgrid>`
      :text-align: center

      🕸️

      Computational foundation defining spatial and temporal discretization

   .. grid-item-card:: :doc:`Medium <kwave.kmedium>`
      :text-align: center

      🌊

      Material properties through which acoustic waves propagate

   .. grid-item-card:: :doc:`Source <kwave.ksource>`
      :text-align: center

      📡

      How acoustic energy is introduced into the simulation

   .. grid-item-card:: :doc:`Sensor <kwave.ksensor>`
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

This creates a computational domain suitable for simulating ~15 MHz ultrasound.

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

Now we combine all four components and run the simulation:

.. code-block:: python

   from kwave import kspaceFirstOrder2D
   
   # Run the simulation
   sensor_data = kspaceFirstOrder2D(
       grid=grid,
       medium=medium, 
       source=source,
       sensor=sensor,
       simulation_options={'PMLInside': False, 'PlotSim': False}
   )

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

Next Steps: Explore Real Applications
------------------------------------

Now that you understand the four-component structure, explore these examples to see how the same framework applies to different applications:

**Beginner Examples** (start here):

- :doc:`../examples/ivp_photoacoustic_waveforms/README` - See how 2D and 3D wave propagation differs
- :doc:`../examples/us_defining_transducer/README` - Learn about ultrasound transducers

**Medical Imaging Applications**:

- :doc:`../examples/us_bmode_linear_transducer/README` - Full B-mode ultrasound imaging pipeline
- :doc:`../examples/pr_2D_FFT_line_sensor/README` - Photoacoustic image reconstruction

**Advanced Transducer Modeling**:

- :doc:`../examples/at_array_as_source/README` - Complex array transducers without staircasing
- :doc:`../examples/at_focused_bowl_3D/README` - Focused ultrasound applications

**Acoustic Field Analysis**:

- :doc:`../examples/us_beam_patterns/README` - Understand beam formation and focusing
- :doc:`../examples/sd_focussed_detector_2D/README` - Sensor directivity effects

Each example builds on the same four-component framework but demonstrates different aspects of acoustic simulation. The key insight is that no matter how complex the application, every k-Wave simulation follows this same logical structure.