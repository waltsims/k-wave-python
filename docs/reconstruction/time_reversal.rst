Time Reversal Reconstruction
========================

The :class:`TimeReversal` class provides functionality for time reversal reconstruction in photoacoustic imaging.
It handles the reconstruction of initial pressure distributions from recorded sensor data.

.. tip::
   **Migration from Legacy Time Reversal**
   
   Previous versions used ``sensor.time_reversal_boundary_data`` directly with ``kspaceFirstOrder2D/3D``.
   This approach is now deprecated. Instead, use the new :class:`TimeReversal` class:

   .. code-block:: python

      # Old approach (deprecated)
      sensor.time_reversal_boundary_data = sensor_data
      p0_recon = kspaceFirstOrder2D(kgrid, source, sensor, medium, simulation_options, execution_options)

      # New approach
      sensor.recorded_pressure = sensor_data  # Store recorded data
      tr = TimeReversal(kgrid, medium, sensor)
      p0_recon = tr(kspaceFirstOrder2D, simulation_options, execution_options)

Class Documentation
-----------------

.. autoclass:: kwave.reconstruction.time_reversal.TimeReversal
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
-----------

- Automatic compensation for half-plane recording
- Support for both 2D and 3D reconstructions
- Input validation and error checking
- Clean interface separating data recording from reconstruction

Example Usage
------------

Here's a complete example showing forward simulation and time reversal reconstruction:

.. code-block:: python

   import numpy as np
   from kwave import *

   # Setup grid and medium
   kgrid = kWaveGrid([64, 64], [0.1e-3, 0.1e-3])
   medium = kWaveMedium(sound_speed=1500)
   kgrid.makeTime(medium.sound_speed)

   # Create initial pressure
   source = kSource()
   source.p0 = np.zeros(kgrid.N)
   source.p0[32, 32] = 1

   # Setup sensor
   sensor = kSensor()
   sensor.mask = np.zeros(kgrid.N)
   sensor.mask[0, :] = 1  # Line sensor

   # Forward simulation
   sensor_data = kspaceFirstOrder2D(kgrid, source, sensor, medium, 
                                  simulation_options, execution_options)
   sensor.recorded_pressure = sensor_data["p"].T

   # Time reversal reconstruction
   tr = TimeReversal(kgrid, medium, sensor)
   p0_recon = tr(kspaceFirstOrder2D, simulation_options, execution_options)

Notes
-----

- The compensation factor (default=2.0) accounts for energy loss when recording over a half-plane
- The time array must be explicitly defined (not 'auto') for time reversal
- The sensor mask must have at least one active point
- The sensor mask shape must match the grid dimensions 