Sensor: Data Acquisition
========================

The sensor defines where and what acoustic data is recorded during simulation. It forms the fourth and final component of the four core elements that define every k-Wave simulation.

Sensor Positioning
------------------

**Binary Mask**: Define sensor positions using a binary array matching grid dimensions:

.. code-block:: python

   sensor = kSensor(mask=sensor_mask)  # 1 where sensors are located, 0 elsewhere

**Cartesian Points**: Specify exact sensor coordinates:

.. code-block:: python

   sensor_points = np.array([[x1, y1], [x2, y2], ...])  # [N_sensors × N_dims]
   sensor = kSensor(mask=sensor_points)

Recording Options
-----------------

Control what acoustic parameters to record using the ``record`` parameter:

.. code-block:: python

   # Record pressure (default)
   sensor = kSensor(mask=sensor_mask, record=['p'])
   
   # Record multiple parameters
   sensor = kSensor(mask=sensor_mask, record=['p', 'u', 'p_final'])

**Available parameters**:
- ``'p'``: Pressure at each time step
- ``'p_final'``: Final pressure field
- ``'u'``: Particle velocity components (ux, uy, uz)
- ``'I'``: Acoustic intensity

Advanced Features
-----------------

**Frequency Response**: Apply sensor bandwidth characteristics:

.. code-block:: python

   sensor.frequency_response = [center_freq, bandwidth_percent]

Common Patterns
---------------

.. code-block:: python

   # Point sensors at specific locations
   sensor_positions = np.array([[0.01, 0.01], [0.02, 0.01]])
   sensor = kSensor(mask=sensor_positions, record=['p'])
   
   # Line of sensors (imaging array)
   sensor_mask = np.zeros(grid.N)
   sensor_mask[64, :] = 1  # Horizontal line
   sensor = kSensor(mask=sensor_mask)

For advanced sensor configurations and reconstruction techniques, see :doc:`../fundamentals/understanding_sensors`. 