Source: Acoustic Input
======================

The source defines how acoustic energy is introduced into the simulation. It forms the third of the four core components that define every k-Wave simulation.

Source Types
------------

**Initial Pressure** (``p0``): Initial pressure distribution (e.g., photoacoustic imaging):

.. code-block:: python

   source.p0 = initial_pressure_map  # Same size as grid

**Time-Varying Pressure** (``p``, ``p_mask``): Pressure sources that vary in time (e.g., ultrasound transducers):

.. code-block:: python

   source.p_mask = transducer_positions  # Binary mask
   source.p = pressure_time_series       # [N_source_points × N_time_steps]

**Velocity Sources** (``ux``, ``uy``, ``uz``, ``u_mask``): Particle-velocity sources.

**Stress Sources** (``sxx``, ``syy`` …): For elastic-wave simulations.

Source Modes
------------

**Additive** (default): Source terms are added to the field equations.

**Dirichlet**: Source values are enforced as boundary conditions.

Common Patterns
---------------

.. code-block:: python

   # Photoacoustic initial pressure
   source = kSource()
   source.p0 = initial_pressure_distribution
   
   # Ultrasound transducer
   source = kSource()
   source.p_mask = transducer_mask
   source.p = tone_burst_signal

For transducer modeling and advanced source configurations, see :doc:`../fundamentals/understanding_sources`. 