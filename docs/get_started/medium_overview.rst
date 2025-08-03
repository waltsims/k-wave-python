Medium: Material Properties
===========================

The medium defines the material properties through which acoustic waves propagate. It forms the second of the four core components that define every k-Wave simulation.

Key Properties
--------------

**Sound Speed** (``sound_speed``): Speed at which acoustic waves travel through the material [m/s]. Required for all simulations.

**Density** (``density``): Material density [kg/m³]. Affects impedance and reflection at interfaces.

**Absorption** (``alpha_coeff``, ``alpha_power``): Acoustic energy loss. Follows power-law model: ``α = α₀ × f^y`` where *f* is frequency.

**Nonlinearity** (``BonA``): B/A parameter controlling nonlinear wave-steepening effects.

Homogeneous vs Heterogeneous
----------------------------

**Homogeneous**: Uniform properties throughout domain:

.. code-block:: python

   medium = kWaveMedium(sound_speed=1540)  # Water-like medium

**Heterogeneous**: Spatially varying properties:

.. code-block:: python

   medium = kWaveMedium(sound_speed=c_map)  # c_map has same size as grid

Common Media
------------

.. code-block:: python

   # Water (lossless)
   water = kWaveMedium(sound_speed=1500, density=1000)
   
   # Soft tissue (with absorption)
   tissue = kWaveMedium(
       sound_speed=1540,
       density=1000,
       alpha_coeff=0.75,  # dB/(MHz·cm)
       alpha_power=1.5,
   )

For detailed material-property databases and advanced absorption modeling, see :doc:`../fundamentals/understanding_media`. 