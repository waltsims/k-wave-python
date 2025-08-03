Grid (kWaveGrid): Computational Foundation
===========================================

The computational grid defines the spatial and temporal discretization for k-Wave simulations. It forms one of the four core components (Grid, Medium, Source, Sensor) that define every acoustic simulation.

Key Concepts
------------

**Grid Spacing**: Determines simulation accuracy. Use ``dx ≤ λ_min/3`` where λ_min is the smallest wavelength in your simulation.

**Grid Size**: Total number of points. Larger grids provide finer resolution but increase computational cost significantly.

**Time Step**: Automatically calculated from CFL stability condition. Can be overridden for custom temporal sampling.

Quick Start
-----------

.. code-block:: python

   from kwave import kWaveGrid
   
   # 2D grid: 2cm × 1cm domain, 100μm resolution
   grid = kWaveGrid([200, 100], [100e-6, 100e-6])

For detailed tutorials and parameter selection guidelines, see :doc:`../fundamentals/understanding_grids`.

API Reference
=============

.. automodule:: kwave.kgrid
   :members:
   :undoc-members:
   :show-inheritance:
