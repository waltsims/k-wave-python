Grid: Computational Foundation
==============================

The computational grid defines the spatial and temporal discretization for k-Wave simulations. It forms one of the four core components (Grid, Medium, Source, Sensor) that define every acoustic simulation.

Key Concepts
------------

**Grid Spacing**: Determines simulation accuracy. Use :math:`\Delta x \le \lambda_\mathrm{min}/3` where :math:`\lambda_\mathrm{min}` is the smallest wavelength. k‑Wave: Nyquist limit = 2 PPW; k‑Wave defaults to 3 PPW as a safer minimum — increase to ~6–15+ PPW for nonlinear/heterogeneous problems.

**Grid Size**: Total number of points. Larger grids provide finer resolution but increase computational cost significantly.

**Time Step**: Computed from the CFL condition (e.g., :math:`\Delta t = \mathrm{CFL}\cdot \min(\Delta x)/c_\mathrm{max}`). You can override CFL for custom temporal sampling. See :doc:`../kwave.options`.

Quick Start
-----------

.. code-block:: python

   from kwave.kgrid import kWaveGrid
   
   # 2 cm × 1 cm domain, 100 µm resolution
   grid = kWaveGrid([200, 100], [100e-6, 100e-6])

For detailed tutorials and parameter-selection guidelines, see :doc:`../fundamentals/understanding_grids`. 