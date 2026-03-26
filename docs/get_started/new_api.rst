Unified API (v0.6.0)
====================

Starting with v0.6.0, ``kspaceFirstOrder()`` is the preferred way to run
simulations.  It replaces the legacy ``kspaceFirstOrder2D``,
``kspaceFirstOrder3D``, and their GPU variants with a single function that
auto-detects dimensionality from the grid.

.. contents:: On this page
   :local:
   :depth: 2

Quick Start
-----------

.. code-block:: python

   from kwave.kgrid import kWaveGrid
   from kwave.kmedium import kWaveMedium
   from kwave.ksource import kSource
   from kwave.ksensor import kSensor
   from kwave.kspaceFirstOrder import kspaceFirstOrder
   import numpy as np

   # Setup (works for 1D, 2D, or 3D — just change grid dimensions)
   kgrid = kWaveGrid([128, 128], [0.1e-3, 0.1e-3])
   kgrid.makeTime(1500)

   medium = kWaveMedium(sound_speed=1500, density=1000)

   source = kSource()
   source.p0 = np.zeros((128, 128))
   source.p0[64, 64] = 1.0

   sensor = kSensor(mask=np.ones((128, 128), dtype=bool))

   # Run with the NumPy/CuPy solver in Python
   result = kspaceFirstOrder(kgrid, medium, source, sensor)

   print(result["p"].shape)       # (16384, Nt) — time series at each sensor
   print(result["p_final"].shape) # (128, 128)  — final pressure field

Backend and Device
------------------

Two independent choices control how the simulation runs:

- **backend** selects the simulation engine:

  - ``"python"`` (default) — pure Python solver using NumPy or CuPy.
    No external dependencies beyond NumPy.
  - ``"cpp"`` — serializes to HDF5 and invokes the pre-compiled C++ binary.
    Requires FFTW (``brew install fftw`` on macOS).

- **device** selects the hardware:

  - ``"cpu"`` (default) — NumPy for ``backend="python"``, OMP binary for
    ``backend="cpp"``.
  - ``"gpu"`` — CuPy for ``backend="python"`` (requires CuPy + CUDA),
    CUDA binary for ``backend="cpp"``.

.. code-block:: python

   # NumPy on CPU (default)
   result = kspaceFirstOrder(kgrid, medium, source, sensor)

   # CuPy on GPU (requires CuPy + CUDA)
   result = kspaceFirstOrder(kgrid, medium, source, sensor, device="gpu")

   # C++ binary on CPU (requires FFTW)
   result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp")

   # C++ CUDA binary on GPU
   result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", device="gpu")

PML Options
-----------

The perfectly-matched layer (PML) absorbs outgoing waves at the domain
boundary.  Three controls:

.. code-block:: python

   # Fixed size (default: 20 grid points on all sides)
   result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=20)

   # Per-dimension sizes
   result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=(10, 15))

   # Auto-optimal size (computed from grid via FFT analysis)
   result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size="auto")

   # PML inside the user domain (saves memory, but PML overlaps your grid)
   result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_inside=True)

By default (``pml_inside=False``), the grid is automatically expanded so
the PML sits outside your domain.  Full-grid output fields (``_final``,
``_max``, etc.) are cropped back to the original size.

Save-Only Mode (Cluster Submission)
------------------------------------

Generate HDF5 input files for the C++ binary without running it:

.. code-block:: python

   result = kspaceFirstOrder(
       kgrid, medium, source, sensor,
       backend="cpp",
       save_only=True,
       data_path="/path/to/output",
   )
   print(result["input_file"])   # path to HDF5 input
   print(result["output_file"])  # path where C++ will write results

Full Parameter Reference
------------------------

.. autofunction:: kwave.kspaceFirstOrder.kspaceFirstOrder

Migrating from Legacy API
--------------------------

The ``kwave.compat`` module provides ``options_to_kwargs()`` to convert
old ``SimulationOptions`` / ``SimulationExecutionOptions`` to keyword
arguments:

.. code-block:: python

   from kwave.compat import options_to_kwargs
   from kwave.options.simulation_options import SimulationOptions
   from kwave.options.simulation_execution_options import SimulationExecutionOptions

   sim_opts = SimulationOptions(smooth_p0=False, pml_inside=True)
   exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

   kwargs = options_to_kwargs(simulation_options=sim_opts, execution_options=exec_opts)
   result = kspaceFirstOrder(kgrid, medium, source, sensor, **kwargs)

The legacy ``kspaceFirstOrder2D`` and ``kspaceFirstOrder3D`` functions
continue to work but emit ``DeprecationWarning``.

Environment Variables
---------------------

Two environment variables override the ``backend`` and ``device`` parameters
for all ``kspaceFirstOrder()`` calls.  Useful for CI and cluster environments:

- ``KWAVE_BACKEND`` — set to ``"python"`` or ``"cpp"``
- ``KWAVE_DEVICE`` — set to ``"cpu"`` or ``"gpu"``

.. code-block:: bash

   # Run all simulations with the Python backend on CPU
   KWAVE_BACKEND=python KWAVE_DEVICE=cpu python my_script.py
