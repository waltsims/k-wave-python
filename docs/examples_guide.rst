Examples
========

k-Wave-python examples are organized by topic. Each example uses the ``setup()/run()`` pattern and can be run directly:

.. code-block:: bash

   uv run python examples/ivp_homogeneous_medium.py

No GPU required — all examples run on CPU with NumPy.

Start with the :doc:`get_started/first_simulation` tutorial, then explore by topic:

Initial Value Problems (IVP)
----------------------------

Learn fundamental wave physics using initial pressure distributions.

- :ghfile:`Homogeneous medium <examples/ivp_homogeneous_medium.py>` — simplest 2D simulation
- :ghfile:`Heterogeneous medium <examples/ivp_heterogeneous_medium.py>` — spatially varying sound speed and density
- :ghfile:`1D simulation <examples/ivp_1D_simulation.py>` — reflections at impedance boundaries
- :ghfile:`3D simulation <examples/ivp_3D_simulation.py>` — extending to three dimensions
- :ghfile:`Binary sensor mask <examples/ivp_binary_sensor_mask.py>` — defining sensor regions
- :ghfile:`Recording particle velocity <examples/ivp_recording_particle_velocity.py>` — velocity field output
- :ghfile:`Photoacoustic waveforms <examples/ivp_photoacoustic_waveforms.py>` — 2D vs 3D wave spreading
- :ghfile:`Loading external image <examples/ivp_loading_external_image.py>` — image-based initial pressure

Time-Varying Pressure Sources (TVSP)
-------------------------------------

Transducer-driven simulations with time-varying sources.

- :ghfile:`Monopole source <examples/tvsp_homogeneous_medium_monopole.py>` — single point source
- :ghfile:`Dipole source <examples/tvsp_homogeneous_medium_dipole.py>` — dipole radiation pattern
- :ghfile:`Steering linear array <examples/tvsp_steering_linear_array.py>` — beam steering with delays
- :ghfile:`Snell's law <examples/tvsp_snells_law.py>` — refraction at material boundaries
- :ghfile:`Doppler effect <examples/tvsp_doppler_effect.py>` — moving source frequency shift
- :ghfile:`3D simulation <examples/tvsp_3D_simulation.py>` — time-varying source in 3D

Sensor Directivity (SD)
------------------------

How sensor geometry affects measurements.

- :ghfile:`Directivity modelling 2D <examples/sd_directivity_modelling_2D.py>` — finite sensor size effects
- :ghfile:`Directivity modelling 3D <examples/sd_directivity_modelling_3D.py>` — 3D directivity
- :ghfile:`Focused detector 2D <examples/sd_focussed_detector_2D.py>` — directional sensitivity
- :ghfile:`Focused detector 3D <examples/sd_focussed_detector_3D.py>` — 3D focused sensors
- :ghfile:`Directional array elements <examples/sd_directional_array_elements.py>` — multi-element arrays

Photoacoustic Reconstruction (PR)
----------------------------------

Image reconstruction from sensor data.

- :ghfile:`2D FFT line sensor <examples/pr_2D_FFT_line_sensor.py>` — FFT-based reconstruction
- :ghfile:`3D FFT planar sensor <examples/pr_3D_FFT_planar_sensor.py>` — 3D FFT reconstruction

Numerical Analysis (NA)
------------------------

Computational methods and optimization.

- :ghfile:`Controlling the PML <examples/na_controlling_the_PML.py>` — boundary absorption settings
- :ghfile:`Source smoothing <examples/na_source_smoothing.py>` — spatial filtering of sources
- :ghfile:`Filtering part 1 <examples/na_filtering_part_1.py>` — grid-imposed frequency limits
- :ghfile:`Filtering part 2 <examples/na_filtering_part_2.py>` — filtering source signals
- :ghfile:`Filtering part 3 <examples/na_filtering_part_3.py>` — combined filtering
- :ghfile:`Modelling nonlinearity <examples/na_modelling_nonlinearity.py>` — nonlinear wave propagation
- :ghfile:`Optimising performance <examples/na_optimising_performance.py>` — speed and memory tips
