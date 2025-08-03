Examples: Learning Path
=======================

The k-Wave Python examples are organized to help you progress from basic wave physics to advanced applications. Each example demonstrates the four-component framework (Grid, Medium, Source, Sensor) with increasing complexity.

Start with the :doc:`get_started/first_simulation` tutorial, then follow this suggested learning path:

Basic Wave Propagation (IVP - Initial Value Problems)
-----------------------------------------------------

**Start Here**: Learn fundamental wave physics using initial pressure distributions (the simplest source type).

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Core Concept
     - Topics
   * - :ghfile:`Photoacoustic Waveforms <examples/ivp_photoacoustic_waveforms/>`
     - 2D vs 3D wave propagation physics
     - **IVP** • Wave spreading • Compact support

Simple Transducers & Sources
-----------------------------

**Next Step**: Introduction to time-varying sources and practical transducer modeling.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Core Concept
     - Topics
   * - :ghfile:`Defining Transducers <examples/us_defining_transducer/>`
     - Basic ultrasound transducer setup
     - **US** • Transducer basics • Time-varying sources
   * - :ghfile:`Circular Piston 3D <examples/at_circular_piston_3D/>`
     - Simple focused geometry
     - **AT** • 3D focusing • Geometric sources
   * - :ghfile:`Circular Piston (Axisymmetric) <examples/at_circular_piston_AS/>`
     - Computational efficiency with symmetry
     - **AT** • Axisymmetric • Computational optimization

Medical Imaging Applications
----------------------------

**Practical Applications**: See how basic concepts combine into real-world medical imaging systems.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Application
     - Topics
   * - :ghfile:`Beam Patterns <examples/us_beam_patterns/>`
     - Understanding acoustic beam formation
     - **US** • Beam focusing • Field patterns
   * - :ghfile:`B-mode Linear Transducer <examples/us_bmode_linear_transducer/>`
     - Complete ultrasound imaging pipeline
     - **US** • Medical imaging • Signal processing
   * - :ghfile:`2D FFT Line Sensor <examples/pr_2D_FFT_line_sensor/>`
     - Photoacoustic image reconstruction
     - **PR** • Image reconstruction • FFT methods
   * - :ghfile:`2D Time Reversal Line Sensor <examples/pr_2D_TR_line_sensor/>`
     - Alternative reconstruction approach
     - **PR** • Time reversal • Reconstruction

Advanced Transducer Modeling (AT - Array Transducers)
-----------------------------------------------------

**Complex Geometries**: Learn sophisticated techniques for modeling complex transducer arrays using Cartesian space methods.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Advanced Technique
     - Topics
   * - :ghfile:`Array as Source <examples/at_array_as_source/>`
     - kWaveArray for complex geometries
     - **AT** • Array modeling • Anti-aliasing
   * - :ghfile:`Array as Sensor <examples/at_array_as_sensor/>`
     - Complex sensor array geometries
     - **AT** • Sensor arrays • Flexible positioning
   * - :ghfile:`Linear Array Transducer <examples/at_linear_array_transducer/>`
     - Multi-element linear arrays
     - **AT** • Linear arrays • Element spacing
   * - :ghfile:`Focused Bowl 3D <examples/at_focused_bowl_3D/>`
     - 3D focused ultrasound therapy
     - **AT** • Therapeutic US • 3D focusing
   * - :ghfile:`Focused Annular Array <examples/at_focused_annular_array_3D/>`
     - Multi-element focused systems
     - **AT** • Annular arrays • Complex focusing

Advanced Imaging & Reconstruction (PR - Pressure/Photoacoustic Reconstruction)
-------------------------------------------------------------------------------

**3D Reconstruction**: Advanced reconstruction techniques for photoacoustic and pressure field imaging.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Reconstruction Method
     - Topics
   * - :ghfile:`3D FFT Planar Sensor <examples/pr_3D_FFT_planar_sensor/>`
     - 3D FFT-based reconstruction
     - **PR** • 3D imaging • Planar arrays
   * - :ghfile:`3D Time Reversal Planar <examples/pr_3D_TR_planar_sensor/>`
     - 3D time reversal reconstruction
     - **PR** • 3D time reversal • Volumetric imaging
   * - :ghfile:`B-mode Phased Array <examples/us_bmode_phased_array/>`
     - Advanced ultrasound beamforming
     - **US** • Phased arrays • Electronic steering

Sensor Physics & Directivity (SD - Sensor Directivity)
------------------------------------------------------

**Sensor Modeling**: Understanding how sensor size, shape, and directivity affect measurements.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Physics Concept
     - Topics
   * - :ghfile:`Sensor Directivity 2D <examples/sd_directivity_modelling_2D/>`
     - How sensor size affects measurements
     - **SD** • Directivity • Finite sensor size
   * - :ghfile:`Focused Detector 2D <examples/sd_focussed_detector_2D/>`
     - Directional sensor sensitivity
     - **SD** • Focused detection • Sensor design
   * - :ghfile:`Focused Detector 3D <examples/sd_focussed_detector_3D/>`
     - 3D focused sensor modeling
     - **SD** • 3D detection • Sensor focusing

Computational Optimization (NA - Numerical Analysis)
----------------------------------------------------

**Advanced Numerics**: Optimize simulations and understand computational aspects.

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Example
     - Optimization Topic
     - Topics
   * - :ghfile:`Controlling the PML <examples/na_controlling_the_pml/>`
     - Boundary conditions and efficiency
     - **NA** • PML boundaries • Computational domains

Understanding the Prefixes
--------------------------

- **IVP** = Initial Value Problems (wave propagation from initial pressure)
- **US** = Ultrasound (medical and therapeutic ultrasound applications)  
- **AT** = Array Transducers (complex geometries using Cartesian space methods)
- **PR** = Pressure/Photoacoustic Reconstruction (image reconstruction techniques)
- **SD** = Sensor Directivity (sensor physics and measurement effects)
- **NA** = Numerical Analysis (computational optimization and methods)

Learning Strategy
-----------------

1. **Start with IVP**: Understand basic wave physics
2. **Move to simple US/AT**: Learn transducer basics
3. **Apply to imaging**: See concepts in real applications (US, PR)
4. **Master advanced AT**: Handle complex geometries
5. **Understand sensors**: Learn about measurement physics (SD)
6. **Optimize**: Improve computational efficiency (NA)

Each example builds on the four-component framework, but with increasing sophistication in how the components are configured and used.