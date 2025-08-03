Welcome to k-Wave-python's documentation!
=========================================

k-Wave is an open source acoustics toolbox for MATLAB and C++ developed by Bradley Treeby and Ben Cox (University College London) and Jiri Jaros (Brno University of Technology). The software is designed for time domain acoustic and ultrasound simulations in complex and tissue-realistic media. The simulation functions are based on the k-space pseudospectral method and are both fast and easy to use.

.. mdinclude:: README.md
   :start-line: 5
   :end-line: -3

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

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   get_started/first_simulation
   examples_guide

.. toctree::
   :maxdepth: 2
   :caption: Development & Contributing
   :hidden:

   development/development_environment
   get_started/contrib

.. toctree::
   :maxdepth: 1
   :caption: License
   :hidden:

   get_started/license

.. toctree::
   :caption: Core Components
   :titlesonly:
   :maxdepth: 2
   :hidden:

   kwave.kgrid
   kwave.kmedium
   kwave.ksource
   kwave.ksensor

.. toctree::
   :caption: Simulation & Analysis
   :titlesonly:
   :maxdepth: 4
   :hidden:

   kwave.kWaveSimulation
   kwave.kspaceFirstOrder2D
   kwave.kspaceFirstOrder3D
   kwave.kspaceFirstOrderAS
   kwave.ktransducer
   kwave.reconstruction

.. toctree::
   :caption: Utilities & Advanced
   :titlesonly:
   :maxdepth: 4
   :hidden:

   kwave.data
   kwave.enums
   kwave.executor
   kwave.options
   kwave.recorder
   kwave.utils
   kwave.kWaveSimulation_helper

