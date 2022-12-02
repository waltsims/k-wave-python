Welcome to k-Wave-python's documentation!
=========================================

k-Wave is an open source acoustics toolbox for MATLAB and C++ developed by Bradley Treeby and Ben Cox (University College London) and Jiri Jaros (Brno University of Technology). The software is designed for time domain acoustic and ultrasound simulations in complex and tissue-realistic media. The simulation functions are based on the k-space pseudospectral method and are both fast and easy to use.

.. mdinclude:: ../README.md
   :start-line: 5
   :end-line: -6

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   get_started/contrib
   get_started/license

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/development_environment

.. toctree::
   :caption: kwave
   :titlesonly:
   :maxdepth: 4
   :hidden:

   kwave.data
   kwave.enums
   kwave.executor
   kwave.kWaveSimulation
   kwave.kgrid
   kwave.kmedium
   kwave.ksensor
   kwave.ksource
   kwave.kspaceFirstOrder
   kwave.kspaceFirstOrder2D
   kwave.kspaceFirstOrder3D
   kwave.kspaceFirstOrderAS
   kwave.ktransducer
   kwave.options
   kwave.recorder
   kwave.utils
   kwave.kWaveSimulation_helper
   kwave.reconstruction

