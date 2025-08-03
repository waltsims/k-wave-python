Welcome to k-Wave-python's documentation!
=========================================

k-Wave is an open source acoustics toolbox for MATLAB and C++ developed by Bradley Treeby and Ben Cox (University College London) and Jiri Jaros (Brno University of Technology). The software is designed for time domain acoustic and ultrasound simulations in complex and tissue-realistic media. The simulation functions are based on the k-space pseudospectral method and are both fast and easy to use.

.. mdinclude:: README.md
   :start-line: 5
   :end-line: -3


.. toctree::
   :maxdepth: 2
   :caption: K-Wave Step-by-Step
   :hidden:

   get_started/first_simulation
   get_started/grid_overview
   get_started/medium_overview
   get_started/source_overview
   get_started/sensor_overview
   examples_guide

.. toctree::
   :caption: Python API
   :titlesonly:
   :maxdepth: 4
   :hidden:

   kwave.kgrid
   kwave.kmedium
   kwave.ksource
   kwave.ksensor
   kwave.kWaveSimulation
   kwave.kspaceFirstOrder2D
   kwave.kspaceFirstOrder3D
   kwave.kspaceFirstOrderAS
   kwave.ktransducer
   kwave.reconstruction
   kwave.data
   kwave.enums
   kwave.executor
   kwave.options
   kwave.recorder
   kwave.utils
   kwave.kWaveSimulation_helper

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

