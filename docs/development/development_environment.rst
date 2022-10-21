Development Environment
=======================

Currently, this package serves as an interface to the cpp binaries of k-Wave.
For this reason, the k-Wave binaries are packaged with the code in this repository.
The k-Wave binaries can currently be found on the `k-Wave download page <http://www.k-wave.org/download.php>`_.

In order to correctly set up your development environment for this repository, clone the repository from github, and install the project dependencies..

.. code-block:: bash

    git clone https://github.com/waltsims/k-wave-python
    cd k-wave-python
    pip install -e .

Next, download all k-Wave binaries from the `k-Wave download page <http://www.k-wave.org/download.php>`_.

Lastly, place the contents of the linux-binaries, and windows-executables directories in the project directory structure under ``kwave/bin/linux/``, ``kwave/bin/darwin`` and ``kwave/bin/windows`` respectively.
You will have to create the directory structure yourself.

With this, you are ready to develop k-Wave-python
If you have any issues or questions, please post them on the `GitHub discussions page <https://github.com/waltsims/k-wave-python/discussions>`_ to discuss. We look forward to interacting with you.