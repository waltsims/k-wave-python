Development Environment
=======================

Currently, this package serves as an interface to the cpp binaries of k-Wave.
For this reason, the k-Wave binaries are packaged with the code in this repository.
The k-Wave binaries can currently be found on the `k-Wave download page <http://www.k-wave.org/download.php>`_.

In order to correctly set up your development environment for this repository, clone the repository from github, and install the project dependencies..

.. code-block:: bash

    git clone https://github.com/waltsims/k-wave-python
    cd k-wave-python
    pip install -r requirements.txt

Next, download all k-Wave binaries from the `k-Wave download page <http://www.k-wave.org/download.php>`_.

Lastly, place the contents of the linux-binaries, and windows-executables directories in the project directory structure under ``kwave/bin/linux/`` and ``kwave/bin/windows`` respectively.

Now your development environment has been set up, and you can run and build the ``k-Wave-python`` codebase.

If you have any issues or questions, please join the `k-Wave-python Development Telegram group <https://t.me/+ILL4yGgcX0A2Y2Y6>`_ to discuss. We look forward to interacting with you.