Development Environment
=======================

Overview
--------
k-Wave-python provides both a pure Python/NumPy solver and an interface to pre-compiled C++ binaries. The Python backend works out of the box — no extra dependencies needed.

Environment Setup
-----------------

1. Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_

2. Clone and install:

.. code-block:: bash

    git clone https://github.com/waltsims/k-wave-python
    cd k-wave-python
    uv sync --extra dev --extra test
    uv run pre-commit install

Testing
-------

.. code-block:: bash

    uv run pytest

Most tests run without MATLAB. Tests that need MATLAB reference data will skip gracefully if the references are not available.

.. note::
   **Without MATLAB:** Download pre-generated references from
   `GitHub CI <https://nightly.link/waltsims/k-wave-python/workflows/pytest/master/matlab_reference_test_values.zip>`_
   and unpack into ``tests/matlab_test_data_collectors/python_testers/collectedValues/``.

Test coverage:

.. code-block:: bash

    uv run coverage run

Running Examples
~~~~~~~~~~~~~~~~

.. code-block:: bash

    uv run python examples/ivp_homogeneous_medium.py

Force CPU (skip GPU even if available):

.. code-block:: bash

    KWAVE_FORCE_CPU=1 uv run python examples/ivp_homogeneous_medium.py

Publishing
----------

`uv <https://docs.astral.sh/uv/>`_ is used to build and publish to `PyPI <https://pypi.org/>`_.

.. note::
    Only for maintainers with PyPI write access.

.. code-block:: bash

    uv build
    uv publish --token $PYPI_TOKEN
