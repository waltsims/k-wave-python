First Steps
===========

.. code-block:: bash

    export LD_LIBRARY_PATH=;
    export OMP_PLACES=cores;  
    export OMP_PROC_BIND=SPREAD;

    binary_name=kspaceFirstOrder-CUDA

    <PATH_TO_KWAVE_BINARIES_FOLDER>/$binary_name \
        -i <input_filename> \
        -o <output_filename> \
        --p_raw

