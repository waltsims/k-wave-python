# Kwave.py

This project is a Python interface to the [kWave simulation binaries](http://www.k-wave.org/download.php).

The documentation can be found [here](http://waltersimson.com/k-wave-python/)
## Installation

```commandline
git clone https://github.com/waltsims/k-wave-python
cd k-wave-python
pip install -r requirements.txt
```


## Getting started

Currently, this project creates the HDF5 file that can be used to run
the accelerated kWave binaries.

```commandline
export LD_LIBRARY_PATH=;
export OMP_PLACES=cores;  
export OMP_PROC_BIND=SPREAD;

binary_name=kspaceFirstOrder-CUDA

<PATH_TO_KWAVE_BINARIES_FOLDER>/$binary_name \
    -i <input_filename> \
    -o <output_filename> \
    --p_raw
```
