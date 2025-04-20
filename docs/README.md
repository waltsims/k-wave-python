# k-Wave-python

[![Support](https://dcbadge.vercel.app/api/server/Yq5Qj6D9vN?style=flat)](https://discord.gg/Yq5Qj6D9vN)
[![Documentation Status](https://readthedocs.org/projects/k-wave-python/badge/?version=latest)](https://k-wave-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/waltsims/k-wave-python/graph/badge.svg?token=6ofwtPiDNG)](https://codecov.io/gh/waltsims/k-wave-python)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/waltsims/k-wave-python/master)

This project is a Python implementation of v1.4.0 of the [MATLAB toolbox k-Wave](http://www.k-wave.org/) as well as an
interface to the pre-compiled v1.3 of k-Wave simulation binaries, which support NVIDIA sm 5.0 (Maxwell) to sm 9.0a (Hopper) GPUs.

## Mission

With this project, we hope to increase the accessibility and reproducibility of [k-Wave](http://www.k-wave.org/) simulations
for medical imaging, algorithmic prototyping, and testing. Many tools and methods of [k-Wave](http://www.k-wave.org/) can
be found here, but this project has and will continue to diverge from the original [k-Wave](http://www.k-wave.org/) APIs
to leverage pythonic practices.

## Getting started

![](_static/example_bmode.png)

A large [collection of examples](../examples/) exists to get started with k-wave-python. All examples can be run in Google Colab notebooks with a few clicks. One can begin with e.g. the [B-mode reconstruction example notebook](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/examples/us_bmode_linear_transducer/us_bmode_linear_transducer.ipynb).

This example file steps through the process of:
 1. Generating a simulation medium
 2. Configuring a transducer
 3. Running the simulation
 4. Reconstructing the simulation

## Installation

To install the most recent build of k-Wave-python from PyPI, run:

```bash
pip install k-wave-python
```

After installing the Python package, the required binaries will be downloaded and installed the first time you run a
simulation.

## Development

If you're enjoying k-Wave-python and want to contribute, development instructions can be
found [here](https://k-wave-python.readthedocs.io/en/latest/development/development_environment.html).

## Related Projects

1. [k-Wave](https://github.com/ucl-bug/k-wave): A MATLAB toolbox for the time-domain simulation of acoustic wave fields.
2. [j-wave](https://github.com/ucl-bug/jwave): Differentiable acoustic simulations in JAX.
3. [ADSeismic.jl](https://github.com/kailaix/ADSeismic.jl): a finite difference acoustic simulator with support for AD
   and JIT compilation in Julia.
4. [stride](https://github.com/trustimaging/stride): a general optimisation framework for medical ultrasound tomography.

## Documentation

The documentation for k-wave-python can be found [here](https://k-wave-python.readthedocs.io/en/latest/).

## Citation
```bibtex
@software{k-Wave-Python,
author = {Yagubbbayli, Farid and Sinden, David and Simson, Walter},
license = {GPL-3.0},
title = {{k-Wave-Python}},
url = {https://github.com/waltsims/k-wave-python}
}
```
## Contact

e-mail [wsimson@stanford.edu](mailto:wsimson@stanford.edu).
