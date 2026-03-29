# k-wave-python

[![Support](https://img.shields.io/discord/1234942672418897960?style=flat&logo=discord)](https://discord.gg/your-invite-code)
[![Documentation Status](https://readthedocs.org/projects/k-wave-python/badge/?version=latest)](https://k-wave-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/waltsims/k-wave-python/graph/badge.svg?token=6ofwtPiDNG)](https://codecov.io/gh/waltsims/k-wave-python)

A Python implementation of [k-Wave](http://www.k-wave.org/) — an acoustics toolbox for time-domain simulation of acoustic wave fields. Includes a pure NumPy/CuPy solver (`backend="python"`) and an interface to the pre-compiled k-Wave C++ binaries (`backend="cpp"`) with NVIDIA GPU support (sm 5.0–9.0a).

## Mission

Increase the accessibility and reproducibility of [k-Wave](http://www.k-wave.org/) simulations for medical imaging, algorithmic prototyping, and testing.

## Getting started

![](_static/example_bmode.png)

A [collection of examples](../examples/) covers common simulation scenarios. Run any example locally:

```bash
uv run examples/ivp_homogeneous_medium.py
```

No GPU required — all examples run on CPU with NumPy.

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add k-wave-python
```

Or with pip:

```bash
pip install k-wave-python
```

## Development

Development instructions can be found [here](https://k-wave-python.readthedocs.io/en/latest/development/development_environment.html).

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
author = {Yagubbayli, Farid and Sinden, David and Simson, Walter},
license = {GPL-3.0},
title = {{k-Wave-Python}},
url = {https://github.com/waltsims/k-wave-python}
}
```
## Contact

e-mail [walter.a.simson@gmail.com](mailto:walter.a.simson@gmail.com).
