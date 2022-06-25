# k-Wave-python

This project is a Python implementation of most of the MATLAB toolbox k-wave as well as an interface to the pre-compiled v1.3 of k-Wave simulation binaries which support NVIDIA sm 3.0 to sm 7.5.

## Mission

With this project, we hope to increase accessibility and reproducablitiy of k-Wave simulation for medical imaging, algorithmic prototyping and testing. Many tools and methods of k-wave can be found here, but this project has and will continue to diverge from the original k-wave APIs in order to leverage pythonic practices.

## Documentation

The documentation for k-wave-python can be found [here](http://waltersimson.com/k-wave-python/)

## Installation

```bash
pip install k-wave-python
```

Currently, we are looking for beta testers on Windows.


## Getting started
![](docs/images/example_bmode.png)

After installation, run the B-mode reconstruction example in the `examples` directory of the repository:

```bash
git clone https://github.com/waltsims/k-wave-python
pip install -r example_requirements.txt
python3 examples/bmode_reconstruction_example.py
```

This example file steps through the process of:
 1. Generating a simulation medium
 2. Configuring a transducer
 3. Running the simulation
 4. Reconstructing the simulation

### Requirements
This example expects an NVIDIA GPU by default to simulate with k-Wave.

To test the reconstruction on a machine without a GPU, set `RUN_SIMULATION` [on line 14 of `bmode_reconstruction_example.py`](https://github.com/waltsims/k-wave-python/blob/master/examples/bmode_reconstruction_example.py#L18) to `False` and the exmaple will run with pre-computed data.

## Development

If you're enjoying k-Wave-python and want to contribute, development instructions can be found [here](https://waltersimson.com/k-wave-python/development/development_environment.html).
If you would like to get involved, open an issue letting us know, or message us on the [k-Wave-python Telegram chat](https://t.me/+ILL4yGgcX0A2Y2Y6) 

## Contact
e-mail [walter.simson@tum.de](mailto:walter.simson@tum.de).
