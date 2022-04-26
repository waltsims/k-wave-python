# k-Wave-python

This project is a Python interface to the [k-Wave simulation binaries](http://www.k-wave.org/download.php).

The documentation can be found [here](http://waltersimson.com/k-wave-python/)
## Installation

```commandline
git clone https://github.com/waltsims/k-wave-python
cd k-wave-python
python setup.py install
```
jk
## Getting started
![](docs/images/example_bmode.png)

After installation, run the B-mode reconstruction example in the `examples` directory of the repository:

```bash
pip install -r example_requirements.txt
python3 examples/bmode_reconstruction_example.py
```

This example file steps through the process of:
 1. Generating a simulation medium
 2. Configuring a transducer
 3. Running the simulation
 4. Reconstructing the simulation

## Development

If your're enjoying k-Wave-python and want to contribute, development instructions can be found [here](https://waltersimson.com/k-wave-python/development/development_environment.html).
