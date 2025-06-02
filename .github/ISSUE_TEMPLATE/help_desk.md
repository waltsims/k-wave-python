---
name: Help Desk
about: Despite multiple attempts you repeatedly fail to solve an issue?  
title: "[Help Desk]"
labels: help desk
assignees: ''

---
 
## Please note
<!-- A brief description of what you expected to happen. -->
- We may ignore any issues due to a lack of time and apologize.
- Please first always refer to the [documentation](https://k-wave-python.readthedocs.io/en/latest/index.html), or specific [examples](https://github.com/waltsims/k-wave-python/tree/master/examples) before raising an issue
- Code dumps "can you help, here is my code" will be closed without comment 
- Be as specific as possible and write in detail what you have already tried, add links and any reference that may help us understand 
- Run your simulation with verbosity level set to two
```python
execution_options = SimulationExecutionOptions(is_gpu_simulation=True, verbose_level=2)
```

## Environment information
<!-- Details of your environment are crucial for reproducing the error. -->
- **k-wave-python Version** (e.g. 0.3.3.)
- **Operating System** (e.g., Windows 10, Ubuntu 20.04, macOS Monterey):
- **Python Version** (output of `python --version`):
- **CUDA Version** (output of `nvcc --version` or `nvidia-smi` if applicable):
- **Other Relevant Software/Hardware Info** (e.g., GPU model, driver version):

## Description of your problem
<!-- A clear and concise description of what the error is. -->
- Please state what your problem is and be specific but concise.

## Steps to reproduce
<!-- Step-by-step process to reproduce the behavior -->
- What script/example did you run? Please provide necessary information.  

## Additional context
<!-- Any additional information that can help us diagnose the issue effectively. -->
- Please add additional resources here e.g. screenshot or links/ solution you have already refered to. 

---
