---
name: Error Report
about: Create a report to help us diagnose and fix errors with you
title: "[ERROR]"
labels: error
assignees: ''

---
 
## General Instructions
<!-- A brief description of what you expected to happen. -->
- If possible please run your simulation with verbosity level set to two
```python
execution_options = SimulationExecutionOptions(is_gpu_simulation=True, verbose_level=2)
```

## Description of the Error
<!-- A clear and concise description of what the error is. -->
- Please copy & paste your error message here!

## Environment Information
<!-- Details of your environment are crucial for reproducing the error. -->
- **k-wave-python Version** (e.g. 0.3.3.)
- **Operating System** (e.g., Windows 10, Ubuntu 20.04, macOS Monterey):
- **Python Version** (output of `python --version`):
- **CUDA Version** (output of `nvcc --version` or `nvidia-smi` if applicable):
- **Other Relevant Software/Hardware Info** (e.g., GPU model, driver version):

## Steps to Reproduce
<!-- Step-by-step process to reproduce the behavior -->
- What script/example did you run? Please provide py-file name only or "custom" if running own script. 

## Additional Context
<!-- Any additional information that can help us diagnose the issue effectively. -->
- Please copy & paste entire code below. 

---
