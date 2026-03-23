# k-Wave Examples

Each `.py` file is both a runnable script and an interactive notebook.

## Running examples

**As a script:**
```bash
python examples/ivp_1D_simulation.py
```

**Interactively (VS Code / JupyterLab / PyCharm):**
Open any `.py` file — the `# %%` markers define cells you can run one at a time.

**On Google Colab:**
Notebooks are auto-generated in the [`notebooks/`](../notebooks/) directory. Open any example with:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waltsims/k-wave-python/blob/master/notebooks/ivp_1D_simulation.ipynb)

Replace the filename in the URL for other examples.

## Example categories

| Prefix | Topic |
|--------|-------|
| `ivp_` | Initial value problems (photoacoustics) |
| `pr_`  | Photoacoustic reconstruction (FFT, time reversal) |
| `sd_`  | Sensor directivity and focused detectors |
| `at_`  | Acoustic transducers (pistons, bowls, arrays) |
| `us_`  | Ultrasound imaging (B-mode, beam patterns) |
| `na_`  | Numerical analysis (PML, performance) |

## Writing a new example

1. Create `examples/your_example/your_example.py`
2. Start with a markdown header and imports:
   ```python
   # %% [markdown]
   # # Your Example Title
   # One-line description of what this demonstrates.

   # %%
   import numpy as np
   from kwave.kgrid import kWaveGrid
   from kwave.kmedium import kWaveMedium
   from kwave.ksource import kSource
   from kwave.ksensor import kSensor
   from kwave.kspaceFirstOrder import kspaceFirstOrder
   ```
3. Add `# %%` before each logical section (grid, source, simulation, visualization)
4. Use `kspaceFirstOrder()` — not the legacy `kspaceFirstOrder2D/3D`
5. Notebooks are auto-generated from `.py` files on merge to master — don't create `.ipynb` by hand
