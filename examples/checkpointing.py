"""
Checkpointing Example

Demonstrates how to checkpoint a simulation run with the C++ backend, using
the same simulation parameters as the 3D FFT Reconstruction For A Planar
Sensor example.

kspaceFirstOrder(..., backend="cpp", save_only=True) writes the HDF5 input
file without running it.  The C++ binary is then invoked twice on the same
input file with the checkpoint options:

1. the first run stops after ``checkpoint_timesteps`` time steps and saves
   its state to the checkpoint file;
2. the second run restarts from the checkpoint file, completes the
   simulation, and deletes the checkpoint file.

The result is compared with an uninterrupted run of the same simulation.
Alternatively, a checkpoint can be taken after a given wall-clock interval
by passing ``--checkpoint_interval <seconds>`` instead of
``--checkpoint_timesteps <steps>``.  Both values must be integers.
"""

# %%
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import kwave
from kwave.data import Vector
from kwave.executor import Executor
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball

PML_size = 10  # size of the PML in grid points


# %%
def setup():
    """Set up the simulation physics (grid, medium, source).

    Returns:
        tuple: (kgrid, medium, source)
    """
    # create the computational grid (the PML is added outside this grid)
    N = Vector([32, 64, 64]) - 2 * PML_size  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3])  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using make_ball
    ball_magnitude = 10  # [Pa]
    ball_radius = 3  # [grid points]
    p0 = ball_magnitude * make_ball(N, N / 2, ball_radius)

    # smooth the initial pressure distribution and restore the magnitude
    source = kSource()
    source.p0 = smooth(p0.astype(float), restore_max=True)

    # create the time array
    kgrid.makeTime(medium.sound_speed)

    return kgrid, medium, source


# %%
def write_input_file(data_path):
    """Write the HDF5 input file for the C++ binary without running it.

    Returns:
        tuple: (input_file, output_file)
    """
    kgrid, medium, source = setup()

    # define a binary planar sensor
    sensor_mask = np.zeros(tuple(int(n) for n in kgrid.N), dtype=bool)
    sensor_mask[0, :, :] = True
    sensor = kSensor(mask=sensor_mask)

    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend="cpp",
        save_only=True,
        data_path=str(data_path),
        pml_inside=False,
        pml_size=PML_size,
        smooth_p0=False,  # already smoothed manually
    )
    return result["input_file"], result["output_file"]


# %%
def run_binary(input_file, output_file, *, device="cpu", quiet=True, extra_args=()):
    """Invoke the bundled C++ binary directly on an HDF5 input file."""
    binary_name = "kspaceFirstOrder-CUDA" if device == "gpu" else "kspaceFirstOrder-OMP"
    binary = kwave.BINARY_PATH / binary_name
    command = [str(binary), "-i", str(input_file), "-o", str(output_file), *extra_args]
    if quiet:
        command += ["--verbose", "0"]
    subprocess.run(command, check=True, capture_output=quiet)


# %%
def run(device="cpu", quiet=True, checkpoint_timesteps=None):
    """Run the simulation in two halves using a checkpoint, plus a reference run.

    Args:
        checkpoint_timesteps: number of time steps after which the first run
            is checkpointed.  Defaults to half the total number of steps.

    Returns:
        dict with keys:
            'p': sensor data from the checkpointed (restarted) simulation
            'p_ref': sensor data from an uninterrupted simulation
            'files_after_first_run': HDF5 files present after the first run
            'files_after_second_run': HDF5 files present after the second run
    """
    kgrid, _, _ = setup()
    if checkpoint_timesteps is None:
        checkpoint_timesteps = int(kgrid.Nt) // 2

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # checkpointed simulation
        input_file, output_file = write_input_file(tmpdir / "checkpointed")
        checkpoint_file = tmpdir / "checkpointed" / "kwave_checkpoint.h5"
        checkpoint_args = ["--checkpoint_file", str(checkpoint_file), "--checkpoint_timesteps", str(checkpoint_timesteps)]

        # 1st run: stops after checkpoint_timesteps and writes the checkpoint file
        run_binary(input_file, output_file, device=device, quiet=quiet, extra_args=checkpoint_args)
        files_after_first_run = sorted(f.name for f in checkpoint_file.parent.glob("*.h5"))

        # 2nd run: restarts from the checkpoint file and deletes it on completion
        run_binary(input_file, output_file, device=device, quiet=quiet, extra_args=checkpoint_args)
        files_after_second_run = sorted(f.name for f in checkpoint_file.parent.glob("*.h5"))
        p = Executor.parse_executable_output(output_file)["p"]

        # uninterrupted reference simulation
        ref_input_file, ref_output_file = write_input_file(tmpdir / "reference")
        run_binary(ref_input_file, ref_output_file, device=device, quiet=quiet)
        p_ref = Executor.parse_executable_output(ref_output_file)["p"]

    return {
        "p": p,
        "p_ref": p_ref,
        "files_after_first_run": files_after_first_run,
        "files_after_second_run": files_after_second_run,
    }


# %%
if __name__ == "__main__":
    kgrid, _, _ = setup()
    print(f"The simulation has {kgrid.Nt} time steps; checkpointing after {int(kgrid.Nt) // 2}.")

    result = run(quiet=True)

    print("Data directory after the 1st run (including checkpoint):")
    print("\t-", "\n\t- ".join(result["files_after_first_run"]))
    print("Data directory after the 2nd run (checkpoint has been deleted):")
    print("\t-", "\n\t- ".join(result["files_after_second_run"]))

    max_diff = np.max(np.abs(result["p"] - result["p_ref"]))
    print(f"Max difference between checkpointed and uninterrupted runs: {max_diff:.3e} Pa")
