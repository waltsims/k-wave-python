from pathlib import Path
from shutil import copy2

import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_ball

EXAMPLE_OUTPUT_DIR = Path("/tmp/example_runs/checkpointing")

# This script demonstrates how to use the checkpointing feature of k-Wave.
# It runs the same simulation twice, with the second run starting from a
# checkpoint file created during the first run.
# This implementation checkpoints based on the number of completed timesteps.
# Alternatively, you can checkpoint based on the simulation time. Below is a
# snippet demonstrating how:
#
# checkpoint_interval = 1  # 1 second
# execution_options = SimulationExecutionOptions(
#   checkpoint_file=checkpoint_filename,
#   checkpoint_interval=checkpoint_interval
# )
# Note: checkpoint timesteps and checkpoint interval must be integers.


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def make_simulation_parameters(directory: Path, checkpoint_timesteps: int):
    """
    See the 3D FFT Reconstruction For A Planar Sensor example for context.
    """
    scale = 1

    # create the computational grid
    PML_size = 10  # size of the PML in grid points
    N = Vector([32, 64, 64]) * scale - 2 * PML_size  # number of grid points
    d = Vector([0.2e-3, 0.2e-3, 0.2e-3]) / scale  # grid point spacing [m]
    kgrid = kWaveGrid(N, d)

    # define the properties of the propagation medium
    medium = kWaveMedium(sound_speed=1500)  # [m/s]

    # create initial pressure distribution using makeBall
    ball_magnitude = 10  # [Pa]
    ball_radius = 3 * scale  # [grid points]
    p0 = ball_magnitude * make_ball(N, N / 2, ball_radius)
    p0 = smooth(p0, restore_max=True)

    source = kSource()
    source.p0 = p0

    # define a binary planar sensor
    sensor = kSensor()
    sensor_mask = np.zeros(N)
    sensor_mask[0] = 1
    sensor.mask = sensor_mask

    input_filename = directory / "kwave_input.h5"
    output_filename = directory / "kwave_output.h5"
    checkpoint_filename = directory / "kwave_checkpoint.h5"

    # set the input arguments
    simulation_options = SimulationOptions(
        save_to_disk=True,
        pml_size=PML_size,
        pml_inside=False,
        smooth_p0=False,
        data_cast="single",
        allow_file_overwrite=True,
        input_filename=input_filename,
        output_filename=output_filename,
    )

    execution_options = SimulationExecutionOptions(checkpoint_file=checkpoint_filename, checkpoint_timesteps=checkpoint_timesteps)
    return kgrid, medium, source, sensor, simulation_options, execution_options


def main():
    # This simulation has 212 timesteps, therefore, the checkpoint is taken at
    # the halfway point.
    checkpoint_timesteps = 106

    EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("kwave_input.h5", "kwave_output.h5", "kwave_checkpoint.h5"):
        (EXAMPLE_OUTPUT_DIR / filename).unlink(missing_ok=True)

    # create the simulation parameters for the 1st run
    kgrid, medium, source, sensor, simulation_options, execution_options = make_simulation_parameters(
        directory=EXAMPLE_OUTPUT_DIR,
        checkpoint_timesteps=checkpoint_timesteps,
    )
    kspaceFirstOrder3D(kgrid, source, sensor, medium, simulation_options, execution_options)
    copy2(simulation_options.input_filename, _next_available_path(EXAMPLE_OUTPUT_DIR / "run_1_kwave_input.h5"))
    copy2(simulation_options.output_filename, _next_available_path(EXAMPLE_OUTPUT_DIR / "run_1_kwave_output.h5"))
    print("Checkpoint output directory after run_1:")
    print("\t-", "\n\t- ".join([f.name for f in EXAMPLE_OUTPUT_DIR.glob("*.h5")]))

    # create the simulation parameters for the 2nd run
    kgrid, medium, source, sensor, simulation_options, execution_options = make_simulation_parameters(
        directory=EXAMPLE_OUTPUT_DIR,
        checkpoint_timesteps=checkpoint_timesteps,
    )
    kspaceFirstOrder3D(kgrid, source, sensor, medium, simulation_options, execution_options)
    copy2(simulation_options.input_filename, _next_available_path(EXAMPLE_OUTPUT_DIR / "run_2_kwave_input.h5"))
    copy2(simulation_options.output_filename, _next_available_path(EXAMPLE_OUTPUT_DIR / "run_2_kwave_output.h5"))
    print("Checkpoint output directory after run_2:")
    print("\t-", "\n\t- ".join([f.name for f in EXAMPLE_OUTPUT_DIR.glob("*.h5")]))


if __name__ == "__main__":
    main()
