import os
import warnings
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

from kwave import BINARY_DIR, PLATFORM
from kwave.ksensor import kSensor

logger = getLogger(__name__)


class SimulationExecutionOptions:
    """
    A class to manage and configure the execution options for k-Wave simulations.
    """

    def __init__(
        self,
        is_gpu_simulation: bool = False,
        binary_path: Optional[str] = None,
        binary_dir: Optional[str] = None,
        binary_name: Optional[str] = None,
        kwave_function_name: Optional[str] = "kspaceFirstOrder3D",
        delete_data: bool = True,
        device_num: Optional[int] = None,
        num_threads: Optional[int] = None,
        thread_binding: Optional[bool] = None,
        system_call: Optional[str] = None,
        verbose_level: int = 0,
        auto_chunking: Optional[bool] = True,
        show_sim_log: bool = True,
        checkpoint_interval: Optional[int] = None,  # [seconds]
        checkpoint_timesteps: Optional[int] = None,  # [timestep integer]
        checkpoint_file: Optional[Path | str] = None,  # [path to hdf5 file]
    ):
        self.is_gpu_simulation = is_gpu_simulation
        self._binary_path = binary_path
        self._binary_name = binary_name
        self._binary_dir = binary_dir
        self.kwave_function_name = kwave_function_name
        self.delete_data = delete_data
        self.device_num = device_num
        self.num_threads = num_threads
        self.thread_binding = thread_binding
        self.system_call = system_call
        self.verbose_level = verbose_level
        self.auto_chunking = auto_chunking
        self.show_sim_log = show_sim_log
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_timesteps = checkpoint_timesteps
        self.checkpoint_file = checkpoint_file

        if self.checkpoint_file is not None:
            if self.checkpoint_interval is None and self.checkpoint_timesteps is None:
                raise ValueError("One of checkpoint_interval or checkpoint_timesteps must be set when checkpoint_file is set.")

    @property
    def num_threads(self) -> Union[int, str]:
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: Union[int, str]):
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("Unable to determine the number of CPUs on this system. Please specify the number of threads explicitly.")

        if value == "all":
            warnings.warn(
                "The 'all' option is deprecated. The value of None sets the maximal number of threads (excluding Windows).",
                DeprecationWarning,
            )
            value = cpu_count

        if value is None:
            value = cpu_count

        if not isinstance(value, int):
            raise ValueError("Got {value}. Number of threads must be 'all' or a positive integer")

        if value <= 0 or value > cpu_count:
            raise ValueError(f"Number of threads {value} must be a positive integer and less than total threads on the system {cpu_count}.")

        self._num_threads = value

    @property
    def verbose_level(self) -> int:
        return self._verbose_level

    @verbose_level.setter
    def verbose_level(self, value: int):
        if not (isinstance(value, int) and 0 <= value <= 2):
            raise ValueError("Verbose level must be between 0 and 2")
        self._verbose_level = value

    @property
    def is_gpu_simulation(self) -> Optional[bool]:
        return self._is_gpu_simulation

    @is_gpu_simulation.setter
    def is_gpu_simulation(self, value: Optional[bool]):
        "Set the flag to enable default GPU simulation. This option will supersede custom binary paths."
        self._is_gpu_simulation = value
        # Automatically update the binary name based on the GPU simulation flag
        if value is not None:
            self._binary_name = None

    @property
    def binary_name(self) -> str:
        valid_binary_names = ["kspaceFirstOrder-OMP", "kspaceFirstOrder-CUDA"]
        if self._binary_name is None:
            # set default binary name based on GPU simulation value
            if self.is_gpu_simulation is None:
                raise ValueError("`is_gpu_simulation` must be set to either True or False before determining the binary name.")

            if self.is_gpu_simulation:
                self._binary_name = "kspaceFirstOrder-CUDA"
            else:
                self._binary_name = "kspaceFirstOrder-OMP"

            if PLATFORM == "windows":
                self._binary_name += ".exe"
                valid_binary_names = [name + ".exe" for name in valid_binary_names]

        elif self._binary_name not in valid_binary_names:
            warnings.warn("Custom binary name set. Ignoring `is_gpu_simulation` state.")
        return self._binary_name

    @binary_name.setter
    def binary_name(self, value: str):
        self._binary_name = value

    @property
    def binary_path(self) -> Path:
        if self._binary_path is not None:
            return self._binary_path

        binary_dir = BINARY_DIR if self._binary_dir is None else self._binary_dir

        if binary_dir is None:
            raise ValueError("Binary directory is not specified.")

        path = Path(binary_dir) / self.binary_name
        if PLATFORM == "windows" and not path.name.endswith(".exe"):
            path = path.with_suffix(".exe")
        return path

    @binary_path.setter
    def binary_path(self, value: str):
        # check if the binary path is a valid path
        if not os.path.exists(value):
            raise FileNotFoundError(
                f"Binary path {value} does not exist. If you are trying to set `binary_dir`, use the `binary_dir` attribute instead."
            )
        self._binary_path = value

    @property
    def binary_dir(self) -> str:
        return BINARY_DIR if self._binary_dir is None else self._binary_dir

    @binary_dir.setter
    def binary_dir(self, value: str):
        # check if binary_dir is a directory
        if not os.path.isdir(value):
            raise NotADirectoryError(
                f"{value} is not a directory. If you are trying to set the `binary_path`, use the `binary_path` attribute instead."
            )
        self._binary_dir = Path(value)

    @property
    def device_num(self) -> Optional[int]:
        return self._device_num

    @device_num.setter
    def device_num(self, value: Optional[int]):
        if value is not None and value < 0:
            raise ValueError("Device number must be non-negative")
        self._device_num = value

    @property
    def checkpoint_interval(self) -> Optional[int]:
        return self._checkpoint_interval

    @checkpoint_interval.setter
    def checkpoint_interval(self, value: Optional[int]):
        if value is not None:
            if not isinstance(value, int) or value < 0:
                raise ValueError("Checkpoint interval must be a positive integer")
        self._checkpoint_interval = value

    @property
    def checkpoint_timesteps(self) -> Optional[int]:
        return self._checkpoint_timesteps

    @checkpoint_timesteps.setter
    def checkpoint_timesteps(self, value: Optional[int]):
        if value is not None:
            if not isinstance(value, int) or value < 0:
                raise ValueError("Checkpoint timesteps must be a positive integer")
        self._checkpoint_timesteps = value

    @property
    def checkpoint_file(self) -> Optional[Path]:
        if self._checkpoint_file is None:
            return None
        return self._checkpoint_file

    @checkpoint_file.setter
    def checkpoint_file(self, value: Optional[Path | str]):
        if value is not None:
            if not isinstance(value, (str, Path)):
                raise ValueError("Checkpoint file must be a string or Path object.")
            if isinstance(value, str):
                value = Path(value)
            if not value.parent.is_dir():
                raise FileNotFoundError(f"Checkpoint folder {value.parent} does not exist.")
            if value.suffix != ".h5":
                raise ValueError(f"Checkpoint file {value} must have .h5 extension.")
        self._checkpoint_file = value

    def as_list(self, sensor: kSensor) -> list[str]:
        options_list = []

        if self.device_num is not None:
            options_list.append("-g")
            options_list.append(str(self.device_num))

        if self._num_threads is not None and PLATFORM != "windows":
            options_list.append("-t")
            options_list.append(str(self._num_threads))

        if self.verbose_level > 0:
            options_list.append("--verbose")
            options_list.append(str(self.verbose_level))

        if (self.checkpoint_interval is not None or self.checkpoint_timesteps is not None) and self.checkpoint_file is not None:
            if self.checkpoint_timesteps is not None:
                options_list.append("--checkpoint_timesteps")
                options_list.append(str(self.checkpoint_timesteps))
            if self.checkpoint_interval is not None:
                options_list.append("--checkpoint_interval")
                options_list.append(str(self.checkpoint_interval))

            options_list.append("--checkpoint_file")
            options_list.append(str(self.checkpoint_file))

        record_options_map = {
            "p": "p_raw",
            "p_max": "p_max",
            "p_min": "p_min",
            "p_rms": "p_rms",
            "p_max_all": "p_max_all",
            "p_min_all": "p_min_all",
            "p_final": "p_final",
            "u": "u_raw",
            "u_max": "u_max",
            "u_min": "u_min",
            "u_rms": "u_rms",
            "u_max_all": "u_max_all",
            "u_min_all": "u_min_all",
            "u_final": "u_final",
        }

        if sensor.record is not None:
            matching_keys = sorted(set(sensor.record).intersection(record_options_map.keys()))
            options_list.extend([f"--{record_options_map[key]}" for key in matching_keys])

            if "u_non_staggered" in sensor.record or "I_avg" in sensor.record or "I" in sensor.record:
                options_list.append("--u_non_staggered_raw")

            if ("I_avg" in sensor.record or "I" in sensor.record) and ("p" not in sensor.record):
                options_list.append("--p_raw")
        else:
            options_list.append("--p_raw")

        if sensor.record_start_index is not None:
            options_list.append("-s")
            options_list.append(f"{sensor.record_start_index}")

        return options_list

    def get_options_string(self, sensor: kSensor) -> str:
        # raise a deprecation warning
        warnings.warn("This method is deprecated. Use `as_list` method instead.", DeprecationWarning)
        options_list = self.as_list(sensor)

        return " ".join(options_list)

    @property
    def env_vars(self) -> dict:
        env = os.environ

        if PLATFORM != "darwin":
            env.update({"OMP_PLACES": "cores"})

        if self.thread_binding is not None:
            if PLATFORM == "darwin":
                raise ValueError("Thread binding is not supported in MacOS.")
            # read the parameters and update the system options
            if self.thread_binding:
                env.update({"OMP_PROC_BIND": "SPREAD"})
            else:
                env.update({"OMP_PROC_BIND": "CLOSE"})
        else:
            if PLATFORM != "darwin":
                env.update({"OMP_PROC_BIND": "SPREAD"})

        return env
