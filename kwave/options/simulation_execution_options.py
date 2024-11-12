from pathlib import Path
from typing import Optional, Union
import os

from kwave import PLATFORM, BINARY_DIR
from kwave.ksensor import kSensor


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
        num_threads: Union[int, str] = "all",
        thread_binding: Optional[bool] = None,
        system_call: Optional[str] = None,
        verbose_level: int = 0,
        auto_chunking: Optional[bool] = True,
        show_sim_log: bool = True,
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

    @property
    def num_threads(self) -> Union[int, str]:
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: Union[int, str]):
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("Unable to determine the number of CPUs on this system. Please specify the number of threads explicitly.")

        if value == "all":
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
        "Set the flag to enable default GPU simulation. This option will supercede custom binary paths."
        self._is_gpu_simulation = value
        # Automatically update the binary name based on the GPU simulation flag
        if value is not None:
            self._binary_name = None

    @property
    def binary_name(self) -> str:
        valid_binary_names = ["kspaceFirstOrder-CUDA", "kspaceFirstOrder-OMP"]
        if PLATFORM == "windows":
            valid_binary_names = [name + ".exe" for name in valid_binary_names]

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
        elif self._binary_name not in valid_binary_names:
            import warnings

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

    def get_options_string(self, sensor: kSensor) -> str:
        options_list = []
        if self.device_num is not None and self.device_num < 0:
            raise ValueError("Device number must be non-negative")
        if self.device_num is not None:
            options_list.append(f" -g {self.device_num}")

        if self.num_threads is not None:
            options_list.append(f" -t {self.num_threads}")

        if self.verbose_level > 0:
            options_list.append(f" --verbose {self.verbose_level}")

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
            matching_keys = set(sensor.record).intersection(record_options_map.keys())
            for key in matching_keys:
                options_list.append(f" --{record_options_map[key]}")

            if "u_non_staggered" in sensor.record or "I_avg" in sensor.record or "I" in sensor.record:
                options_list.append(" --u_non_staggered_raw")

            if ("I_avg" in sensor.record or "I" in sensor.record) and ("p" not in sensor.record):
                options_list.append(" --p_raw")
        else:
            options_list.append(" --p_raw")

        if sensor.record_start_index is not None:
            options_list.append(f" -s {sensor.record_start_index}")

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
