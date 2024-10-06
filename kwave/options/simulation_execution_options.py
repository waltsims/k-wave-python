from typing import Optional, Union
import os

from kwave import PLATFORM, BINARY_PATH
from kwave.ksensor import kSensor
from kwave.utils.checks import is_unix


class SimulationExecutionOptions:
    """
    A class to manage and configure the execution options for k-Wave simulations.
    """

    def __init__(
        self,
        is_gpu_simulation: bool = False,
        binary_path: Optional[str] = BINARY_PATH,
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
        self.kwave_function_name = kwave_function_name
        self.delete_data = delete_data
        self.device_num = device_num
        self.num_threads = num_threads
        self.thread_binding = thread_binding
        self.system_call = system_call
        self.verbose_level = verbose_level
        self.auto_chunking = auto_chunking
        self.show_sim_log = show_sim_log
        self._refresh_binary_attributes()

    @property
    def num_threads(self) -> Union[int, str]:
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: Union[int, str]):
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 1
        if isinstance(value, int):
            if value <= 0 or value == float("inf"):
                raise ValueError("Number of threads must be a positive integer")
        elif value == "all":
            value = cpu_count
        else:
            raise ValueError("Number of threads must be 'all' or a positive integer")
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
    def is_gpu_simulation(self) -> bool:
        return self._is_gpu_simulation

    @is_gpu_simulation.setter
    def is_gpu_simulation(self, value: bool):
        self._is_gpu_simulation = value
        self._refresh_binary_attributes()

    @property
    def binary_name(self) -> str:
        if self._binary_name is None:
            if self.is_gpu_simulation:
                self._binary_name = "kspaceFirstOrder-CUDA"
            else:
                self._binary_name = "kspaceFirstOrder-OMP"
        return self._binary_name

    @binary_name.setter
    def binary_name(self, value: str):
        self._binary_name = value

    @property
    def binary_path(self) -> str:
        path = BINARY_PATH / self.binary_name
        if PLATFORM == "windows" and not path.name.endswith(".exe"):
            path = path.with_suffix(".exe")
        return path

    @binary_path.setter
    def binary_path(self, value: str):
        self._binary_path = value

    def get_options_string(self, sensor: kSensor) -> str:
        options_list = []
        if self.device_num is not None and self.device_num < 0:
            raise ValueError("Device number must be non-negative")
        if self.device_num is not None:
            options_list.append(f" -g {self.device_num}")

        if self.num_threads is not None:
            options_list.append(f" -t {self.num_threads}")

        if self.verbose_level is not None:
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

    def _construct_system_string(self, env_set_str: str, sys_sep_str: str) -> str:
        omp_proc_bind = "SPREAD" if self.thread_binding else "CLOSE"
        system_string = f"{env_set_str}OMP_PLACES=cores{sys_sep_str} {env_set_str}OMP_PROC_BIND={omp_proc_bind}{sys_sep_str}"

        if self.system_call:
            system_string += f" {self.system_call}" + sys_sep_str

        return system_string

    @property
    def system_string(self):
        env_set_str = "" if is_unix() else "set "
        sys_sep_str = " " if is_unix() else " & "
        return self._construct_system_string(env_set_str, sys_sep_str)
