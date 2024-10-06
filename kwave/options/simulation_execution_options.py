from kwave.ksensor import kSensor
from kwave import BINARY_PATH, SIMULATION_OPTIONS_DEPRICATION_VERSION
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
import warnings
import os

# Set up warnings to always be shown.
warnings.filterwarnings("always")
warnings.simplefilter("always", DeprecationWarning)

# Set up logger
logger = logging.getLogger(__name__)


# Unified deprecation warning function
def warn_deprecation(feature_name: str, alternative: str):
    message = (
        f"The `{feature_name}` argument is deprecated and will be removed in version {SIMULATION_OPTIONS_DEPRICATION_VERSION}. "
        f"Please use `{alternative}` instead."
    )
    warnings.warn(message, DeprecationWarning)
    logger.warning(message)


@dataclass
class SimulationExecutionOptions:
    _gpu_simulation_enabled: bool = field(default=False, init=False)
    _binary_name: str = field(init=False)
    _binary_dir: Optional[Path] = field(default=Path(BINARY_PATH), init=False)

    gpu_simulation_enabled: bool = False
    kwave_function_name: Optional[str] = "kspaceFirstOrder3D"
    delete_data: bool = True
    device_num: Optional[int] = None
    num_threads: Union[int, str] = "all"
    thread_binding: Optional[bool] = None
    system_call: Optional[str] = None
    verbose_level: int = 0
    auto_chunking: Optional[bool] = True
    show_sim_log: bool = True

    def __init__(self, is_gpu_simulation: Optional[bool] = None, **kwargs):
        if is_gpu_simulation is not None:
            warn_deprecation("is_gpu_simulation", "gpu_simulation_enabled")
            self._gpu_simulation_enabled = is_gpu_simulation

        self.kwave_function_name = kwargs.get("kwave_function_name", "kspaceFirstOrder3D")
        self.delete_data = kwargs.get("delete_data", True)
        self._device_num = kwargs.get("device_num", None)
        self._num_threads = kwargs.get("num_threads", "all")
        self.thread_binding = kwargs.get("thread_binding", None)
        self.system_call = kwargs.get("system_call", None)
        self.verbose_level = kwargs.get("verbose_level", 0)
        self.auto_chunking = kwargs.get("auto_chunking", True)
        self.show_sim_log = kwargs.get("show_sim_log", True)
        self._set_binary_name()

    def _set_binary_name(self):
        self._binary_name = "kspaceFirstOrder-CUDA" if self._gpu_simulation_enabled else "kspaceFirstOrder-OMP"

    @property
    def gpu_simulation_enabled(self):
        return self._gpu_simulation_enabled

    @gpu_simulation_enabled.setter
    def gpu_simulation_enabled(self, value: bool):
        self._gpu_simulation_enabled = value
        self._set_binary_name()
        logger.info(f"GPU simulation set to {'enabled' if value else 'disabled'}. Binary name updated to: {self._binary_name}")

    @property
    def binary_path(self):
        return self._binary_dir / self._binary_name

    @binary_path.setter
    def binary_path(self, path: str):
        self._binary_dir = Path(path)
        self._binary_path = str(self._binary_dir / self._binary_name)
        if not Path(self._binary_path).exists():
            logger.warning(f"Specified binary path does not exist: {self._binary_path}")
        logger.info(f"Binary directory updated to: {self._binary_dir}")

    @property
    def num_threads(self):
        if self._num_threads == "all":
            return os.cpu_count()
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value: Union[int, str]):
        if isinstance(value, int) and value <= 0:
            raise ValueError("Number of threads must be a positive integer or 'all'.")
        self._num_threads = value
        logger.info(f"Number of threads set to: {value}")

    @property
    def device_num(self):
        return self._device_num

    @device_num.setter
    def device_num(self, value: Optional[int]):
        self._device_num = value
        if value is not None:
            logger.info(f"Device number set to: {value}")
        else:
            logger.info("Device number not specified.")

    def get_options_string(self, sensor: "kSensor") -> str:
        options_dict = {}
        if self.device_num:
            options_dict["-g"] = self.device_num

        if self.num_threads:
            if isinstance(self.num_threads, int):
                assert self.num_threads > 0 and self.num_threads != float("inf")
            options_dict["-t"] = self.num_threads

        if self.verbose_level:
            assert isinstance(self.verbose_level, int) and 0 <= self.verbose_level <= 2
            options_dict["--verbose"] = self.verbose_level

        options_string = ""
        for flag, value in options_dict.items():
            if value:
                options_string += f" {flag} {str(value)}"

        if sensor.record is not None:
            record = sensor.record

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
            for k, v in record_options_map.items():
                if k in record:
                    options_string = options_string + f" --{v}"

            if "u_non_staggered" in record or "I_avg" in record or "I" in record:
                options_string = options_string + " --u_non_staggered_raw"

            if ("I_avg" in record or "I" in record) and ("p" not in record):
                options_string = options_string + " --p_raw"
        else:
            options_string = options_string + " --p_raw"

        if sensor.record_start_index is not None:
            options_string = options_string + " -s " + str(sensor.record_start_index)
        return options_string

    @property
    def system_string(self):
        if os.name == "posix":
            env_set_str = ""
            sys_sep_str = " "
        else:
            env_set_str = "set "
            sys_sep_str = " & "

        system_string = env_set_str + "OMP_PLACES=cores" + sys_sep_str

        if self.thread_binding is not None:
            if self.thread_binding:
                system_string += env_set_str + "OMP_PROC_BIND=SPREAD" + sys_sep_str
            else:
                system_string += env_set_str + "OMP_PROC_BIND=CLOSE" + sys_sep_str
        else:
            system_string += env_set_str + "OMP_PROC_BIND=SPREAD" + sys_sep_str

        if self.system_call:
            system_string += self.system_call + sys_sep_str

        return system_string


# Example usage
if __name__ == "__main__":
    options = SimulationExecutionOptions()
    options.gpu_simulation_enabled = True  # Using the setter to configure GPU usage.
    options.num_threads = 4  # Using the setter to configure number of threads.
    options.binary_path = "/custom/path"  # Using the setter to configure binary path.
