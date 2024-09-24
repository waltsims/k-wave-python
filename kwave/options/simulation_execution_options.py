from kwave.ksensor import kSensor
from kwave.utils.checks import is_unix
from kwave import PLATFORM, BINARY_PATH
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings("always")
warnings.simplefilter("always", DeprecationWarning)


logger = logging.getLogger(__name__)

DEPRECATION_VERSION = "0.3.7"


@dataclass
class SimulationExecutionOptions:
    _is_gpu_simulation: bool = field(default=False, init=False)
    _binary_name: Optional[str] = field(default=None, init=False)
    _binary_path: Optional[Path] = field(default=BINARY_PATH, init=False)

    is_gpu_simulation: bool = False
    kwave_function_name: Optional[str] = "kspaceFirstOrder3D"
    delete_data: bool = True
    device_num: Optional[int] = None
    num_threads: Union[int, str] = "all"
    thread_binding: Optional[bool] = None
    system_call: Optional[str] = None
    verbose_level: int = 0
    auto_chunking: Optional[bool] = True
    show_sim_log: bool = True

    def __post_init__(self):
        if self.is_gpu_simulation:
            warnings.warn(
                f"Constructor argument `is_gpu_simulation` is deprecated and will be removed in version {DEPRECATION_VERSION}. Use configure() method instead.",
                DeprecationWarning,
            )
        if self.binary_name:
            warnings.warn(
                f"Constructor argument `binary_name` is deprecated and will be removed in version {DEPRECATION_VERSION}. Use configure() method instead.",
                DeprecationWarning,
            )
        self.configure(is_gpu_simulation=self.is_gpu_simulation, binary_path=self.binary_path)

    @property
    def is_gpu_simulation(self):
        return self._is_gpu_simulation

    @is_gpu_simulation.setter
    def is_gpu_simulation(self, value):
        warnings.warn(
            f"Setting is_gpu_simulation directly is deprecated and will be removed in version {DEPRECATION_VERSION}. Use configure() method instead.",
            DeprecationWarning,
        )
        self._is_gpu_simulation = value

    @property
    def binary_name(self):
        return self._binary_name

    @binary_name.setter
    def binary_name(self, value):
        warnings.warn(
            f"Setting binary_name directly is deprecated and will be removed in version {DEPRECATION_VERSION}. Use configure() method instead.",
            DeprecationWarning,
        )
        self._binary_name = value

    @property
    def binary_path(self):
        return self._binary_path

    @binary_path.setter
    def binary_path(self, value):
        warnings.warn(
            f"Setting binary_path directly is deprecated and will be removed in version {DEPRECATION_VERSION}. Use configure() method instead.",
            DeprecationWarning,
        )
        self._binary_path = Path(value)

    def configure(self, is_gpu_simulation: bool, binary_path: Optional[str] = None):
        self._is_gpu_simulation = is_gpu_simulation

        if is_gpu_simulation:
            self._binary_name = "kspaceFirstOrder-CUDA"
        else:
            self._binary_name = "kspaceFirstOrder-OMP"

        if PLATFORM == "windows" and not self._binary_name.endswith(".exe"):
            self._binary_name += ".exe"

        if binary_path:
            self._binary_path = Path(binary_path) / self._binary_name
            if not self._binary_path.exists():
                logger.warning(f"Specified binary path does not exist: {self._binary_path}")
        else:
            self._binary_path = Path(BINARY_PATH) / self._binary_name
            if not self._binary_path.exists():
                logger.error(f"Default binary not found at: {self._binary_path}")
                raise FileNotFoundError(f"Default binary not found at: {self._binary_path}")

        logger.info(f"Configured with binary path: {self._binary_path}")

    @classmethod
    def create_gpu_config(cls, binary_path: Optional[str] = None):
        instance = cls()
        instance.configure(is_gpu_simulation=True, binary_path=binary_path)
        return instance

    @classmethod
    def create_cpu_config(cls, binary_path: Optional[str] = None):
        instance = cls()
        instance.configure(is_gpu_simulation=False, binary_path=binary_path)
        return instance

    def get_options_string(self, sensor: kSensor) -> str:
        options_dict = {}
        if self.device_num:
            options_dict["-g"] = self.device_num

        if self.num_threads:
            if isinstance(self.num_threads, int):
                assert self.num_threads > 0 and self.num_threads != float("inf")
            else:
                assert self.num_threads == "all"
                self.num_threads = None
            options_dict["-t"] = self.num_threads

        if self.verbose_level:
            assert isinstance(self.verbose_level, int) and 0 <= self.verbose_level <= 2
            options_dict["--verbose"] = self.verbose_level

        options_string = ""
        for flag, value in options_dict.items():
            if value:
                options_string += f" {flag} {str(value)}"

        # check if sensor.record is given
        if sensor.record is not None:
            record = sensor.record

            # set the options string to record the required output fields
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
            # if sensor.record is not given, record the raw time series of p
            options_string = options_string + " --p_raw"

        # check if sensor.record_start_index is given
        if sensor.record_start_index is not None:
            options_string = options_string + " -s " + str(sensor.record_start_index)
        return options_string

    @property
    def system_string(self):
        # set OS string for setting environment variables
        if is_unix():
            env_set_str = ""
            sys_sep_str = " "
        else:
            env_set_str = "set "
            sys_sep_str = " & "

        # set system string to define domain for thread migration
        system_string = env_set_str + "OMP_PLACES=cores" + sys_sep_str

        if self.thread_binding is not None:
            # read the parameters and update the system options
            if self.thread_binding:
                system_string = system_string + " " + env_set_str + "OMP_PROC_BIND=SPREAD" + sys_sep_str
            else:
                system_string = system_string + " " + env_set_str + "OMP_PROC_BIND=CLOSE" + sys_sep_str
        else:
            # set to round-robin over places
            system_string = system_string + " " + env_set_str + "OMP_PROC_BIND=SPREAD" + sys_sep_str

        if self.system_call:
            system_string = system_string + " " + self.system_call + sys_sep_str

        return system_string


# Example usage
if __name__ == "__main__":
    # Create a GPU configuration
    gpu_options = SimulationExecutionOptions.create_gpu_config("/custom/gpu/path")

    # Create a CPU configuration
    cpu_options = SimulationExecutionOptions.create_cpu_config()

    # Manual configuration
    manual_options = SimulationExecutionOptions()
    manual_options.configure(is_gpu_simulation=True, binary_path="/another/custom/path")
