from dataclasses import dataclass
import os
from typing import Optional, Union

from kwave import PLATFORM, BINARY_PATH
from kwave.ksensor import kSensor


@dataclass
class SimulationExecutionOptions:
    # Are we going to run the simulation on the GPU?
    # Affects binary name and the way the simulation is run
    is_gpu_simulation: bool = False

    # user defined location for the binary
    binary_path: Optional[str] = BINARY_PATH
    # user defined location for the binary
    binary_name: Optional[str] = None
    # user defined number of threads
    kwave_function_name: Optional[str] = "kspaceFirstOrder3D"
    # Whether to delete the input and output files after the simulation
    delete_data: bool = True
    # GPU device flag
    device_num: Optional[int] = None
    # number of threads
    num_threads: Union[int, str] = "all"

    # user defined thread binding option
    thread_binding: Optional[bool] = None

    # user input for system string
    system_call: Optional[str] = None
    verbose_level: int = 0

    # determine whether chunking is handled automatically (the default), or manually
    auto_chunking: Optional[bool] = True

    # show simulation log
    show_sim_log: bool = True

    def __post_init__(self):
        self.validate()

        if self.binary_name is None:
            if self.is_gpu_simulation:
                self.binary_name = "kspaceFirstOrder-CUDA"
            else:
                self.binary_name = "kspaceFirstOrder-OMP"

        if PLATFORM == "windows":
            if not self.binary_name.endswith(".exe"):
                self.binary_name += ".exe"

        self.binary_path = self.binary_path / self.binary_name

    def validate(self):
        if isinstance(self.num_threads, int):
            assert self.num_threads > 0 and self.num_threads != float("inf")
        else:
            assert self.num_threads == "all"
            self.num_threads = None

        assert isinstance(self.verbose_level, int) and 0 <= self.verbose_level <= 2

    def get_options_string(self, sensor: kSensor) -> str:
        options_dict = {}
        if self.device_num:
            options_dict["-g"] = self.device_num

        if self.num_threads:
            options_dict["-t"] = self.num_threads

        if self.verbose_level:
            options_dict["--verbose"] = self.verbose_level

        options_string = ""
        for flag, value in options_dict.items():
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

        # check if sensor.record_start_imdex is given
        if sensor.record_start_index is not None:
            options_string = options_string + " -s " + str(sensor.record_start_index)
        return options_string

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
