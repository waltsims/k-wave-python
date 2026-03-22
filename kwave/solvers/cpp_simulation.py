"""
Clean C++ backend for kspaceFirstOrder().

Handles HDF5 serialization and C++ binary execution without
depending on kWaveSimulation or the legacy options classes.
"""
import os
import stat
import subprocess
import tempfile

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.utils.io import write_attributes, write_matrix

# Source injection mode mapping (shared by pressure and velocity sources)
_SOURCE_MODE_MAP = {"dirichlet": 0, "additive-no-correction": 1, "additive": 2}


class CppSimulation:
    """Serialize simulation to HDF5 and run the C++ k-Wave binary."""

    def __init__(self, kgrid, medium, source, sensor, *, pml_size, pml_alpha, use_sg=True):
        self.kgrid = kgrid
        self.medium = medium
        self.source = source
        self.sensor = sensor
        self.pml_size = pml_size  # per-dimension tuple
        self.pml_alpha = pml_alpha  # per-dimension tuple
        self.use_sg = use_sg
        self.ndim = kgrid.dim

    def prepare(self, data_path=None):
        """Write HDF5 input file. Returns (input_file, output_file)."""
        if data_path is None:
            data_path = tempfile.mkdtemp(prefix="kwave_")
        os.makedirs(data_path, exist_ok=True)

        input_file = os.path.join(data_path, "kwave_input.h5")
        output_file = os.path.join(data_path, "kwave_output.h5")

        self._write_hdf5(input_file)
        return input_file, output_file

    def run(self, *, use_gpu=False, num_threads=None, device_num=None, quiet=False, debug=False):
        """Prepare + execute C++ binary + parse results."""
        input_file, output_file = self.prepare()
        self._execute(input_file, output_file, use_gpu=use_gpu, num_threads=num_threads, device_num=device_num, quiet=quiet, debug=debug)
        return self._parse_output(output_file)

    # -- HDF5 serialization --

    def _write_hdf5(self, filepath):
        """Write all variables to HDF5 input file.

        TODO: write_matrix() opens/closes the HDF5 file on each call (~30
        calls per serialization). Batching writes into a single open/close
        would improve performance, but requires changing the shared utility.
        """
        kgrid = self.kgrid
        medium = self.medium
        source = self.source
        sensor = self.sensor
        ndim = self.ndim

        # Grid dimensions
        Nx, Ny, Nz = int(kgrid.N[0]), 1, 1
        if ndim >= 2:
            Ny = int(kgrid.N[1])
        if ndim >= 3:
            Nz = int(kgrid.N[2])

        dx = float(kgrid.spacing[0])
        dy = float(kgrid.spacing[1]) if ndim >= 2 else dx
        dz = float(kgrid.spacing[2]) if ndim >= 3 else dx
        dt = float(kgrid.dt)
        Nt = int(kgrid.Nt)

        pml_x_size = int(self.pml_size[0])
        pml_y_size = int(self.pml_size[1]) if ndim >= 2 else 0
        pml_z_size = int(self.pml_size[2]) if ndim >= 3 else 0
        pml_x_alpha = float(self.pml_alpha[0])
        pml_y_alpha = float(self.pml_alpha[1]) if ndim >= 2 else 0.0
        pml_z_alpha = float(self.pml_alpha[2]) if ndim >= 3 else 0.0

        # Sound speed and density
        c0 = np.atleast_1d(np.asarray(medium.sound_speed, dtype=np.float32))
        rho0 = np.atleast_1d(np.asarray(medium.density if medium.density is not None else 1000.0, dtype=np.float32))
        c_ref = float(np.max(c0))

        # Staggered grid density
        if rho0.size == 1 or not self.use_sg:
            rho0_sgx = rho0_sgy = rho0_sgz = rho0
        else:
            rho0_sgx, rho0_sgy, rho0_sgz = self._compute_staggered_density(rho0)

        # Source flags
        has_p0 = source.p0 is not None and np.any(source.p0 != 0)
        has_p = source.p is not None and np.any(np.asarray(source.p) != 0)
        has_ux = source.ux is not None
        has_uy = source.uy is not None if ndim >= 2 else False
        has_uz = source.uz is not None if ndim >= 3 else False

        # Sensor mask index (1-based, Fortran order)
        if ndim == 2:
            grid_shape = (Nx, Ny)
        elif ndim == 3:
            grid_shape = (Nx, Ny, Nz)
        else:
            grid_shape = (Nx,)

        sensor_mask = np.asarray(sensor.mask, dtype=bool).reshape(grid_shape)
        sensor_mask_index = np.where(sensor_mask.flatten(order="F") != 0)[0] + 1
        sensor_mask_index = sensor_mask_index.astype(np.uint64)

        # -- Write integer variables --
        ints = {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz if ndim >= 3 else 1,
            "Nt": Nt,
            "pml_x_size": pml_x_size,
            "pml_y_size": pml_y_size,
            "pml_z_size": pml_z_size if ndim >= 3 else 0,
            "p_source_flag": int(has_p),
            "p0_source_flag": int(has_p0),
            "ux_source_flag": int(has_ux),
            "uy_source_flag": int(has_uy),
            "uz_source_flag": int(has_uz),
            "sxx_source_flag": 0,
            "syy_source_flag": 0,
            "szz_source_flag": 0,
            "sxy_source_flag": 0,
            "sxz_source_flag": 0,
            "syz_source_flag": 0,
            "transducer_source_flag": 0,
            "nonuniform_grid_flag": 0,
            "nonlinear_flag": int(medium.BonA is not None and np.any(np.asarray(medium.BonA) != 0)),
            "absorbing_flag": self._get_absorbing_flag(),
            "elastic_flag": 0,
            "axisymmetric_flag": 0,
            "sensor_mask_type": 0,
        }

        for name, val in ints.items():
            write_matrix(filepath, np.array(val, dtype=np.uint64), name)

        # Sensor mask index
        write_matrix(filepath, sensor_mask_index.reshape(-1, 1).astype(np.uint64), "sensor_mask_index")

        # -- Write float variables --
        floats = {"dx": dx, "dt": dt, "c0": c0, "c_ref": c_ref, "rho0": rho0}
        if ndim >= 2:
            floats["dy"] = dy
            floats["pml_y_alpha"] = pml_y_alpha
        if ndim >= 3:
            floats["dz"] = dz
            floats["pml_z_alpha"] = pml_z_alpha
        floats["pml_x_alpha"] = pml_x_alpha
        floats["rho0_sgx"] = rho0_sgx
        floats["rho0_sgy"] = rho0_sgy if ndim >= 2 else rho0
        if ndim >= 3:
            floats["rho0_sgz"] = rho0_sgz

        # Medium properties
        if medium.BonA is not None and np.any(np.asarray(medium.BonA) != 0):
            floats["BonA"] = np.asarray(medium.BonA, dtype=np.float32)
        if medium.alpha_coeff is not None:
            floats["alpha_coeff"] = np.asarray(medium.alpha_coeff, dtype=np.float32)
        if medium.alpha_power is not None:
            floats["alpha_power"] = np.float32(medium.alpha_power)

        # Write real-valued floats
        for name, val in floats.items():
            write_matrix(filepath, np.atleast_1d(np.asarray(val, dtype=np.float32)), name)

        # Source data
        if has_p0:
            write_matrix(filepath, np.asarray(source.p0, dtype=np.float32), "p0_source_input")
        if has_p:
            p_data = np.asarray(source.p, dtype=np.float32)
            write_matrix(filepath, p_data, "p_source_input")
            p_mask_idx = np.where(np.asarray(source.p_mask).flatten(order="F") != 0)[0] + 1
            write_matrix(filepath, p_mask_idx.astype(np.uint64).reshape(-1, 1), "p_source_index")
            mode_map = _SOURCE_MODE_MAP
            p_mode = getattr(source, "p_mode", "additive")
            write_matrix(filepath, np.array(mode_map.get(p_mode, 2), dtype=np.uint64), "p_source_mode")
            write_matrix(filepath, np.array(int(p_data.ndim > 1 and p_data.shape[0] > 1), dtype=np.uint64), "p_source_many")

        for vel, flag in [("ux", has_ux), ("uy", has_uy), ("uz", has_uz)]:
            if flag:
                vel_data = np.asarray(getattr(source, vel), dtype=np.float32)
                write_matrix(filepath, vel_data, f"{vel}_source_input")

        if has_ux or has_uy or has_uz:
            u_mask_idx = np.where(np.asarray(source.u_mask).flatten(order="F") != 0)[0] + 1
            write_matrix(filepath, u_mask_idx.astype(np.uint64).reshape(-1, 1), "u_source_index")
            mode_map = _SOURCE_MODE_MAP
            u_mode = getattr(source, "u_mode", "additive")
            write_matrix(filepath, np.array(mode_map.get(u_mode, 2), dtype=np.uint64), "u_source_mode")
            # Check if multiple waveforms
            first_vel = next(getattr(source, v) for v in ["ux", "uy", "uz"] if getattr(source, v, None) is not None)
            write_matrix(filepath, np.array(int(np.ndim(first_vel) > 1), dtype=np.uint64), "u_source_many")

        # File attributes
        write_attributes(filepath)

    def _compute_staggered_density(self, rho0):
        """Compute staggered grid density by averaging neighbors.

        TODO: This uses np.roll (spatial averaging) which matches the legacy
        save_to_disk_func behavior. The C++ binary computes its own spectral
        interpolation internally, so this is only used as the initial estimate.
        For full physical accuracy with heterogeneous density, consider using
        spectral interpolation (FFT-based shift) instead.
        """
        ndim = self.ndim
        rho0_sg = []
        for axis in range(ndim):
            shifted = np.roll(rho0, -1, axis=axis)
            rho0_sg.append(0.5 * (rho0 + shifted))
        # Fill remaining dims
        while len(rho0_sg) < 3:
            rho0_sg.append(rho0)
        return rho0_sg[0], rho0_sg[1], rho0_sg[2]

    def _get_absorbing_flag(self):
        """Determine absorption type: 0=lossless, 1=power-law, 2=stokes."""
        medium = self.medium
        if medium.alpha_coeff is None or np.all(np.asarray(medium.alpha_coeff) == 0):
            return 0
        if medium.alpha_power is not None and abs(float(np.asarray(medium.alpha_power).flat[0]) - 2.0) < 1e-10:
            return 2  # Stokes
        return 1  # Power-law

    def _execute(self, input_file, output_file, *, use_gpu, num_threads, device_num, quiet, debug):
        """Run the C++ k-Wave binary."""
        import kwave

        binary_name = "kspaceFirstOrder-CUDA" if use_gpu else "kspaceFirstOrder-OMP"
        binary_path = kwave.BINARY_PATH / binary_name
        if not binary_path.exists():
            if kwave.PLATFORM == "darwin" and use_gpu:
                raise ValueError(
                    "GPU simulations are currently not supported on MacOS. " "Try running the simulation on CPU by setting use_gpu=False."
                )
            raise FileNotFoundError(f"C++ binary not found at {binary_path}. Install with: pip install k-wave-data")
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)

        # Build command-line options
        options = ["-i", input_file, "-o", output_file]
        if num_threads is not None:
            options.extend(["-t", str(num_threads)])
        if device_num is not None:
            options.extend(["-g", str(device_num)])

        # Verbosity: quiet=0, default=1, debug=2
        if quiet:
            options.extend(["--verbose", "0"])
        elif debug:
            options.extend(["--verbose", "2"])

        command = [str(binary_path)] + options
        # capture_output: suppress in quiet mode, show in default/debug
        subprocess.run(command, capture_output=quiet, text=True, check=True)

    @staticmethod
    def _parse_output(output_file):
        """Parse HDF5 output file into result dict."""
        from kwave.executor import Executor

        return Executor.parse_executable_output(output_file)
