"""
kWaveDiffusion
==============

Python/NumPy port of the MATLAB kWaveDiffusion class from k-Wave.

Solves:

    A * dT/dt = div(K * grad(T)) - B * (T - Ta) + Q

or, when diffusion_coeff is supplied:

    dT/dt = div(D * grad(T)) - P * (T - Ta) + S

where

    D = K / A
    P = B / A
    S = Q / A

The implementation below currently supports periodic boundaries.

Supported dimensions:
    1D
    2D
    3D

Supported media:
    homogeneous
    heterogeneous

Supported properties:
    diffusion_coeff
    density
    specific_heat
    thermal_conductivity

    perfusion_coeff
    blood_density
    blood_specific_heat
    blood_perfusion_rate
    blood_ambient_temperature

Supported source:
    T0
    Q

References
----------
Treeby & Saratoon, k-Wave MATLAB kWaveDiffusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


ArrayLike = Union[np.ndarray, float, int]


@dataclass
class DiffusionMedium:
    """
    Medium parameters.

    Either provide:

        diffusion_coeff

    or:

        density
        specific_heat
        thermal_conductivity

    For perfusion, either provide:

        perfusion_coeff

    or:

        blood_density
        blood_specific_heat
        blood_perfusion_rate

    blood_ambient_temperature is required when perfusion is used.
    """

    # Diffusion representation
    diffusion_coeff: Optional[ArrayLike] = None

    density: Optional[ArrayLike] = None
    specific_heat: Optional[ArrayLike] = None
    thermal_conductivity: Optional[ArrayLike] = None

    # Perfusion representation
    perfusion_coeff: Optional[ArrayLike] = None

    blood_density: Optional[ArrayLike] = None
    blood_specific_heat: Optional[ArrayLike] = None
    blood_perfusion_rate: Optional[ArrayLike] = None

    blood_ambient_temperature: Optional[ArrayLike] = None

    # k-space reference coefficients
    diffusion_coeff_ref: Union[str, float] = "max"
    perfusion_coeff_ref: Union[str, float] = "max"

    # Boundary condition
    boundary_condition: str = "periodic"


@dataclass
class DiffusionSource:
    """
    Thermal source.

    T0:
        Initial temperature [degC]

    Q:
        Volumetric heat deposition [W/m^3]
    """

    T0: ArrayLike
    Q: ArrayLike = 0.0


class kWaveDiffusion:
    """
    Time-domain k-space solver for the diffusion / Pennes bioheat equation.

    This is a Python port of the MATLAB kWaveDiffusion class.

    Parameters
    ----------
    kgrid:
        kWaveGrid instance.

    medium:
        DiffusionMedium or a dictionary containing the medium parameters.

    source:
        DiffusionSource or dictionary containing T0 and optionally Q.

    sensor:
        Optional binary sensor mask.

    use_kspace:
        If True, use the k-space temporal correction.

    dtype:
        np.float64 or np.float32.

    display_updates:
        Print simulation information.

    device:
        "cpu" (NumPy) or "gpu"/"cuda" (CuPy CUDA backend).
        GPU execution requires a CUDA-compatible CuPy installation.

    Notes
    -----
    Current implementation supports periodic boundaries.

    Insulating/conducting DCT/DST boundaries are intentionally not
    implemented here because those require the k-Wave DTT operators.
    """

    def _is_uniform(self, value):

        value = self.xp.asarray(value)
    
        if value.size == 0:
            return False
    
        return bool(
            self.xp.all(value == value.flat[0])
        )

    def __init__(
        self,
        kgrid,
        medium,
        source,
        sensor=None,
        *,
        use_kspace: bool = True,
        dtype=np.float64,
        display_updates: bool = True,
        device: str = "cpu",
    ):

        self.kgrid = kgrid
        self.dtype = np.dtype(dtype)

        device = str(device).lower()
        if device in ("cuda", "gpu"):
            if cp is None:
                raise ImportError(
                    "CUDA execution requires CuPy. Install a CuPy build "
                    "compatible with your CUDA version."
                )
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device is available.")
            self.device = "gpu"
            self.xp = cp
        elif device == "cpu":
            self.device = "cpu"
            self.xp = np
        else:
            raise ValueError("device must be 'cpu', 'gpu', or 'cuda'.")

        self.use_kspace = use_kspace
        self.display_updates = display_updates

        # ------------------------------------------------------------
        # Determine grid dimensions
        # ------------------------------------------------------------

        self._read_grid(kgrid)

        # ------------------------------------------------------------
        # Read medium/source
        # ------------------------------------------------------------

        if isinstance(medium, dict):
            medium = DiffusionMedium(**medium)

        if isinstance(source, dict):
            source = DiffusionSource(**source)

        self.medium = medium
        self.source = source

        # ------------------------------------------------------------
        # Validate boundary condition
        # ------------------------------------------------------------

        boundary = medium.boundary_condition

        if boundary not in (
            "periodic",
            "insulating",
            "conducting",
        ):
            raise ValueError(
                f"Unknown boundary condition: {boundary}"
            )

        if boundary != "periodic":
            raise NotImplementedError(
                "This Python implementation currently supports "
                "periodic boundaries only. Insulating/conducting "
                "DCT/DST support can be added separately."
            )

        self.boundary_condition = boundary

        # ------------------------------------------------------------
        # Read material properties
        # ------------------------------------------------------------

        self._setup_diffusion(medium)

        self._setup_perfusion(medium)

        self._setup_source(source)

        # ------------------------------------------------------------
        # Determine whether medium is heterogeneous
        # ------------------------------------------------------------

        self.flag_homogeneous_diffusion = (
            self._is_uniform(self.diffusion_p1)
            and self._is_uniform(self.diffusion_p2)
        )
        
        self.flag_homogeneous_perfusion = (
            self._is_uniform(self.perfusion_coeff)
        )

        self.flag_homogeneous = (
            self.flag_homogeneous_diffusion
            and self.flag_homogeneous_perfusion
        )

        # ------------------------------------------------------------
        # Reference coefficients
        # ------------------------------------------------------------

        self.diffusion_coeff_ref = self._reference_value(
            medium.diffusion_coeff_ref,
            self.diffusion_p1 * self.diffusion_p2,
            "diffusion_coeff_ref",
        )

        if bool(self.xp.any(self.perfusion_coeff != 0)):
            self.perfusion_coeff_ref = self._reference_value(
                medium.perfusion_coeff_ref,
                self.perfusion_coeff,
                "perfusion_coeff_ref",
            )
        else:
            self.perfusion_coeff_ref = 0.0

        # ------------------------------------------------------------
        # Initial temperature
        # ------------------------------------------------------------

        T0 = self.xp.asarray(source.T0, dtype=self.dtype)

        if T0.ndim == 0:
            self.T = self.xp.full(
                self.shape,
                T0.item(),
                dtype=self.dtype,
            )
        else:
            if T0.shape != self.shape:
                raise ValueError(
                    f"T0 has shape {T0.shape}, expected {self.shape}"
                )

            self.T = T0.copy()

        # ------------------------------------------------------------
        # Heat source
        # ------------------------------------------------------------

        Q = self.xp.asarray(source.Q, dtype=self.dtype)

        if Q.ndim == 0:
            self.Q = Q.item()
        else:
            if Q.shape != self.shape:
                raise ValueError(
                    f"Q has shape {Q.shape}, expected {self.shape}"
                )

            self.Q = Q.copy()

        # ------------------------------------------------------------
        # Thermal damage
        # ------------------------------------------------------------

        self.cem43 = self.xp.zeros(
            self.shape,
            dtype=self.dtype,
        )

        # ------------------------------------------------------------
        # Time/sensor bookkeeping
        # ------------------------------------------------------------

        self.time_steps_taken = 0

        if sensor is not None:
            if isinstance(sensor, dict):
                sensor = sensor["mask"]

            self.sensor_mask = self.xp.asarray(sensor)

            if self.sensor_mask.shape != self.shape:
                raise ValueError(
                    f"Sensor mask has shape {self.sensor_mask.shape}, "
                    f"expected {self.shape}"
                )

            # MATLAB requires binary mask
            if not bool(self.xp.all(
                (self.sensor_mask == 0)
                | (self.sensor_mask == 1)
            )):
                raise ValueError(
                    "sensor.mask must contain only 0 and 1."
                )

            self.sensor_mask_index = self.xp.flatnonzero(
                self.sensor_mask.ravel(order="C")
            )

            self.num_sensor_points = len(
                self.sensor_mask_index
            )

            self.sensor_data = None

        else:
            self.sensor_mask = None
            self.sensor_mask_index = None
            self.num_sensor_points = 0
            self.sensor_data = None

        # ------------------------------------------------------------
        # Build Fourier operators
        # ------------------------------------------------------------

        self._build_wavenumbers()

    # ==================================================================
    # GRID
    # ==================================================================

    def _read_grid(self, kgrid):

        """
        Read grid dimensions and spacings.

        k-Wave-Python's kWaveGrid stores the grid dimensions in N
        and the spacings in spacing. This function also supports
        common legacy attribute layouts.
        """

        # New k-Wave-Python API
        if hasattr(kgrid, "N"):
            N = np.asarray(kgrid.N).astype(int).ravel()

            if hasattr(kgrid, "spacing"):
                spacing = np.asarray(
                    kgrid.spacing
                ).astype(float).ravel()
            else:
                spacing = None

        else:
            N = []

            for name in ("Nx", "Ny", "Nz"):
                if hasattr(kgrid, name):
                    N.append(int(getattr(kgrid, name)))

            N = np.asarray(N, dtype=int)

            spacing = []

            for name in ("dx", "dy", "dz"):
                if hasattr(kgrid, name):
                    spacing.append(float(getattr(kgrid, name)))

            spacing = np.asarray(
                spacing,
                dtype=float,
            )

        if N.size == 0:
            raise ValueError(
                "Could not determine grid dimensions from kgrid."
            )

        self.dim = int(N.size)

        self.Nx = int(N[0])

        self.Ny = int(N[1]) if self.dim >= 2 else 1
        self.Nz = int(N[2]) if self.dim >= 3 else 1

        self.shape = tuple(
            int(v) for v in N
        )

        if spacing is None or spacing.size < self.dim:
            spacing = np.ones(
                self.dim,
                dtype=float,
            )

            for i, name in enumerate(
                ("dx", "dy", "dz")
            ):
                if hasattr(kgrid, name):
                    spacing[i] = float(
                        getattr(kgrid, name)
                    )

        self.dx = float(spacing[0])

        self.dy = (
            float(spacing[1])
            if self.dim >= 2
            else self.dx
        )

        self.dz = (
            float(spacing[2])
            if self.dim >= 3
            else self.dx
        )

    # ==================================================================
    # MATERIAL
    # ==================================================================

    def _as_field(
        self,
        value,
        name,
        allow_none=False,
    ):

        if value is None:
            if allow_none:
                return self.xp.asarray(
                    0.0,
                    dtype=self.dtype,
                )

            raise ValueError(
                f"{name} must be specified."
            )

        arr = self.xp.asarray(
            value,
            dtype=self.dtype,
        )

        if arr.ndim == 0:
            return arr

        if arr.shape != self.shape:
            raise ValueError(
                f"{name} has shape {arr.shape}; "
                f"expected scalar or {self.shape}."
            )

        return arr

    def _setup_diffusion(self, medium):
        
        # ------------------------------------------------------------
        # Either D or rho/c/K
        # ------------------------------------------------------------

        if medium.diffusion_coeff is not None:

            self.diffusion_p1 = self.xp.asarray(
                1.0,
                dtype=self.dtype,
            )

            self.diffusion_p2 = self._as_field(
                medium.diffusion_coeff,
                "diffusion_coeff",
            )

            if bool(self.xp.any(self.diffusion_p2 < 0)):
                raise ValueError(
                    "diffusion_coeff must be non-negative."
                )

        else:

            if (
                medium.density is None
                or medium.specific_heat is None
                or medium.thermal_conductivity is None
            ):
                raise ValueError(
                    "Specify either diffusion_coeff or "
                    "density, specific_heat, and "
                    "thermal_conductivity."
                )

            density = self._as_field(
                medium.density,
                "density",
            )

            specific_heat = self._as_field(
                medium.specific_heat,
                "specific_heat",
            )

            conductivity = self._as_field(
                medium.thermal_conductivity,
                "thermal_conductivity",
            )

            if bool(self.xp.any(density <= 0)):
                raise ValueError("density must be positive.")
            
            if bool(self.xp.any(specific_heat <= 0)):
                raise ValueError("specific_heat must be positive.")
            
            if bool(self.xp.any(conductivity <= 0)):
                raise ValueError(
                    "thermal_conductivity must be positive."
                )

            self.diffusion_p1 = (
                1.0 / (density * specific_heat)
            )

            self.diffusion_p2 = conductivity

    # ==================================================================

    def _setup_perfusion(self, medium):

        # No perfusion parameters at all
        if (
            medium.perfusion_coeff is None
            and medium.blood_density is None
            and medium.blood_specific_heat is None
            and medium.blood_perfusion_rate is None
            and medium.blood_ambient_temperature is None
        ):
            self.perfusion_coeff = self.xp.asarray(
                0.0,
                dtype=self.dtype,
            )

            self.blood_ambient_temperature = self.xp.asarray(
                0.0,
                dtype=self.dtype,
            )

            return

        # Direct P
        if medium.perfusion_coeff is not None:

            if medium.blood_ambient_temperature is None:
                raise ValueError(
                    "blood_ambient_temperature is required "
                    "when perfusion_coeff is specified."
                )

            self.perfusion_coeff = self._as_field(
                medium.perfusion_coeff,
                "perfusion_coeff",
            )

            self.blood_ambient_temperature = self._as_field(
                medium.blood_ambient_temperature,
                "blood_ambient_temperature",
            )

            return

        # Full blood parameters
        required = (
            medium.blood_density,
            medium.blood_specific_heat,
            medium.blood_perfusion_rate,
            medium.blood_ambient_temperature,
            medium.density,
            medium.specific_heat,
        )

        if any(x is None for x in required):

            raise ValueError(
                "For blood-property perfusion specify:\n"
                "blood_density\n"
                "blood_specific_heat\n"
                "blood_perfusion_rate\n"
                "blood_ambient_temperature\n"
                "density\n"
                "specific_heat"
            )

        blood_density = self._as_field(
            medium.blood_density,
            "blood_density",
        )

        blood_specific_heat = self._as_field(
            medium.blood_specific_heat,
            "blood_specific_heat",
        )

        blood_perfusion_rate = self._as_field(
            medium.blood_perfusion_rate,
            "blood_perfusion_rate",
        )

        density = self._as_field(
            medium.density,
            "density",
        )

        specific_heat = self._as_field(
            medium.specific_heat,
            "specific_heat",
        )

        self.perfusion_coeff = (
            blood_density
            * blood_perfusion_rate
            * blood_specific_heat
            / (density * specific_heat)
        )

        self.blood_ambient_temperature = self._as_field(
            medium.blood_ambient_temperature,
            "blood_ambient_temperature",
        )

    # ==================================================================

    
    def _setup_source(self, source):
    
        if source.Q is None:
            source.Q = 0.0
    
        has_material_properties = (
            self.medium.density is not None
            and self.medium.specific_heat is not None
        )
    
        if has_material_properties:
    
            density = self._as_field(
                self.medium.density,
                "density",
            )
    
            specific_heat = self._as_field(
                self.medium.specific_heat,
                "specific_heat",
            )
    
            if bool(self.xp.any(density <= 0)):
                raise ValueError("density must be positive.")
    
            if bool(self.xp.any(specific_heat <= 0)):
                raise ValueError(
                    "specific_heat must be positive."
                )
    
            self.q_scale_factor = (
                1.0 / (density * specific_heat)
            )
    
        else:
    
            self.q_scale_factor = None
    
            if bool(self.xp.any(
                self.xp.asarray(source.Q) != 0
            )):
                raise ValueError(
                    "density and specific_heat must be "
                    "supplied when source.Q is nonzero."
                )

    # ==================================================================
    # REFERENCE COEFFICIENT
    # ==================================================================

    def _reference_value(
        self,
        ref,
        values,
        name,
    ):

        values = self.xp.asarray(values)

        if isinstance(ref, str):

            ref_lower = ref.lower()

            if ref_lower == "min":
                return float(float(self.xp.min(values).item()))

            if ref_lower == "mean":
                return float(float(self.xp.mean(values).item()))

            if ref_lower == "max":
                return float(float(self.xp.max(values).item()))

            raise ValueError(
                f"Invalid {name}: {ref}"
            )

        return float(ref)

    # ==================================================================
    # WAVENUMBERS
    # ==================================================================

    def _build_wavenumbers(self):

        """
        Construct FFT wavenumbers.

        Ordering is shifted so that the resulting arrays correspond
        to the ordering used by MATLAB's:

            ifftshift(kgrid.k)
        """

        axes = []

        for n, d in zip(
            self.shape,
            (self.dx, self.dy, self.dz),
        ):

            if n <= 1:
                k = self.xp.zeros(
                    1,
                    dtype=self.dtype,
                )

            else:

                # Equivalent Fourier frequencies in rad/m
                k = (
                    2.0
                    * np.pi
                    * self.xp.fft.fftfreq(
                        n,
                        d=d,
                    )
                )

            axes.append(self.xp.asarray(k, dtype=self.dtype))

        # ------------------------------------------------------------
        # MATLAB kWave ordering:
        #
        # ifftshift(kgrid.k)
        #
        # The normal numpy FFT frequencies are already in FFT order,
        # so we use those directly for multiplication with fftn().
        # ------------------------------------------------------------

        if self.dim == 1:

            self.kx = axes[0]
            self.k = self.xp.abs(self.kx)

        elif self.dim == 2:

            self.kx = axes[0][:, None]
            self.ky = axes[1][None, :]

            self.k = self.xp.sqrt(
                self.kx**2
                + self.ky**2
            )

        elif self.dim == 3:

            self.kx = axes[0][:, None, None]
            self.ky = axes[1][None, :, None]
            self.kz = axes[2][None, None, :]

            self.k = self.xp.sqrt(
                self.kx**2
                + self.ky**2
                + self.kz**2
            )

        else:

            raise ValueError(
                "Only 1D, 2D, and 3D are supported."
            )

        self.k_squared = self.k**2

    # ==================================================================
    # FOURIER TRANSFORMS
    # ==================================================================

    def FT(self, x):
        """
        Forward Fourier transform.
        """

        return self.xp.fft.fftn(x)

    # ------------------------------------------------------------------

    def IT(self, x):
        """
        Inverse Fourier transform.

        The MATLAB implementation explicitly takes real().
        """

        return self.xp.fft.ifftn(x).real

    # ==================================================================
    # K-SPACE CORRECTION
    # ==================================================================

    def _calculate_kappa(self, dt):

        if not self.use_kspace:
            return self.xp.asarray(
                1.0,
                dtype=self.dtype,
            )

        kappa_argument = (
            dt
            * (
                self.diffusion_coeff_ref
                * self.k_squared
                + self.perfusion_coeff_ref
            )
        )

        kappa = self.xp.empty_like(
            kappa_argument,
            dtype=self.dtype,
        )

        # MATLAB:
        #
        # kappa = (1 - exp(-x)) / x
        #
        # Need explicit limit at x=0:
        #
        # lim x->0 = 1

        zero = (
            self.xp.abs(kappa_argument) == 0
        )

        nonzero = ~zero

        kappa[zero] = 1.0

        x = kappa_argument[nonzero]

        kappa[nonzero] = (
            -self.xp.expm1(-x) / x
        )

        return kappa

    # ==================================================================
    # DERIVATIVE OPERATORS
    # ==================================================================

    def _derivative_operators(self, kappa):

        """
        Construct:

            i*kx*sqrt(kappa)
            i*ky*sqrt(kappa)
            i*kz*sqrt(kappa)

        exactly corresponding to the periodic-boundary branch of
        MATLAB getDerivMatrix().
        """

        sqrt_kappa = self.xp.sqrt(kappa)

        deriv_x = (
            1j
            * self.kx
            * sqrt_kappa
        )

        if self.dim >= 2:
            deriv_y = (
                1j
                * self.ky
                * sqrt_kappa
            )
        else:
            deriv_y = None

        if self.dim >= 3:
            deriv_z = (
                1j
                * self.kz
                * sqrt_kappa
            )
        else:
            deriv_z = None

        return (
            deriv_x,
            deriv_y,
            deriv_z,
        )

    # ==================================================================
    # SOURCE TERM
    # ==================================================================

    def _source_update(self, kappa):

        Q = self.xp.asarray(
            self.Q,
            dtype=self.dtype,
        )
    
        if Q.ndim == 0:
            if Q.item() == 0:
                return 0.0
    
            Q = self.xp.full(
                self.shape,
                Q.item(),
                dtype=self.dtype,
            )
    
        elif not bool(self.xp.any(Q != 0)):
            return 0.0
    
        if self.q_scale_factor is None:
            raise ValueError(
                "A nonzero heat source requires density "
                "and specific_heat."
            )
    
        Q_ft = self.FT(Q)
    
        result = self.IT(
            kappa * Q_ft
        )
    
        return self.q_scale_factor * result

    # ==================================================================
    # PERFUSION TERM
    # ==================================================================

    def _perfusion_update(
        self,
        kappa,
    ):

        if bool(self.xp.all(
            self.perfusion_coeff == 0
        )):
            return 0.0

        temperature_difference = (
            self.T
            - self.blood_ambient_temperature
        )

        result = self.IT(
            kappa
            * self.FT(
                temperature_difference
            )
        )

        result = (
            -self.perfusion_coeff
            * result
        )

        return result

    # ==================================================================
    # HOMOGENEOUS DIFFUSION
    # ==================================================================

    def _homogeneous_diffusion_update(
        self,
        kappa,
    ):

        """
        MATLAB:

        d_term =
            diffusion_p1 .* diffusion_p2 .* ...
            IT(-k.^2 .* kappa .* FT(T))
        """

        coefficient = (
            self.diffusion_p1
            * self.diffusion_p2
        )

        T_ft = self.FT(self.T)

        result = self.IT(
            -self.k_squared
            * kappa
            * T_ft
        )

        return coefficient * result

    # ==================================================================
    # HETEROGENEOUS DIFFUSION
    # ==================================================================

    def _heterogeneous_diffusion_update(
        self,
        kappa,
        deriv_x,
        deriv_y,
        deriv_z,
    ):

        """
        MATLAB heterogeneous formulation:

            div(K grad(T)) / A

        implemented spectrally as:

            p1 * [
                D_x^-1 K D_x T
                +
                D_y^-1 K D_y T
                +
                D_z^-1 K D_z T
            ]

        where the derivative operators contain sqrt(kappa).
        """
        
        T_ft = self.FT(self.T)
        
        d_term = 0.0
        
        # X direction
        temp_x = self.xp.fft.ifftn(
            deriv_x * T_ft
        )
        
        flux_x = self.diffusion_p2 * temp_x
        
        d_term += self.xp.fft.ifftn(
            deriv_x * self.FT(flux_x)
        ).real
        
        # Y direction
        if self.dim >= 2:
        
            temp_y = self.xp.fft.ifftn(
                deriv_y * T_ft
            )
        
            flux_y = self.diffusion_p2 * temp_y
        
            d_term += self.xp.fft.ifftn(
                deriv_y * self.FT(flux_y)
            ).real
        
        # Z direction
        if self.dim >= 3:
        
            temp_z = self.xp.fft.ifftn(
                deriv_z * T_ft
            )
        
            flux_z = self.diffusion_p2 * temp_z
        
            d_term += self.xp.fft.ifftn(
                deriv_z * self.FT(flux_z)
            ).real
        
        return self.diffusion_p1 * d_term


    # ==================================================================
    # TIME STEP
    # ==================================================================

    def take_time_step(
        self,
        Nt: int,
        dt: float,
    ):
        """
        Advance the temperature field Nt time steps.

        Parameters
        ----------
        Nt:
            Number of time steps.

        dt:
            Time step [s].

        Returns
        -------
        np.ndarray
            Final temperature field.
        """

        Nt = int(Nt)

        if Nt <= 0:
            raise ValueError(
                "Nt must be positive."
            )

        dt = float(dt)

        if dt <= 0:
            raise ValueError(
                "dt must be positive."
            )

        sensor_start = self.time_steps_taken

        if self.display_updates:

            print(
                "Running k-Wave thermal simulation..."
            )

            print(
                f"  dimensions: {self.shape}"
            )

            print(
                f"  dt: {dt:.6g} s"
            )

            print(
                f"  Nt: {Nt}"
            )

            print(
                f"  t_end: {Nt * dt:.6g} s"
            )

            print(
                f"  heterogeneous: "
                f"{not self.flag_homogeneous}"
            )

        # ------------------------------------------------------------
        # k-space correction
        # ------------------------------------------------------------

        kappa = self._calculate_kappa(dt)

        # ------------------------------------------------------------
        # derivative operators
        # ------------------------------------------------------------

        if not self.flag_homogeneous:

            (
                deriv_x,
                deriv_y,
                deriv_z,
            ) = self._derivative_operators(
                kappa
            )

        else:

            deriv_x = None
            deriv_y = None
            deriv_z = None

        # ------------------------------------------------------------
        # source update
        #
        # Q does not depend on temperature and therefore can be
        # calculated once.
        # ------------------------------------------------------------

        q_term = self._source_update(
            kappa
        )

        # ------------------------------------------------------------
        # Time loop
        # ------------------------------------------------------------

        for t_index in range(Nt):

            # --------------------------------------------------------
            # Perfusion
            # --------------------------------------------------------

            p_term = self._perfusion_update(
                kappa
            )

            # --------------------------------------------------------
            # Diffusion
            # --------------------------------------------------------

            if self.flag_homogeneous:

                d_term = (
                    self._homogeneous_diffusion_update(
                        kappa
                    )
                )

            else:

                d_term = (
                    self._heterogeneous_diffusion_update(
                        kappa,
                        deriv_x,
                        deriv_y,
                        deriv_z,
                    )
                )

            # --------------------------------------------------------
            # Temperature update
            #
            # MATLAB:
            #
            # obj.T = obj.T + dt .* ...
            #     (d_term + p_term + q_term)
            # --------------------------------------------------------

            self.T = (
                self.T
                + dt
                * (
                    d_term
                    + p_term
                    + q_term
                )
            )

            # --------------------------------------------------------
            # CEM43
            # --------------------------------------------------------

            self._update_cem43(
                dt
            )

            # --------------------------------------------------------
            # Sensor
            # --------------------------------------------------------

            if self.num_sensor_points > 0:

                required_steps = sensor_start + Nt

                if (
                    self.sensor_data is None
                    or self.sensor_data.shape[1] < required_steps
                ):

                    old_data = self.sensor_data

                    self.sensor_data = self.xp.empty(
                        (
                            self.num_sensor_points,
                            required_steps,
                        ),
                        dtype=self.dtype,
                    )

                    if old_data is not None:
                        self.sensor_data[:, :old_data.shape[1]] = old_data

                values = self.T.ravel(order="C")[
                    self.sensor_mask_index
                ]

                self.sensor_data[
                    :,
                    sensor_start + t_index,
                ] = values

        self.time_steps_taken += Nt

        if self.display_updates:
            print("Simulation completed.")

        return self.T

    # ==================================================================
    # CEM43
    # ==================================================================

    def _update_cem43(self, dt):

        """
        Port of:

            obj.cem43 = obj.cem43 + dt ./ 60 .* ...
                (0.5 .* (obj.T >= 43) + ...
                 0.25 .* (obj.T >= 37 & obj.T < 43)).^(43 - obj.T);
        """

        factor = self.xp.where(
            self.T >= 43.0,
            0.5,
            self.xp.where(
                (self.T >= 37.0)
                & (self.T < 43.0),
                0.25,
                0.0,
            ),
        )

        # Avoid 0**positive -> 0 naturally.
        # For temperatures below 37 this corresponds to
        # zero thermal dose in the MATLAB expression.

        self.cem43 += (
            dt
            / 60.0
            * self.xp.where(
                factor > 0,
                factor ** (
                    43.0 - self.T
                ),
                0.0,
            )
        )

    # ==================================================================
    # LESION MAP
    # ==================================================================

    @property
    def lesion_map(self):

        return self.cem43 >= 240.0

    # ==================================================================
    # LESION SIZE
    # ==================================================================

    @property
    def lesion_size(self):

        if self.dim == 1:

            cell_volume = self.dx

        elif self.dim == 2:

            cell_volume = (
                self.dx
                * self.dy
            )

        else:

            cell_volume = (
                self.dx
                * self.dy
                * self.dz
            )

        value = self.xp.sum(self.lesion_map) * cell_volume
        return float(value.item()) if self.device == "gpu" else value

    # ==================================================================
    # DT LIMIT
    # ==================================================================

    @property
    def dt_limit(self):

        """
        Port of the MATLAB dt_limit property.

        This is primarily useful for determining whether the chosen
        timestep is inside the conditionally-stable region.

        The homogeneous k-space formulation is unconditionally stable
        when the reference coefficient is sufficiently large.
        """

        diffusion_coeff = (
            self.diffusion_p1
            * self.diffusion_p2
        )

        D_max = float(
            float(self.xp.max(diffusion_coeff).item())
        )

        k_max = float(
            float(self.xp.max(self.k).item())
        )

        k_min = float(
            float(self.xp.min(self.k).item())
        )

        P = self.perfusion_coeff
        
        # Uncorrected explicit diffusion/perfusion update.
        #
        # Applies when k-space correction is disabled or the
        # reference coefficients are zero.
        
        if (
            not self.use_kspace
            or (
                self.diffusion_coeff_ref == 0
                and self.perfusion_coeff_ref == 0
            )
        ):
        
            max_rate = float(
                self.xp.max(
                    diffusion_coeff * self.k_squared
                    + P
                ).item()
            )
        
            if max_rate <= 0:
                return np.inf
        
            return 2.0 / max_rate


        # ------------------------------------------------------------
        # No perfusion
        # ------------------------------------------------------------

        if bool(self.xp.all(P == 0)):

            if (
                self.diffusion_coeff_ref
                >= D_max / 2.0
            ):

                return np.inf

            denominator = (
                self.diffusion_coeff_ref
                * k_max**2
            )

            if denominator == 0:
                return np.inf

            argument = (
                1.0
                - 2.0
                * self.diffusion_coeff_ref
                / D_max
            )

            if argument <= 0:
                return np.inf

            return (
                -self.xp.log(argument)
                / denominator
            )

        # ------------------------------------------------------------
        # Perfusion
        # ------------------------------------------------------------

        reference = (
            self.diffusion_coeff_ref
            * k_max**2
            + self.perfusion_coeff_ref
        )

        condition = (
            0.5
            * (
                diffusion_coeff
                * k_max**2
                + P
            )
        )

        if reference >= float(self.xp.max(condition).item()):

            return np.inf

        denominator_max = self.xp.max(
            diffusion_coeff
            * k_max**2
            + P
        )

        arg_max = (
            1.0
            - 2.0
            * (
                self.diffusion_coeff_ref
                * k_max**2
                + self.perfusion_coeff_ref
            )
            / denominator_max
        )

        if arg_max <= 0:
            kmax_val = np.inf

        else:

            kmax_val = (
                -self.xp.log(arg_max)
                / (
                    self.diffusion_coeff_ref
                    * k_max**2
                    + self.perfusion_coeff_ref
                )
            )

        denominator_min = self.xp.max(
            diffusion_coeff
            * k_min**2
            + P
        )

        arg_min = (
            1.0
            - 2.0
            * (
                self.diffusion_coeff_ref
                * k_min**2
                + self.perfusion_coeff_ref
            )
            / denominator_min
        )

        if arg_min <= 0:

            kmin_val = np.inf

        else:

            denom = (
                self.diffusion_coeff_ref
                * k_min**2
                + self.perfusion_coeff_ref
            )

            if denom == 0:

                kmin_val = np.inf

            else:

                kmin_val = (
                    -self.xp.log(arg_min)
                    / denom
                )

        return min(
            kmax_val,
            kmin_val,
        )

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(
        self,
        T0=None,
    ):
        """
        Reset temperature and thermal dose.

        If T0 is omitted, the original source.T0 is used.
        """

        if T0 is None:
            T0 = self.source.T0

        T0 = self.xp.asarray(
            T0,
            dtype=self.dtype,
        )

        if T0.ndim == 0:

            self.T = self.xp.full(
                self.shape,
                T0.item(),
                dtype=self.dtype,
            )

        else:

            if T0.shape != self.shape:
                raise ValueError(
                    f"T0 shape {T0.shape} "
                    f"does not match {self.shape}"
                )

            self.T = T0.copy()

        self.cem43.fill(0)

        self.time_steps_taken = 0

        self.sensor_data = None
