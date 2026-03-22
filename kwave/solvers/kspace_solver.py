"""
Minimal N-D k-Wave Python Backend.
Supports 1D, 2D, and 3D k-space pseudospectral wave propagation.

Design: Simulation class with setup/step separation for debuggability.
"""
from types import SimpleNamespace

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

# =============================================================================
# Utility Functions
# =============================================================================


# Zero arrays/scalars from MATLAB indicate "disabled" features
def _is_enabled(x):
    return x is not None and not (np.all(x == 0) if hasattr(x, "__len__") else x == 0)


def _to_cpu(x):
    return x.get() if hasattr(x, "get") else x


def _expand_to_grid(val, grid_shape, xp, name="parameter"):
    if val is None:
        raise ValueError(f"Missing required parameter: {name}")
    arr = xp.array(val, dtype=float).flatten(order="F")
    grid_size = int(np.prod(grid_shape))
    if arr.size == 1:
        return xp.full(grid_shape, float(arr[0]), dtype=float)
    if arr.size == grid_size:
        return arr.reshape(grid_shape, order="F")
    raise ValueError(f"{name} size {arr.size} incompatible with grid size {grid_size}")


# =============================================================================
# Simulation Class
# =============================================================================


class Simulation:
    """N-D k-Space Pseudospectral Wave Propagator with setup/step separation.

    Usage:
        sim = Simulation(kgrid, medium, source, sensor)
        sim.setup()                    # Inspect sim.p, sim.op_grad_list, etc.
        sim.step()                     # One time step, inspect fields
        results = sim.run()            # Run remaining steps, return results

    Or simply:
        results = Simulation(kgrid, medium, source, sensor).run()
    """

    def __init__(self, kgrid, medium, source, sensor, backend="auto", use_sg=True, use_kspace=True, smooth_p0=True):
        self.kgrid = kgrid
        self.medium = medium
        self.source = source
        self.sensor = sensor
        self.use_sg = use_sg
        self.use_kspace = use_kspace
        self.smooth_p0 = smooth_p0
        if backend == "gpu":
            if cp is None:
                raise ImportError("CuPy is required for GPU backend but is not installed. Install with: pip install cupy-cuda12x")
            self.xp = cp
        elif backend == "auto":
            self.xp = cp if cp else np
        else:
            self.xp = np
        self._is_setup = False
        self.t = 0  # Current time step

    def setup(self):
        """Initialize operators and fields. Call before step()/run()."""
        xp = self.xp

        self.spacing = []
        grid_dims = []
        for axis in ["x", "y", "z"]:
            N = getattr(self.kgrid, f"N{axis}", None)
            d = getattr(self.kgrid, f"d{axis}", None)
            if N is not None and d is not None:
                grid_dims.append(int(N))
                self.spacing.append(float(d))
            else:
                break

        self.grid_shape = tuple(grid_dims)
        self.ndim = len(grid_dims)
        self.Nt = int(self.kgrid.Nt)
        self.dt = float(self.kgrid.dt)

        self.c0 = _expand_to_grid(self.medium.sound_speed, self.grid_shape, xp, "sound_speed")
        self.rho0 = _expand_to_grid(getattr(self.medium, "density", 1000.0), self.grid_shape, xp, "density")
        self.c_ref = float(xp.max(self.c0))

        self._setup_sensor_mask()
        self._setup_pml()
        self._setup_kspace_operators()
        self._setup_physics_operators()
        self._setup_source_operators()
        self._setup_fields()

        self._is_setup = True
        return self

    def _setup_sensor_mask(self):
        """Build self._extract(field) → sensor values."""
        xp = self.xp

        mask_raw = getattr(self.sensor, "mask", None)
        grid_numel = int(np.prod(self.grid_shape))

        def _is_binary(arr):
            """numel matches grid → boolean mask (matches MATLAB isCartesian logic)."""
            return arr.size == 1 or arr.size == grid_numel

        def _is_cartesian(arr):
            """First dimension matches ndim, remaining are query points."""
            return arr.ndim == 2 and arr.shape[0] == self.ndim

        if mask_raw is None:
            self.n_sensor_points = grid_numel
            self._extract = lambda f: f.flatten(order="F")
        else:
            mask_arr = np.asarray(mask_raw, dtype=float)
            # Check Cartesian first to avoid ambiguity when size == grid_numel
            if _is_cartesian(mask_arr):
                self._setup_cartesian_extract(mask_arr)
            elif _is_binary(mask_arr):
                bmask = xp.array(mask_arr, dtype=bool).flatten(order="F")
                if bmask.size == 1:
                    bmask = xp.full(grid_numel, bool(bmask[0]), dtype=bool)
                self.n_sensor_points = int(xp.sum(bmask))
                idx = xp.where(bmask)[0]
                self._extract = lambda f, _i=idx: f.flatten(order="F")[_i]
            else:
                raise ValueError(
                    f"Sensor mask shape {mask_arr.shape} is neither binary " f"(numel={grid_numel}) nor Cartesian ({self.ndim}, N_points)"
                )

        # Parse sensor.record (default matches MATLAB: pressure time-series + final snapshot)
        record = getattr(self.sensor, "record", ("p", "p_final"))
        if isinstance(record, str):
            record = (record,)
        self.record = set(record)
        if "u" in self.record:
            self.record.discard("u")
            self.record.update(f"u{a}" for a in "xyz"[: self.ndim])
        if "u_staggered" in self.record:
            self.record.discard("u_staggered")
            self.record.update(f"u{a}_staggered" for a in "xyz"[: self.ndim])

        # Expand shorthand velocity keys: u_max → ux_max, uy_max, uz_max
        for suffix in ("_max", "_min", "_rms", "_final"):
            if f"u{suffix}" in self.record:
                self.record.discard(f"u{suffix}")
                self.record.update(f"u{a}{suffix}" for a in "xyz"[: self.ndim])

        # Expand intensity shorthands
        if "I" in self.record or "I_avg" in self.record:
            if "I" in self.record:
                self.record.discard("I")
                self.record.update(f"I{a}" for a in "xyz"[: self.ndim])
            if "I_avg" in self.record:
                self.record.discard("I_avg")
                self.record.update(f"I{a}_avg" for a in "xyz"[: self.ndim])

        # Snapshot of what the user actually requested (after shorthand expansion)
        self._requested_record = set(self.record)

        # Add internal dependencies: aggregates need base time-series, intensity needs p + u
        for key in list(self.record):
            for suffix in ("_max", "_min", "_rms"):
                if key.endswith(suffix):
                    self.record.add(key[: -len(suffix)])
        if any(k.startswith("I") for k in self._requested_record):
            self.record.update(["p"] + [f"u{a}" for a in "xyz"[: self.ndim]])

        # MATLAB uses 1-based indexing; convert to Python's 0-based for array slicing
        record_start_raw = getattr(self.sensor, "record_start_index", 1)
        self.record_start_index = int(record_start_raw) - 1
        self.num_recorded_time_points = self.Nt - self.record_start_index

    def _setup_cartesian_extract(self, cart_pos):
        """Build self._extract using bilinear/trilinear interpolation on the regular grid."""
        xp = self.xp
        cart = cart_pos if cart_pos.ndim == 2 else cart_pos.reshape(self.ndim, -1)
        self.n_sensor_points = cart.shape[1]

        # Reconstruct kWaveGrid coordinate axes (centered at origin)
        axis_coords = [(np.arange(N) - N // 2) * d for N, d in zip(self.grid_shape, self.spacing)]

        if self.ndim == 1:
            x_vec, cart_x = axis_coords[0], cart.flatten()

            def _extract_1d_interp(f):
                return xp.asarray(np.interp(cart_x, x_vec, _to_cpu(f).flatten(order="F")))

            self._extract = _extract_1d_interp
        else:
            # Convert Cartesian positions to continuous grid indices
            frac_idx = np.array([(cart[d] - axis_coords[d][0]) / self.spacing[d] for d in range(self.ndim)])  # (ndim, n_pts)
            int_idx = np.clip(np.floor(frac_idx).astype(int), 0, np.array(self.grid_shape)[:, None] - 2)
            local = frac_idx - int_idx

            # F-order strides and 2^ndim corner enumeration
            strides = np.cumprod([1] + list(self.grid_shape[:-1]))
            n_corners = 2**self.ndim
            corner_indices = np.zeros((self.n_sensor_points, n_corners), dtype=int)
            corner_weights = np.ones((self.n_sensor_points, n_corners))
            for c in range(n_corners):
                for d in range(self.ndim):
                    bit = (c >> d) & 1
                    corner_indices[:, c] += (int_idx[d] + bit) * strides[d]
                    corner_weights[:, c] *= local[d] if bit else (1 - local[d])

            corner_indices = xp.array(corner_indices)
            corner_weights = xp.array(corner_weights)

            def _extract_bilinear(f):
                return (f.flatten(order="F")[corner_indices] * corner_weights).sum(axis=1)

            self._extract = _extract_bilinear

    def _setup_pml(self):
        """Build Perfectly Matched Layer absorption operators for each dimension."""
        xp = self.xp
        axis_names = ["x", "y", "z"]

        # TODO: reuse kwave/utils/pml.py:get_pml instead of reimplementing (needs xp= arg for CuPy)
        self.pml_list = []  # For pressure/density
        self.pml_sg_list = []  # For velocity (staggered grid)
        self.pml_sizes = []  # PML thickness per axis (for interior slicing)

        from kwave.utils.pml import get_pml

        for axis in range(self.ndim):
            N = self.grid_shape[axis]
            dx = self.spacing[axis]
            name = axis_names[axis]

            pml_size = int(getattr(self.kgrid, f"pml_size_{name}", 0))
            pml_alpha = float(getattr(self.kgrid, f"pml_alpha_{name}", 0))
            self.pml_sizes.append(pml_size if pml_alpha != 0 else 0)

            if pml_size == 0 or pml_alpha == 0:
                shape = [1] * self.ndim
                shape[axis] = N
                self.pml_list.append(xp.ones(shape, dtype=float))
                self.pml_sg_list.append(xp.ones(shape, dtype=float))
            else:
                # dimension=2 gives shape (1, N) which we reshape for broadcasting
                pml = get_pml(N, dx, self.dt, self.c_ref, pml_size, pml_alpha, staggered=False, dimension=2, xp=xp)
                pml_sg = get_pml(N, dx, self.dt, self.c_ref, pml_size, pml_alpha, staggered=True, dimension=2, xp=xp)

                shape = [1] * self.ndim
                shape[axis] = N
                self.pml_list.append(pml.flatten().reshape(shape))
                self.pml_sg_list.append(pml_sg.flatten().reshape(shape))

    def _setup_kspace_operators(self):
        """Build k-space gradient/divergence operators for each dimension."""
        xp = self.xp
        self.k_list = []

        # First pass: build k-vectors for each dimension
        for axis, (N, dx) in enumerate(zip(self.grid_shape, self.spacing)):
            k = 2 * np.pi * xp.fft.fftfreq(N, d=dx)
            shape = [1] * self.ndim
            shape[axis] = N
            self.k_list.append(k.reshape(shape))

        k_mag_sq = self.k_list[0] ** 2
        for k in self.k_list[1:]:
            k_mag_sq = k_mag_sq + k**2
        k_mag = xp.sqrt(k_mag_sq)
        self._k_mag = k_mag  # cached for _fractional_laplacian
        if self.use_kspace:
            self.kappa = xp.sinc((self.c_ref * k_mag * self.dt / 2) / np.pi)
            self.source_kappa = xp.cos(self.c_ref * k_mag * self.dt / 2)
        else:
            self.kappa = 1
            self.source_kappa = 1

        # Per-dimension operators with kappa pre-multiplied to avoid per-step work
        # Staggered grid shifts: exp(±jk*dx/2) offsets between pressure and velocity grids
        self.op_grad_list = []
        self.op_div_list = []
        for axis, (N, dx) in enumerate(zip(self.grid_shape, self.spacing)):
            k = self.k_list[axis]
            if self.use_sg:
                self.op_grad_list.append(1j * k * xp.exp(1j * k * dx / 2) * self.kappa)
                self.op_div_list.append(1j * k * xp.exp(-1j * k * dx / 2) * self.kappa)
            else:
                self.op_grad_list.append(1j * k * self.kappa)
                self.op_div_list.append(1j * k * self.kappa)

    def _setup_physics_operators(self):
        """Build absorption, dispersion, and nonlinearity operators."""
        xp = self.xp

        # Absorption/dispersion
        alpha_coeff_raw = getattr(self.medium, "alpha_coeff", 0)
        if not _is_enabled(alpha_coeff_raw):
            self._absorption = lambda div_u: 0
            self._dispersion = lambda rho: 0
        else:
            alpha_coeff = _expand_to_grid(alpha_coeff_raw, self.grid_shape, xp, "alpha_coeff")
            alpha_power = float(xp.array(getattr(self.medium, "alpha_power", 1.5)).flatten()[0])
            alpha_np = 100 * alpha_coeff * (1e-6 / (2 * np.pi)) ** alpha_power / (20 * np.log10(np.e))

            if abs(alpha_power - 2.0) < 1e-10:  # Stokes
                self._absorption = lambda div_u: -2 * alpha_np * self.c0 * self.rho0 * div_u
                self._dispersion = lambda rho: 0
            else:  # Power-law with fractional Laplacian
                tau = -2 * alpha_np * self.c0 ** (alpha_power - 1)
                eta = 2 * alpha_np * self.c0**alpha_power * xp.tan(np.pi * alpha_power / 2)
                nabla1 = self._fractional_laplacian(alpha_power - 2)
                nabla2 = self._fractional_laplacian(alpha_power - 1)
                # Fractional Laplacian already includes full k-space structure; no kappa needed
                self._absorption = lambda div_u: tau * self._diff(self.rho0 * div_u, nabla1)
                self._dispersion = lambda rho: eta * self._diff(rho, nabla2)

        # Nonlinearity
        BonA_raw = getattr(self.medium, "BonA", 0)
        self._has_nonlinearity = _is_enabled(BonA_raw)
        if not self._has_nonlinearity:
            self._nonlinearity = lambda rho: 0
            self._nonlinear_factor = lambda rho: 1.0
        else:
            BonA = _expand_to_grid(BonA_raw, self.grid_shape, xp, "BonA")
            self._nonlinearity = lambda rho: BonA * rho**2 / (2 * self.rho0)
            self._nonlinear_factor = lambda rho: (2 * rho + self.rho0) / self.rho0

    def _setup_source_operators(self):
        """Build time-varying source injection operators.

        Source scaling follows kspaceFirstOrder_scaleSourceTerms.m.

        Pressure sources are injected as mass sources into the split density
        fields (rho_x, rho_y, ...).  Since p = c0^2 * (rho_x + rho_y + ...),
        the user-supplied pressure must be converted to density and divided
        equally across N = ndim components:

            dirichlet:  source.p / (N * c0^2)
            additive:   source.p * 2*dt / (N * c0 * dx)

        The 1/c0^2 converts pressure to density (equation of state).
        The 1/N splits evenly so the sum reconstructs the correct total.
        The 2*dt/(c0*dx) factor accounts for the leapfrog time discretization
        (see Cox et al., IEEE IUS 2018).

        Velocity sources use per-axis grid spacing (dx, dy, dz):

            additive:   source.ux * 2*c0*dt / dx   (and dy for uy, dz for uz)
        """
        xp = self.xp
        grid_size = int(np.prod(self.grid_shape))

        def build_op(mask_raw, signal_raw, mode, scale):
            """Build a source injection operator for one field variable."""
            if not (_is_enabled(mask_raw) and _is_enabled(signal_raw)):
                return None

            mask = xp.array(mask_raw, dtype=bool).flatten(order="F")
            if mask.size == 1:
                mask = xp.full(self.grid_shape, bool(mask[0]), dtype=bool).flatten(order="F")
            n_src = int(xp.sum(mask))

            signal_arr = xp.array(signal_raw, dtype=float, order="F")
            if signal_arr.ndim == 1:
                signal = signal_arr.reshape(1, -1)
            else:
                signal = signal_arr.reshape(-1, signal_arr.shape[-1], order="F") if signal_arr.ndim > 2 else signal_arr

            scaled = signal * xp.atleast_1d(xp.asarray(scale))[:, None]
            signal_len = scaled.shape[1]

            def get_val(t):
                if scaled.shape[0] == 1:
                    return xp.full(n_src, float(scaled[0, t]))
                return scaled[:, t]

            def dirichlet(t, field):
                if t >= signal_len:
                    return field
                flat = field.flatten(order="F")
                flat[mask] = get_val(t)
                return flat.reshape(self.grid_shape, order="F")

            # Pre-allocate buffer to avoid per-step allocation
            _src_buf = xp.zeros(grid_size, dtype=float)

            def additive_kspace(t, field):
                if t >= signal_len:
                    return field
                _src_buf[:] = 0
                _src_buf[mask] = get_val(t)
                src = _src_buf.reshape(self.grid_shape, order="F")
                return field + self._diff(src, self.source_kappa)

            ops = {"dirichlet": dirichlet, "additive": additive_kspace}
            if mode not in ops:
                raise ValueError(f"Unknown source mode: {mode!r}. Use 'additive' or 'dirichlet'.")
            return ops[mode]

        def source_scale(mask_raw, c0):
            """Get per-source-point sound speed values."""
            mask = xp.array(mask_raw, dtype=bool).flatten(order="F")
            if mask.size == 1:
                mask = xp.full(self.grid_shape, bool(mask[0]), dtype=bool).flatten(order="F")
            c0_flat = c0.flatten(order="F")
            n_src = int(xp.sum(mask))
            return c0_flat[mask] if c0_flat.size > 1 else xp.full(n_src, float(c0_flat))

        # --- Pressure source (per-axis spacing for non-isotropic grids) ---
        p_mask = getattr(self.source, "p_mask", 0)
        p_signal = getattr(self.source, "p", 0)
        p_mode = getattr(self.source, "p_mode", "additive")
        N = self.ndim
        if _is_enabled(p_mask) and _is_enabled(p_signal):
            c0_src = source_scale(p_mask, self.c0)
            if p_mode == "dirichlet":
                scale = 1.0 / (N * c0_src**2)
                op = build_op(p_mask, p_signal, p_mode, scale)
                self._source_p_ops = [op] * self.ndim
            else:
                # Per-axis: rho_i += source.p * 2*dt / (N * c0 * d_i)
                self._source_p_ops = []
                for i in range(self.ndim):
                    di = self.spacing[i]
                    scale_i = 2 * self.dt / (N * c0_src * di)
                    self._source_p_ops.append(build_op(p_mask, p_signal, p_mode, scale_i))
        else:
            self._source_p_ops = [lambda t, field: field] * self.ndim

        # --- Velocity sources (per-axis grid spacing) ---
        u_mask = getattr(self.source, "u_mask", 0)
        u_mode = getattr(self.source, "u_mode", "additive")
        self._source_u_ops = []
        for i, vel in enumerate(["ux", "uy", "uz"][: self.ndim]):
            u_signal = getattr(self.source, vel, 0)
            di = self.spacing[i]  # dx for ux, dy for uy, dz for uz
            if _is_enabled(u_mask) and _is_enabled(u_signal):
                c0_src = source_scale(u_mask, self.c0)
                if u_mode == "dirichlet":
                    scale = xp.ones_like(c0_src)
                else:
                    # u_i += source.u_i * 2*c0*dt / d_i
                    scale = 2 * c0_src * self.dt / di
                op = build_op(u_mask, u_signal, u_mode, scale)
                self._source_u_ops.append(op)
            else:
                self._source_u_ops.append(lambda t, field: field)

    def _setup_fields(self):
        """Initialize pressure, velocity, and density fields."""
        xp = self.xp

        self.p = xp.zeros(self.grid_shape, dtype=float)
        self.u = [xp.zeros(self.grid_shape, dtype=float) for _ in range(self.ndim)]
        # Split density per dimension enables independent PML absorption in each direction
        self.rho_split = [xp.zeros(self.grid_shape, dtype=float) for _ in range(self.ndim)]

        if self.use_sg:
            self.rho0_staggered = [self._stagger(self.rho0, axis) for axis in range(self.ndim)]
        else:
            self.rho0_staggered = [self.rho0] * self.ndim

        # Precompute fixed coefficients used every time step
        self.c0_sq = self.c0**2
        self.dt_over_rho0 = [self.dt / rho for rho in self.rho0_staggered]

        # Sensor data storage (sized based on record_start_index)
        self.sensor_data = {}
        if "p" in self.record:
            self.sensor_data["p"] = xp.zeros((self.n_sensor_points, self.num_recorded_time_points), dtype=float)
        for a in "xyz"[: self.ndim]:
            for suffix in ("", "_staggered"):
                v = f"u{a}{suffix}"
                if v in self.record:
                    self.sensor_data[v] = xp.zeros((self.n_sensor_points, self.num_recorded_time_points), dtype=float)

        # Spectral shift: move velocity from staggered (mid-cell) to collocated (pressure) grid
        if any(f"u{a}" in self.sensor_data for a in "xyz"[: self.ndim]):
            self.unstagger_ops = [xp.exp(-1j * self.k_list[ax] * self.spacing[ax] / 2) for ax in range(self.ndim)]

        # Initial pressure source (p0)
        p0_raw = getattr(self.source, "p0", 0)
        if _is_enabled(p0_raw):
            p0 = _expand_to_grid(p0_raw, self.grid_shape, xp, "p0")
            if self.smooth_p0 and self.ndim >= 2:
                from kwave.utils.filters import smooth

                p0 = xp.asarray(smooth(_to_cpu(p0), restore_max=True))
            self._p0_initial = p0
        else:
            self._p0_initial = None

    def step(self):
        """Advance simulation by one time step. Returns self for chaining."""
        if not self._is_setup:
            self.setup()
        if self.t >= self.Nt:
            return self

        xp = self.xp

        # Momentum equation: du_i/dt = -grad_i(p)/rho, with PML
        # Share forward FFT of p across all gradient axes
        P = xp.fft.fftn(self.p)
        for i in range(self.ndim):
            pml_sg = self.pml_sg_list[i]
            grad_p_i = xp.real(xp.fft.ifftn(self.op_grad_list[i] * P))
            self.u[i] = pml_sg * (pml_sg * self.u[i] - self.dt_over_rho0[i] * grad_p_i)
            self.u[i] = self._source_u_ops[i](self.t, self.u[i])

        # Mass conservation: drho_i/dt = -rho0 * div_i(u_i), with PML
        # Only compute rho_total before the loop when nonlinearity needs it
        nl_factor = self._nonlinear_factor(sum(self.rho_split)) if self._has_nonlinearity else 1.0
        div_u_total = xp.zeros(self.grid_shape, dtype=float)
        for i in range(self.ndim):
            pml = self.pml_list[i]
            div_u_i = self._diff(self.u[i], self.op_div_list[i])
            div_u_total += div_u_i
            self.rho_split[i] = pml * (pml * self.rho_split[i] - self.dt * self.rho0 * div_u_i * nl_factor)
            self.rho_split[i] = self._source_p_ops[i](self.t, self.rho_split[i])

        rho_total = self.rho_split[0] + sum(self.rho_split[1:]) if self.ndim > 1 else self.rho_split[0]
        self.p = self.c0_sq * (rho_total + self._absorption(div_u_total) - self._dispersion(rho_total) + self._nonlinearity(rho_total))

        # At t=0, override equation of state with p0; set u(-dt/2) for leapfrog
        # Velocity uses negative half-step: u(-dt/2) = -dt/(2*rho0) * grad(p0)
        # (MATLAB: kspaceFirstOrder_initialiseKgridVariables.m)
        if self.t == 0 and self._p0_initial is not None:
            self.p = self._p0_initial.copy()
            for i in range(self.ndim):
                self.rho_split[i] = self._p0_initial / (self.c0_sq * self.ndim)
                self.u[i] = -(self.dt_over_rho0[i] / 2) * self._diff(self.p, self.op_grad_list[i])

        # Record sensor data (binary: index extraction, Cartesian: bilinear/trilinear interpolation)
        if self.t >= self.record_start_index:
            file_index = self.t - self.record_start_index
            if "p" in self.sensor_data:
                self.sensor_data["p"][:, file_index] = self._extract(self.p)
            for i, a in enumerate("xyz"[: self.ndim]):
                if f"u{a}" in self.sensor_data:  # colocated
                    shifted = xp.real(xp.fft.ifftn(self.unstagger_ops[i] * xp.fft.fftn(self.u[i])))
                    self.sensor_data[f"u{a}"][:, file_index] = self._extract(shifted)
                if f"u{a}_staggered" in self.sensor_data:  # raw staggered
                    self.sensor_data[f"u{a}_staggered"][:, file_index] = self._extract(self.u[i])
        self.t += 1
        return self

    def run(self):
        """Run simulation to completion. Returns results dict."""
        if not self._is_setup:
            self.setup()
        while self.t < self.Nt:
            self.step()
        result = {k: _to_cpu(v) for k, v in self.sensor_data.items()}
        result.update(_compute_aggregates(result, self.ndim, self.record))
        if "p" in result and any(f"u{a}" in result for a in "xyz"):
            if any(k.startswith("I") for k in self._requested_record):
                result.update(acoustic_intensity(result))
        # Final-state snapshots: interior grid (excluding PML) at last timestep
        interior = tuple(slice(s, N - s if s else None) for s, N in zip(self.pml_sizes, self.grid_shape))
        if "p_final" in self.record:
            result["p_final"] = _to_cpu(self.p[interior].copy())
        if any(f"u{a}_final" in self.record for a in "xyz"):
            for i, a in enumerate("xyz"[: self.ndim]):
                if f"u{a}_final" in self.record:
                    result[f"u{a}_final"] = _to_cpu(self.u[i][interior].copy())
        return {k: v for k, v in result.items() if k in self._requested_record}

    # Helper methods
    def _diff(self, f, op):
        """Spectral differentiation: F^-1[op * F[f]].

        For gradient/divergence ops, kappa is pre-multiplied at setup.
        For fractional Laplacian ops (absorption/dispersion), kappa is not needed.
        """
        xp = self.xp
        return xp.real(xp.fft.ifftn(op * xp.fft.fftn(f)))

    def _stagger(self, arr, axis):
        """Compute staggered grid values (average neighbors along axis)."""
        if arr.size == 1:
            return arr
        xp = self.xp
        lo = [slice(None)] * arr.ndim
        hi = [slice(None)] * arr.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        avg = 0.5 * (arr[tuple(lo)] + arr[tuple(hi)])
        last = [slice(None)] * arr.ndim
        last[axis] = slice(-1, None)
        return xp.concatenate([avg, arr[tuple(last)]], axis=axis)

    def _fractional_laplacian(self, power):
        """N-D fractional Laplacian |k|^power, using cached k_mag from setup."""
        xp = self.xp
        k_mag = self._k_mag
        with np.errstate(divide="ignore", invalid="ignore"):
            return xp.where(k_mag == 0, 0, k_mag**power)


# =============================================================================
# Post-Processing
# =============================================================================


def _compute_aggregates(result, ndim, record):
    """Compute max/min/rms from time-series. Only computes requested keys."""
    out = {}
    for prefix in ["p"] + [f"u{a}" for a in "xyz"[:ndim]]:
        ts = result.get(prefix)
        if ts is None:
            continue
        if f"{prefix}_max" in record:
            out[f"{prefix}_max"] = np.max(ts, axis=-1)
        if f"{prefix}_min" in record:
            out[f"{prefix}_min"] = np.min(ts, axis=-1)
        if f"{prefix}_rms" in record:
            out[f"{prefix}_rms"] = np.sqrt(np.mean(ts**2, axis=-1))
    return out


def acoustic_intensity(result):
    """Compute acoustic intensity from simulation result dict.

    Temporally shifts velocity forward by dt/2 (Fourier interpolant)
    to align with pressure, then computes I = p * u per component.

    Returns dict with Ix, Iy, Iz, Ix_avg, Iy_avg, Iz_avg.
    """
    p = result["p"]
    n_time = p.shape[-1]
    freq = 2 * np.pi * np.arange(-(n_time // 2), n_time - n_time // 2) / n_time
    # DC is already 0; Nyquist (freq[0] = -π) is implicitly suppressed by np.real() below
    shift_op = np.fft.ifftshift(np.exp(1j * freq * 0.5))

    out = {}
    for a in "xyz":
        u = result.get(f"u{a}")
        if u is None:
            break
        u_shifted = np.real(np.fft.ifft(shift_op * np.fft.fft(u, axis=-1), axis=-1))
        out[f"I{a}"] = p * u_shifted
        out[f"I{a}_avg"] = np.mean(out[f"I{a}"], axis=-1)
    return out


# =============================================================================
# MATLAB Interop
# =============================================================================


def _to_namespace(d):
    """Convert dict to SimpleNamespace."""
    return SimpleNamespace(**dict(d))


# MATLAB code uses both c0/sound_speed and rho0/density; normalize to canonical names
def _normalize_medium(m):
    d = dict(m)
    if "c0" in d and "sound_speed" not in d:
        d["sound_speed"] = d.pop("c0")
    if "rho0" in d and "density" not in d:
        d["density"] = d.pop("rho0")
    return d


def create_simulation(kgrid, medium, source, sensor, backend="auto"):
    """MATLAB interop: create Simulation from dicts (for step-by-step debugging)."""
    return Simulation(_to_namespace(kgrid), _to_namespace(_normalize_medium(medium)), _to_namespace(source), _to_namespace(sensor), backend)


def simulate_from_dicts(kgrid, medium, source, sensor, backend="auto"):
    """MATLAB interop entry point."""
    return create_simulation(kgrid, medium, source, sensor, backend).run()
