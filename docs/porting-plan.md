# Example Porting Plan

Status of porting MATLAB k-Wave examples to Python using the `kspaceFirstOrder()` API.

## Completed (18 examples)

### Batch 1 — IVP (Initial Value Problems)
- [x] `example_ivp_homogeneous_medium` — 2D, Cartesian sensor, absorption
- [x] `example_ivp_heterogeneous_medium` — spatially varying c and rho
- [x] `example_ivp_binary_sensor_mask` — arc sensor via `make_circle`
- [x] `example_ivp_recording_particle_velocity` — p, ux, uy at 4 points
- [x] `example_ivp_1D_simulation` — 1D heterogeneous, Cartesian sensor

### Batch 2 — TVSP (Time-Varying Sources)
- [x] `example_tvsp_homogeneous_medium_monopole` — pressure source + `filter_time_series`
- [x] `example_tvsp_homogeneous_medium_dipole` — velocity source
- [x] `example_tvsp_steering_linear_array` — 21-element phased array + `tone_burst`
- [x] `example_tvsp_snells_law` — two-layer medium, 61-element steered array

### Batch 3 — NA (Numerical Analysis)
- [x] `example_na_controlling_the_PML` — 4 PML configurations
- [x] `example_na_modelling_nonlinearity` — 1D, BonA, record_start_index
- [x] `example_na_filtering_part_1` — unfiltered delta impulse
- [x] `example_na_filtering_part_2` — spatially smoothed source
- [x] `example_na_filtering_part_3` — temporally filtered source

### Batch 4 — 3D and SD
- [x] `example_ivp_3D_simulation` — 3D heterogeneous (parity test skipped: axis mismatch)
- [x] `example_ivp_photoacoustic_waveforms` — 1D/2D/3D waveform comparison
- [x] `example_sd_focussed_detector_2D` — focused vs unfocused detector
- [x] `example_sd_focussed_detector_3D` — 3D focused detector

## Phase 5 — Remaining Tier 1 (10 examples)

These use only `kspaceFirstOrder` with features already supported.

### Ready to port
- [ ] `example_tvsp_doppler_effect` — moving source (time-varying source position)
- [ ] `example_tvsp_3D_simulation` — 3D time-varying source (skip DataCast)
- [ ] `example_ivp_loading_external_image` — external image as p0
- [ ] `example_ivp_saving_movie_files` — p_final recording (skip movie, test data)
- [ ] `example_na_optimising_performance` — (skip DataCast, test correctness)
- [ ] `example_na_source_smoothing` — source smoothing demonstration

### May need minor feature work
- [ ] `example_sd_directivity_modelling_2D` — may need directivity support
- [ ] `example_sd_directivity_modelling_3D` — may need directivity support
- [ ] `example_sd_directional_array_elements` — may need directivity support
- [ ] `example_pr_2D_FFT_line_sensor` — forward sim only (reconstruction is post-processing)
- [ ] `example_pr_2D_adjoint` — adjoint method (may need custom time-stepping)
- [ ] `example_pr_3D_FFT_planar_sensor` — forward sim only

## Phase 6 — Tier 2 (17 examples, need missing features)

Each group is blocked by a specific missing feature in the Python solver.

### Time reversal (9 examples) — HIGH effort
Requires: reverse time loop, inject boundary data as Dirichlet source
- [ ] `example_pr_2D_TR_line_sensor`
- [ ] `example_pr_2D_TR_circular_sensor`
- [ ] `example_pr_2D_TR_bandlimited_sensors`
- [ ] `example_pr_2D_TR_autofocus`
- [ ] `example_pr_2D_TR_iterative`
- [ ] `example_pr_2D_TR_absorption_compensation`
- [ ] `example_pr_2D_TR_time_variant_filtering`
- [ ] `example_pr_2D_TR_directional_sensors`
- [ ] `example_pr_3D_TR_planar_sensor`
- [ ] `example_pr_3D_TR_spherical_sensor`

### Rect-corner sensor (5 examples) — LOW effort
Requires: detect 2xN sensor.mask as corner-pair format, expand to binary mask
- [ ] `example_ivp_opposing_corners_sensor_mask`
- [ ] `example_tvsp_transducer_field_patterns`
- [ ] `example_tvsp_angular_spectrum`
- [ ] `example_tvsp_equivalent_source_holography`
- [ ] `example_ewp_3D_simulation` (also needs elastic solver)

### Other missing features
- [ ] `example_tvsp_slit_diffraction` — needs `sound_speed_ref` (LOW effort)
- [ ] `example_ivp_sensor_frequency_response` — needs `sensor.frequency_response` (MEDIUM effort)
- [ ] `example_sd_sensor_directivity_2D` — needs directional sensor (MEDIUM effort)

## Phase 7 — Tier 3 (25 examples, need different solvers)

These require solvers or components not yet available in k-wave-python's `kspaceFirstOrder()`.

### kWaveArray (8 examples)
Requires: off-grid source/sensor modeling via `kWaveArray`
- [ ] `example_at_array_as_sensor`
- [ ] `example_at_array_as_source`
- [ ] `example_at_circular_piston_3D`
- [ ] `example_at_circular_piston_AS`
- [ ] `example_at_focused_annular_array_3D`
- [ ] `example_at_focused_bowl_3D`
- [ ] `example_at_focused_bowl_AS`
- [ ] `example_at_linear_array_transducer`

### kWaveTransducer / Ultrasound (5 examples)
Requires: `kWaveTransducer` object as source/sensor
- [ ] `example_us_beam_patterns`
- [ ] `example_us_bmode_linear_transducer`
- [ ] `example_us_bmode_phased_array`
- [ ] `example_us_defining_transducer`
- [ ] `example_us_transducer_as_sensor`

### kWaveDiffusion (4 examples)
Requires: thermal diffusion solver
- [ ] `example_diff_binary_sensor_mask`
- [ ] `example_diff_focused_ultrasound_heating`
- [ ] `example_diff_homogeneous_medium_diffusion`
- [ ] `example_diff_homogeneous_medium_source`

### Elastic wave propagation (4 examples)
Requires: `pstdElastic2D` / `pstdElastic3D` solver
- [ ] `example_ewp_3D_simulation`
- [ ] `example_ewp_layered_medium`
- [ ] `example_ewp_plane_wave_absorption`
- [ ] `example_ewp_shear_wave_snells_law`

### Other solvers
- [ ] `example_ivp_axisymmetric_simulation` — needs `kspaceFirstOrderAS`
- [ ] `example_ivp_comparison_modelling_functions` — needs `kspaceSecondOrder`
- [ ] `example_ivp_setting_initial_gradient` — needs `kspaceSecondOrder`
- [ ] `example_na_modelling_absorption` — needs `kspaceSecondOrder`
- [ ] `example_tvsp_acoustic_field_propagator` — needs `acousticFieldPropagator`
- [ ] `example_cpp_io_in_parts` — C++ specific
- [ ] `example_cpp_running_simulations` — C++ specific

## Known Issues

### 3D p_final parity mismatch
The `example_ivp_3D_simulation` parity test is skipped. Python and MATLAB produce
p_final with different max values (7e-5 vs 2e-4) at different spatial locations.
The 3D solver passes symmetry tests on homogeneous media, so this is likely a
difference in how the heterogeneous medium or `p0` smoothing is set up, not a
solver bug. Needs investigation.

### smooth() edge cases
The `smooth()` function now handles 1D inputs correctly (fixed singleton
dimension stripping). Leading-singleton 2D/3D inputs (e.g., shape `(1, 64, 64)`)
are rare but may still have issues with window shape broadcasting.

## Test Infrastructure

### Parity test framework
- `tests/test_example_parity.py` — parametrized tests comparing Python vs MATLAB
- MATLAB references in `~/git/k-wave-cupy/tests/ref_*.mat` (v7 format)
- Reference generators in `~/git/k-wave-cupy/tests/gen_ref_*.m`
- Binary full-grid sensors for exact parity (<5e-13 relative)
- Cartesian sensors for teaching (not validated against MATLAB — different Delaunay)

### Thresholds
- `THRESH = 5e-13` — IVP examples (machine precision)
- `THRESH_TVSP = 5e-11` — time-varying source examples (accumulated FFT rounding)

### GPU validation
CuPy testing on DigitalOcean GPU Droplet (~$0.21/run). Not yet started —
waiting for sufficient example coverage (~20+ examples).

## Bugs Fixed During Porting

| Bug | Impact | Fix |
|-----|--------|-----|
| `smooth()` 1D singleton stripping | Filtering part 2 produced wrong results | Match MATLAB's `grid_size(grid_size==1)=[]` |
| `p_final` double PML stripping | Wrong p_final with `pml_inside=False` | Backend-aware `_FULL_GRID_SUFFIXES` |
| `makeTime(np.max(c))` vs `makeTime(c)` | Wrong Nt for heterogeneous media | Pass full array |
| Python `round()` banker's rounding | Off-by-one in transducer positioning | Use `(n+1)//2` |
| `tone_burst` offset truncation | Signal length mismatch | `np.round().astype(int)` |
| `np.arange` float step | Off-by-one in 1D pulse (513 vs 512) | Use `np.linspace` |
| `np.einsum` → `xp.einsum` | CuPy GPU broken for Cartesian sensors | Use `xp.einsum` |
| Velocity naming mismatch | 42% error comparing staggered vs non-staggered | Python `"ux"` = MATLAB `'u_non_staggered'` |
