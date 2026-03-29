"""MATLAB parity tests for ported examples.

Each test imports an example's ``setup()`` function (which returns the physics:
kgrid, medium, source), adds a **full-grid binary sensor**, and compares against
MATLAB reference data generated with the same binary sensor.

Binary sensors give machine-precision parity (<1e-13 relative).  The examples'
own ``run()`` functions may use Cartesian or sparse sensors for teaching — those
are not tested here.

MATLAB references are per-example v7 .mat files in the sibling k-wave-cupy repo.
Regenerate with: ``arch -arm64 matlab -batch "run('tests/gen_ref_batch1.m')"``
"""
import importlib
import os

import numpy as np
import pytest
import scipy.io as sio

from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder

# ---------------------------------------------------------------------------
# Reference data location
# ---------------------------------------------------------------------------

_REF_DIR = os.environ.get(
    "KWAVE_MATLAB_REF_DIR",
    os.path.expanduser("~/git/k-wave-cupy/tests"),
)


def _load_ref(name):
    """Load a per-example .mat reference file, or skip if missing."""
    path = os.path.join(_REF_DIR, f"ref_{name}.mat")
    if not os.path.exists(path):
        pytest.skip(f"MATLAB reference not found: {path}")
    return sio.loadmat(path, squeeze_me=True)


# ---------------------------------------------------------------------------
# F-to-C reorder helpers
# ---------------------------------------------------------------------------


def _reorder_fullgrid(matlab_arr, grid_shape):
    """Reorder MATLAB full-grid sensor data from F-order to C-order."""
    Nt = matlab_arr.shape[1]
    n_pts = int(np.prod(grid_shape))
    return matlab_arr.reshape(*grid_shape, Nt, order="F").reshape(n_pts, Nt)


def _grid_shape_from_ref(ref, ndim):
    """Extract grid shape from reference data, inferring from p_final if needed."""
    if ndim == 1:
        return (int(ref["Nx"]),) if "Nx" in ref else ref["p_final"].shape[:1]
    if ndim == 2:
        if "Nx" in ref and "Ny" in ref:
            return (int(ref["Nx"]), int(ref["Ny"]))
        return ref["p_final"].shape[:2]
    if "Nx" in ref and "Ny" in ref and "Nz" in ref:
        return (int(ref["Nx"]), int(ref["Ny"]), int(ref["Nz"]))
    return ref["p_final"].shape[:3]


# ---------------------------------------------------------------------------
# Shared test logic
# ---------------------------------------------------------------------------

THRESH = 5e-13  # relative threshold for IVP (initial value) examples
THRESH_TVSP = 5e-11  # looser for time-varying source examples (more FFT round-trips)
# MATLAB default PML sizes: 20 for 1D/2D, 10 for 3D (see kspaceFirstOrder_setDefaults.m)
PML_SIZE = {1: 20, 2: 20, 3: 10}


def _assert_close(python, matlab, label="", thresh=None):
    """Assert machine-precision parity between Python and MATLAB arrays."""
    if thresh is None:
        thresh = THRESH
    assert python.shape == matlab.shape, f"{label} shape mismatch: Python {python.shape} vs MATLAB {matlab.shape}"
    peak = np.max(np.abs(matlab))
    if peak < 1e-20:
        return  # trivially zero
    rel = np.max(np.abs(python - matlab)) / peak
    assert rel < thresh, f"{label} relative error {rel:.2e} exceeds {thresh}"


def _run_with_binary_sensor(setup_fn, ndim, record=None):
    """Run an example's setup() with a full-grid binary sensor."""
    kgrid, medium, source = setup_fn()
    if ndim == 1:
        grid_shape = kgrid.N[0] if hasattr(kgrid.N, "__len__") else int(kgrid.N)
        mask = np.ones(grid_shape, dtype=bool)
    else:
        mask = np.ones(tuple(int(n) for n in kgrid.N), dtype=bool)
    sensor = kSensor(mask=mask)
    sensor.record = record or ["p", "p_final"]
    pml_size = PML_SIZE[ndim]
    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend="python",
        quiet=True,
        pml_inside=True,
        pml_size=pml_size,
    )


def _pml_crop(ndim):
    """PML crop slices for p_final."""
    pml = PML_SIZE[ndim]
    return tuple(slice(pml, -pml) for _ in range(ndim))


# ---------------------------------------------------------------------------
# Example registry
# ---------------------------------------------------------------------------

# (name, ndim, p_thresh, p_final_thresh)
_EXAMPLES = [
    # IVP — machine precision for both p and p_final
    ("ivp_homogeneous_medium", 2, THRESH, THRESH),
    ("ivp_heterogeneous_medium", 2, THRESH, THRESH),
    ("ivp_binary_sensor_mask", 2, THRESH, THRESH),
    ("ivp_1D_simulation", 1, THRESH, THRESH),
    ("ivp_loading_external_image", 2, THRESH, THRESH),
    ("ivp_photoacoustic_waveforms", 2, THRESH, THRESH),
    # TVSP — p is machine precision, p_final is looser (more FFT round-trips)
    ("tvsp_homogeneous_medium_monopole", 2, THRESH, THRESH_TVSP),
    ("tvsp_homogeneous_medium_dipole", 2, THRESH, THRESH_TVSP),
    ("tvsp_steering_linear_array", 2, THRESH, THRESH_TVSP),
    ("tvsp_snells_law", 2, THRESH, THRESH_TVSP),
    ("tvsp_doppler_effect", 2, THRESH_TVSP, THRESH_TVSP),
    # NA (filtering) — p is machine precision, p_final looser for parts 2/3
    ("na_filtering_part_1", 1, THRESH, THRESH),
    ("na_filtering_part_2", 1, THRESH, THRESH_TVSP),
    ("na_filtering_part_3", 1, THRESH, THRESH_TVSP),
    # 3D — p_final only (full p recording would be ~880 MB)
    ("ivp_3D_simulation", 3, None, THRESH_TVSP),
    ("tvsp_3D_simulation", 3, None, THRESH_TVSP),
]


# ---------------------------------------------------------------------------
# Standard parity tests: p + p_final
# ---------------------------------------------------------------------------


@pytest.mark.matlab_parity
class TestStandardExamples:
    """Table-driven parity tests for examples using setup() + binary sensor."""

    @pytest.fixture(scope="class", params=_EXAMPLES, ids=[e[0] for e in _EXAMPLES])
    def scenario(self, request):
        name, ndim, p_thresh, pf_thresh = request.param
        mod = importlib.import_module(f"examples.{name}")
        # 3D examples only record p_final (full p would be ~880 MB)
        record = ["p_final"] if p_thresh is None else ["p", "p_final"]
        try:
            result = _run_with_binary_sensor(mod.setup, ndim, record)
        except FileNotFoundError:
            pytest.skip(f"Asset not found for {name}")
        ref = _load_ref(name)
        return result, ref, ndim, p_thresh, pf_thresh

    def test_p(self, scenario):
        result, ref, ndim, p_thresh, _pf_thresh = scenario
        if p_thresh is None:
            pytest.skip("p not recorded (3D example)")
        matlab_p = ref["p"]
        if ndim >= 2:
            matlab_p = _reorder_fullgrid(matlab_p, _grid_shape_from_ref(ref, ndim))
        _assert_close(np.asarray(result["p"]), matlab_p, "p", p_thresh)

    def test_p_final(self, scenario):
        result, ref, ndim, _p_thresh, pf_thresh = scenario
        matlab_pf = ref["p_final"]
        if ndim == 1:
            matlab_pf = matlab_pf.ravel()
        matlab_pf = matlab_pf[_pml_crop(ndim)]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", pf_thresh)


# ---------------------------------------------------------------------------
# Velocity parity test (records ux, uy in addition to p, p_final)
# ---------------------------------------------------------------------------


@pytest.mark.matlab_parity
class TestIVPRecordingParticleVelocity:
    @pytest.fixture(scope="class")
    def scenario(self):
        from examples.ivp_recording_particle_velocity import setup

        result = _run_with_binary_sensor(setup, ndim=2, record=["p", "p_final", "ux", "uy"])
        ref = _load_ref("ivp_recording_particle_velocity")
        shape = _grid_shape_from_ref(ref, 2)
        return result, ref, shape

    def test_p(self, scenario):
        result, ref, shape = scenario
        _assert_close(np.asarray(result["p"]), _reorder_fullgrid(ref["p"], shape), "p")

    def test_p_final(self, scenario):
        result, ref, shape = scenario
        _assert_close(np.asarray(result["p_final"]), ref["p_final"][_pml_crop(2)], "p_final")

    def test_ux(self, scenario):
        result, ref, shape = scenario
        _assert_close(np.asarray(result["ux"]), _reorder_fullgrid(ref["ux"], shape), "ux")

    def test_uy(self, scenario):
        result, ref, shape = scenario
        _assert_close(np.asarray(result["uy"]), _reorder_fullgrid(ref["uy"], shape), "uy")


# ---------------------------------------------------------------------------
# Custom-run examples (call run() directly, non-standard PML/record)
# ---------------------------------------------------------------------------


@pytest.mark.matlab_parity
class TestNAControllingPML:
    @pytest.fixture(scope="class")
    def scenario(self):
        from examples.na_controlling_the_PML import run

        ref = _load_ref("na_controlling_the_PML")
        return run(), ref, _grid_shape_from_ref(ref, 2)

    def test_p(self, scenario):
        result, ref, shape = scenario
        _assert_close(np.asarray(result["p"]), _reorder_fullgrid(ref["p"], shape), "p")

    def test_p_final(self, scenario):
        result, ref, shape = scenario
        # pml_alpha=0 means no absorption at boundary — full field valid, no PML crop
        _assert_close(np.asarray(result["p_final"]), ref["p_final"], "p_final")


@pytest.mark.matlab_parity
class TestNAModellingNonlinearity:
    @pytest.fixture(scope="class")
    def scenario(self):
        from examples.na_modelling_nonlinearity import run

        return run(), _load_ref("na_modelling_nonlinearity")

    def test_p(self, scenario):
        result, ref = scenario
        _assert_close(np.asarray(result["p"]), ref["p"], "p", thresh=THRESH_TVSP)


# ---------------------------------------------------------------------------
# TODO: Parity tests still needed for these ported examples:
#
# - ivp_saving_movie_files (standard 2D IVP — straightforward)
# - na_optimising_performance (uses Cartesian sensor — needs custom ref)
# - na_source_smoothing (uses kspaceFirstOrder not kspaceSecondOrder)
# - pr_2D_FFT_line_sensor (forward sim with pml_inside=False)
# - pr_3D_FFT_planar_sensor (forward sim, pml_inside deviation)
# - sd_directional_array_elements (multi-element averaging — custom harness)
# - sd_directivity_modelling_2D (11 sims — custom harness)
# - sd_directivity_modelling_3D (11 sims on 64^3 — slow, custom harness)
#
# The SD directivity examples need a different test pattern: call run()
# directly (not _run_with_binary_sensor) and compare element-level output.
# ---------------------------------------------------------------------------
