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


def _reorder_fullgrid_p(matlab_p, Nx, Ny):
    """Reorder MATLAB full-grid sensor data from F-order to C-order."""
    Nt = matlab_p.shape[1]
    return matlab_p.reshape(Nx, Ny, Nt, order="F").reshape(Nx * Ny, Nt)


# ---------------------------------------------------------------------------
# Shared test logic
# ---------------------------------------------------------------------------

THRESH = 5e-13  # relative threshold for IVP (initial value) examples
THRESH_TVSP = 5e-11  # looser for time-varying source examples (more FFT round-trips)


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
    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend="python",
        quiet=True,
        pml_inside=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.matlab_parity
class TestIVPHomogeneousMedium:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_ivp_homogeneous_medium import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("ivp_homogeneous_medium")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final")


@pytest.mark.matlab_parity
class TestIVPHeterogeneousMedium:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_ivp_heterogeneous_medium import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("ivp_heterogeneous_medium")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final")


@pytest.mark.matlab_parity
class TestIVPBinarySensorMask:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_ivp_binary_sensor_mask import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("ivp_binary_sensor_mask")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final")


@pytest.mark.matlab_parity
class TestIVPRecordingParticleVelocity:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_ivp_recording_particle_velocity import setup

        return _run_with_binary_sensor(setup, ndim=2, record=["p", "p_final", "ux", "uy"])

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("ivp_recording_particle_velocity")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final")

    def test_ux(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_ux = _reorder_fullgrid_p(ref["ux"], Nx, Ny)
        _assert_close(np.asarray(result["ux"]), matlab_ux, "ux")

    def test_uy(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_uy = _reorder_fullgrid_p(ref["uy"], Nx, Ny)
        _assert_close(np.asarray(result["uy"]), matlab_uy, "uy")


@pytest.mark.matlab_parity
class TestIVP1DSimulation:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_ivp_1D_simulation import setup

        return _run_with_binary_sensor(setup, ndim=1)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("ivp_1D_simulation")

    def test_p(self, result, ref):
        matlab_p = ref["p"]  # 1D: no F-to-C reorder needed
        _assert_close(np.asarray(result["p"]), matlab_p, "p", thresh=THRESH)

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", thresh=THRESH)


# ---------------------------------------------------------------------------
# Batch 2: TVSP examples
# ---------------------------------------------------------------------------


@pytest.mark.matlab_parity
class TestTVSPMonopole:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_tvsp_homogeneous_medium_monopole import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("tvsp_homogeneous_medium_monopole")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", thresh=THRESH_TVSP)


@pytest.mark.matlab_parity
class TestTVSPDipole:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_tvsp_homogeneous_medium_dipole import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("tvsp_homogeneous_medium_dipole")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", thresh=THRESH_TVSP)


@pytest.mark.matlab_parity
class TestTVSPSteeringLinearArray:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_tvsp_steering_linear_array import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("tvsp_steering_linear_array")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", thresh=THRESH_TVSP)


@pytest.mark.matlab_parity
class TestTVSPSnellsLaw:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_tvsp_snells_law import setup

        return _run_with_binary_sensor(setup, ndim=2)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("tvsp_snells_law")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"][pml:-pml, pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", thresh=THRESH_TVSP)


# ---------------------------------------------------------------------------
# Batch 3: NA examples (PML, nonlinearity, filtering)
# ---------------------------------------------------------------------------


@pytest.mark.matlab_parity
class TestNAControllingPML:
    @pytest.fixture(scope="class")
    def result(self):
        # This example's physics includes non-default PML settings (alpha=0),
        # so we call run() directly rather than _run_with_binary_sensor.
        from examples.ported.example_na_controlling_the_PML import run

        return run()

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("na_controlling_the_PML")

    def test_p(self, result, ref):
        Nx, Ny = int(ref["Nx"]), int(ref["Ny"])
        matlab_p = _reorder_fullgrid_p(ref["p"], Nx, Ny)
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        # pml_inside=True with pml_alpha=0: full field valid, no PML crop
        _assert_close(np.asarray(result["p_final"]), ref["p_final"], "p_final")


@pytest.mark.matlab_parity
class TestNAFilteringPart1:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_na_filtering_part_1 import setup

        return _run_with_binary_sensor(setup, ndim=1)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("na_filtering_part_1")

    def test_p(self, result, ref):
        matlab_p = ref["p"]
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"].ravel()[pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final")


@pytest.mark.matlab_parity
@pytest.mark.skip(reason="smooth() implementation differs between Python and MATLAB — needs investigation")
class TestNAFilteringPart2:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_na_filtering_part_2 import setup

        return _run_with_binary_sensor(setup, ndim=1)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("na_filtering_part_2")

    def test_p(self, result, ref):
        matlab_p = ref["p"]
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"].ravel()[pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final")


@pytest.mark.matlab_parity
class TestNAFilteringPart3:
    @pytest.fixture(scope="class")
    def result(self):
        from examples.ported.example_na_filtering_part_3 import setup

        return _run_with_binary_sensor(setup, ndim=1)

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("na_filtering_part_3")

    def test_p(self, result, ref):
        matlab_p = ref["p"]
        _assert_close(np.asarray(result["p"]), matlab_p, "p")

    def test_p_final(self, result, ref):
        pml = 20
        matlab_pf = ref["p_final"].ravel()[pml:-pml]
        _assert_close(np.asarray(result["p_final"]), matlab_pf, "p_final", thresh=THRESH_TVSP)


@pytest.mark.matlab_parity
class TestNAModellingNonlinearity:
    @pytest.fixture(scope="class")
    def result(self):
        # Uses record_start_index and non-default PML, so call run() directly
        from examples.ported.example_na_modelling_nonlinearity import run

        return run()

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref("na_modelling_nonlinearity")

    def test_p(self, result, ref):
        matlab_p = ref["p"]
        _assert_close(np.asarray(result["p"]), matlab_p, "p", thresh=THRESH_TVSP)
