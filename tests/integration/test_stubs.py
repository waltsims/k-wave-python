"""Stub tests for examples not yet supported by the Python backend.

These document the gap and will show up in test reports.
Fill in when the Python backend gains support for each feature.
"""
import pytest

# -- Tier 2: reconstruction / time-reversal / 3D --


@pytest.mark.skip(reason="Reconstruction functions not yet wired through kspaceFirstOrder()")
@pytest.mark.integration
class TestReconstruction:
    def test_pr_2D_FFT_line_sensor(self):
        ...

    def test_pr_2D_TR_line_sensor(self):
        ...

    def test_pr_3D_FFT_planar_sensor(self):
        ...

    def test_pr_3D_TR_planar_sensor(self):
        ...


@pytest.mark.skip(reason="Sensor directivity not yet supported in Python backend")
@pytest.mark.integration
class TestSensorDirectivity:
    def test_sd_directivity_modelling_2D(self):
        ...

    def test_sd_focussed_detector_2D(self):
        ...

    def test_sd_focussed_detector_3D(self):
        ...


# -- Tier 3: kWaveArray / transducer --


@pytest.mark.skip(reason="kWaveArray not yet supported in Python backend")
@pytest.mark.integration
class TestArrayTransducer:
    def test_at_array_as_sensor(self):
        ...

    def test_at_array_as_source(self):
        ...

    def test_at_circular_piston_3D(self):
        ...

    def test_at_circular_piston_AS(self):
        ...

    def test_at_focused_annular_array_3D(self):
        ...

    def test_at_focused_bowl_3D(self):
        ...

    def test_at_focused_bowl_AS(self):
        ...

    def test_at_linear_array_transducer(self):
        ...


@pytest.mark.skip(reason="kWaveTransducer not yet supported in Python backend")
@pytest.mark.integration
class TestUltrasound:
    def test_us_beam_patterns(self):
        ...

    def test_us_bmode_linear_transducer(self):
        ...

    def test_us_bmode_phased_array(self):
        ...
