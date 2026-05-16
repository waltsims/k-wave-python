"""Tests for kWaveMedium alpha_mode normalization and validation."""
import numpy as np
import pytest

from kwave.enums import AlphaMode
from kwave.kmedium import kWaveMedium


class TestAlphaModeNormalization:
    def test_default_is_none(self):
        m = kWaveMedium(sound_speed=1500)
        assert m.alpha_mode is None

    def test_enum_passes_through(self):
        m = kWaveMedium(sound_speed=1500, alpha_mode=AlphaMode.NO_DISPERSION)
        assert m.alpha_mode is AlphaMode.NO_DISPERSION

    @pytest.mark.parametrize("value", ["no_absorption", "no_dispersion", "stokes"])
    def test_valid_string_normalized_at_construction(self, value):
        m = kWaveMedium(sound_speed=1500, alpha_mode=value)
        assert isinstance(m.alpha_mode, AlphaMode)
        assert m.alpha_mode == value

    def test_invalid_string_at_construction_raises_friendly_error(self):
        with pytest.raises(ValueError, match="must be an AlphaMode"):
            kWaveMedium(sound_speed=1500, alpha_mode="garbage")

    def test_post_construction_string_assignment_accepted_by_check_fields(self):
        m = kWaveMedium(sound_speed=1500, alpha_coeff=np.array(0.5), alpha_power=1.5)
        m.alpha_mode = "no_dispersion"  # plain string per type hint
        m.check_fields(np.array([64, 64]))
        # check_fields normalizes for downstream consumers
        assert isinstance(m.alpha_mode, AlphaMode)
        assert m.alpha_mode == "no_dispersion"

    def test_post_construction_invalid_string_rejected_by_check_fields(self):
        m = kWaveMedium(sound_speed=1500, alpha_coeff=np.array(0.5), alpha_power=1.5)
        m.alpha_mode = "garbage"
        with pytest.raises(ValueError, match="must be an AlphaMode"):
            m.check_fields(np.array([64, 64]))

    def test_string_comparison_still_works(self):
        # AlphaMode inherits from str, so == against raw strings must keep working
        m = kWaveMedium(sound_speed=1500, alpha_mode="no_dispersion")
        assert m.alpha_mode == "no_dispersion"
        assert m.alpha_mode in ["no_absorption", "no_dispersion"]
