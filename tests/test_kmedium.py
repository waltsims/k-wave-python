"""
kWaveMedium comprehensive test suite

Tests for code paths not covered by existing tests to improve code coverage.
"""

from unittest.mock import patch

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium


def test_elastic_properties_access():
    """Test access to elastic code related properties (should raise NotImplementedError)"""
    medium = kWaveMedium(sound_speed=1500)

    with pytest.raises(NotImplementedError, match="Elastic simulation"):
        _ = medium.sound_speed_shear

    with pytest.raises(NotImplementedError, match="Elastic simulation"):
        _ = medium.sound_speed_ref_shear

    with pytest.raises(NotImplementedError, match="Elastic simulation"):
        _ = medium.sound_speed_compression

    with pytest.raises(NotImplementedError, match="Elastic simulation"):
        _ = medium.sound_speed_ref_compression

    with pytest.raises(NotImplementedError, match="Elastic simulation"):
        _ = medium.alpha_coeff_compression

    with pytest.raises(NotImplementedError, match="Elastic simulation"):
        _ = medium.alpha_coeff_shear


def test_is_defined_method():
    """Test is_defined method with various scenarios"""
    medium = kWaveMedium(
        sound_speed=1500,
        density=1000,
        alpha_coeff=0.75
    )

    # Test single field
    assert medium.is_defined('sound_speed') == [True]
    assert medium.is_defined('density') == [True]
    assert medium.is_defined('alpha_coeff') == [True]
    assert medium.is_defined('alpha_power') == [False]
    assert medium.is_defined('BonA') == [False]

    # Test multiple fields
    result = medium.is_defined('sound_speed', 'density', 'alpha_power', 'BonA')
    assert result == [True, True, False, False]


def test_ensure_defined_method():
    """Test ensure_defined method"""
    medium = kWaveMedium(sound_speed=1500, density=1000)

    # Test defined fields
    medium.ensure_defined('sound_speed', 'density')  # Should not raise exception

    # Test undefined fields
    with pytest.raises(AssertionError, match="alpha_coeff must not be None"):
        medium.ensure_defined('alpha_coeff')

    with pytest.raises(AssertionError, match="alpha_power must not be None"):
        medium.ensure_defined('alpha_power')


def test_is_nonlinear_method():
    """Test is_nonlinear method"""
    # Linear medium
    medium1 = kWaveMedium(sound_speed=1500)
    assert not medium1.is_nonlinear()

    # Nonlinear medium
    medium2 = kWaveMedium(sound_speed=1500, BonA=6.0)
    assert medium2.is_nonlinear()


def test_stokes_mode_alpha_power_none():
    """Test Stokes mode when alpha_power is None"""
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75)

    # When alpha_power is None, setting Stokes mode should set it to 2
    medium.set_absorbing(is_absorbing=True, is_stokes=True)
    assert medium.alpha_power == 2


def test_stokes_mode_alpha_power_array():
    """Test Stokes mode when alpha_power is an array"""
    # Test multi-element array
    medium1 = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=np.array([1.5, 1.8]))

    with patch('logging.warning') as mock_warning:
        medium1.set_absorbing(is_absorbing=True, is_stokes=True)
        mock_warning.assert_called_once()
        assert "alpha_power = 2" in mock_warning.call_args[0][0]

    assert medium1.alpha_power == 2


def test_absorbing_without_stokes_alpha_power_validation():
    """Test alpha_power validation in non-Stokes absorbing mode"""
    # Test alpha_power must be scalar
    medium1 = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=np.array([1.5, 1.8]))

    with pytest.raises(AssertionError, match="must be scalar"):
        medium1.set_absorbing(is_absorbing=True, is_stokes=False)

    # Test alpha_power must be in range 0-3
    medium2 = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=-0.5)

    with pytest.raises(AssertionError, match="between 0 and 3"):
        medium2.set_absorbing(is_absorbing=True, is_stokes=False)

    medium3 = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=3.5)

    with pytest.raises(AssertionError, match="between 0 and 3"):
        medium3.set_absorbing(is_absorbing=True, is_stokes=False)


def test_alpha_mode_validation_edge_cases():
    """Test alpha_mode validation edge cases"""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))

    # Test None value (should pass)
    medium1 = kWaveMedium(sound_speed=1500, alpha_mode=None)
    medium1.check_fields(kgrid.N)  # Should not raise exception

    # Test empty string (should fail)
    medium2 = kWaveMedium(sound_speed=1500, alpha_mode="")
    with pytest.raises(AssertionError):
        medium2.check_fields(kgrid.N)


def test_alpha_filter_none():
    """Test when alpha_filter is None"""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))

    medium = kWaveMedium(sound_speed=1500, alpha_filter=None)
    medium.check_fields(kgrid.N)  # Should not raise exception


def test_alpha_sign_none():
    """Test when alpha_sign is None"""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))

    medium = kWaveMedium(sound_speed=1500, alpha_sign=None)
    medium.check_fields(kgrid.N)  # Should not raise exception

def test_alpha_sign_wrong_size_raises():
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    medium = kWaveMedium(sound_speed=1500, alpha_sign=np.array([1.0]))
    with pytest.raises(ValueError, match="2 element numeric array"):
        medium.check_fields(kgrid.N)

def test_alpha_sign_non_numeric_raises():
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    medium = kWaveMedium(sound_speed=1500, alpha_sign=np.array(["a", "b"], dtype=object))
    with pytest.raises(ValueError, match="2 element numeric array"):
        medium.check_fields(kgrid.N)

def test_alpha_coeff_none():
    """Test when alpha_coeff is None"""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))

    medium = kWaveMedium(sound_speed=1500, alpha_coeff=None)
    medium.check_fields(kgrid.N)  # Should not raise exception


def test_alpha_coeff_array_validation():
    """Test alpha_coeff array validation"""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))

    # Valid array
    medium1 = kWaveMedium(sound_speed=1500, alpha_coeff=np.array([0.5, 0.6, 0.7]))
    medium1.check_fields(kgrid.N)  # Should not raise exception

    # Array with negative values
    medium2 = kWaveMedium(sound_speed=1500, alpha_coeff=np.array([0.5, -0.1, 0.7]))
    with pytest.raises(ValueError, match="non-negative and real"):
        medium2.check_fields(kgrid.N)

    # Array with complex values
    medium3 = kWaveMedium(sound_speed=1500, alpha_coeff=np.array([0.5, 0.6 + 0.1j, 0.7]))
    with pytest.raises(ValueError, match="non-negative and real"):
        medium3.check_fields(kgrid.N)


def test_post_init_sound_speed_conversion():
    """Test sound_speed conversion in __post_init__"""
    # Scalar input
    medium1 = kWaveMedium(sound_speed=1500)
    assert isinstance(medium1.sound_speed, np.ndarray)
    assert medium1.sound_speed.shape == (1,)

    # Array input
    medium2 = kWaveMedium(sound_speed=np.array([1500, 1600]))
    assert isinstance(medium2.sound_speed, np.ndarray)
    assert medium2.sound_speed.shape == (2,)


def test_stokes_mode_alpha_mode_restrictions():
    """Test alpha_mode restrictions in Stokes mode"""
    # Test no_absorption mode
    medium1 = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_mode='no_absorption')
    with pytest.raises(NotImplementedError, match="not supported with the axisymmetric code"):
        medium1.set_absorbing(is_absorbing=True, is_stokes=True)

    # Test no_dispersion mode
    medium2 = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_mode='no_dispersion')
    with pytest.raises(NotImplementedError, match="not supported with the axisymmetric code"):
        medium2.set_absorbing(is_absorbing=True, is_stokes=True)


def test_absorbing_flags():
    """Test setting of absorbing and stokes flags"""
    medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

    # Initial state
    assert not medium.absorbing
    assert not medium.stokes

    # Set to non-Stokes absorbing
    medium.set_absorbing(is_absorbing=True, is_stokes=False)
    assert medium.absorbing
    assert not medium.stokes

    # Reset
    medium.absorbing = False
    medium.stokes = False

    # Set to Stokes absorbing
    medium.set_absorbing(is_absorbing=True, is_stokes=True)
    assert medium.absorbing
    assert medium.stokes

    # Set to non-absorbing
    medium.set_absorbing(is_absorbing=False, is_stokes=False)
    assert not medium.absorbing
    assert not medium.stokes
