"""
Test suite for time reversal reconstruction.
"""

import time

import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options import SimulationExecutionOptions, SimulationOptions
from kwave.reconstruction import TimeReversal


def test_valid_initialization():
    """Test valid initialization of TimeReversal class."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)  # Set explicit time array
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))  # Set sensor mask

    # Test valid initialization
    tr = TimeReversal(kgrid, medium, sensor)
    assert tr.compensation_factor == 2.0


def test_missing_sensor_mask():
    """Test initialization with missing sensor mask."""
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor_without_mask = kSensor()

    with pytest.raises(ValueError, match="Sensor mask must be set for time reversal"):
        TimeReversal(kgrid, medium, sensor_without_mask)


def test_auto_time_array():
    """Test initialization with auto time array."""
    kgrid_auto = kWaveGrid([100, 100], [0.1, 0.1])  # This creates a grid with auto time array
    medium = kWaveMedium(sound_speed=1500)
    sensor_auto = kSensor(mask=np.ones((100, 100), dtype=bool))

    with pytest.raises(ValueError, match="t_array must be explicitly set for time reversal"):
        TimeReversal(kgrid_auto, medium, sensor_auto)


def test_negative_compensation_factor():
    """Test initialization with negative compensation factor."""
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    with pytest.raises(ValueError, match="compensation_factor must be positive"):
        TimeReversal(kgrid, medium, sensor, compensation_factor=-1.0)


def test_zero_compensation_factor():
    """Test initialization with zero compensation factor."""
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    with pytest.raises(ValueError, match="compensation_factor must be positive"):
        TimeReversal(kgrid, medium, sensor, compensation_factor=0.0)


def test_empty_sensor_mask():
    """Test initialization with empty sensor mask."""
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    empty_sensor = kSensor(mask=np.zeros((100, 100), dtype=bool))

    with pytest.raises(ValueError, match="Sensor mask must have at least one active point"):
        TimeReversal(kgrid, medium, empty_sensor)


def test_no_active_points():
    """Test initialization with sensor mask having no active points."""
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    no_active_sensor = kSensor(mask=np.zeros((100, 100), dtype=bool))
    no_active_sensor.mask[0, 0] = False  # Ensure no active points

    with pytest.raises(ValueError, match="Sensor mask must have at least one active point"):
        TimeReversal(kgrid, medium, no_active_sensor)


def test_wrong_sensor_mask_shape():
    """Test initialization with wrong sensor mask shape."""
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    wrong_shape_sensor = kSensor(mask=np.ones((50, 50), dtype=bool))

    with pytest.raises(ValueError, match="Sensor mask shape \\(50, 50\\) does not match grid dimensions \\[100 100\\]"):
        TimeReversal(kgrid, medium, wrong_shape_sensor)


def test_time_reversal_call():
    """Test the __call__ method of TimeReversal class."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)  # Set explicit time array
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    # Create simulation options
    simulation_options = SimulationOptions()
    execution_options = SimulationExecutionOptions()

    # Create time reversal handler
    tr = TimeReversal(kgrid, medium, sensor)

    # Mock simulation function
    def mock_simulation(*args, **kwargs):
        return {"p_final": np.random.rand(100, 100)}

    # Set up recorded pressure data
    sensor.recorded_pressure = np.random.rand(100, 100)  # Mock recorded pressure data

    # Test reconstruction
    p0_recon = tr(mock_simulation, simulation_options, execution_options)
    assert p0_recon.shape == (100, 100)
    assert np.all(p0_recon >= 0)  # Check positivity condition
    assert np.all(p0_recon <= 2.0)  # Check compensation factor


def test_time_reversal_workflow():
    """Test complete time reversal workflow."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    # Create simulation options
    simulation_options = SimulationOptions()
    execution_options = SimulationExecutionOptions()

    # Mock forward simulation
    def mock_forward_simulation(*args, **kwargs):
        return {"p": np.random.rand(10, 100)}

    # Mock reconstruction simulation
    def mock_reconstruction_simulation(*args, **kwargs):
        return {"p_final": np.random.rand(100, 100)}

    # Run forward simulation
    sensor_data = mock_forward_simulation(kgrid, None, sensor, medium, simulation_options, execution_options)
    sensor.recorded_pressure = sensor_data["p"]  # Set recorded pressure data

    # Run reconstruction
    tr = TimeReversal(kgrid, medium, sensor)
    p0_recon = tr(mock_reconstruction_simulation, simulation_options, execution_options)

    # Verify reconstruction
    assert p0_recon.shape == (100, 100)
    assert np.all(p0_recon >= 0)  # Check positivity condition
    assert np.all(p0_recon <= 2.0)  # Check compensation factor


def test_time_reversal_with_different_operators():
    """Test time reversal with different simulation operators."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    # Create simulation options
    simulation_options = SimulationOptions()
    execution_options = SimulationExecutionOptions()

    # Mock 3D simulation
    def mock_3d_simulation(*args, **kwargs):
        return {"p_final": np.random.rand(100, 100, 100)}

    # Mock 2D simulation
    def mock_2d_simulation(*args, **kwargs):
        return {"p_final": np.random.rand(100, 100)}

    # Create time reversal handler
    tr = TimeReversal(kgrid, medium, sensor)

    # Set up recorded pressure data
    sensor.recorded_pressure = np.random.rand(100, 100)  # Mock recorded pressure data

    # Test with different operators
    p0_recon_3d = tr(mock_3d_simulation, simulation_options, execution_options)
    p0_recon_2d = tr(mock_2d_simulation, simulation_options, execution_options)

    # Verify reconstructions
    assert p0_recon_3d.shape == (100, 100, 100)
    assert p0_recon_2d.shape == (100, 100)
    assert np.all(p0_recon_3d >= 0)  # Check positivity condition
    assert np.all(p0_recon_2d >= 0)  # Check positivity condition
    assert np.all(p0_recon_3d <= 2.0)  # Check compensation factor
    assert np.all(p0_recon_2d <= 2.0)  # Check compensation factor


def test_time_reversal_error_handling():
    """Test error handling in time reversal."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    # Create simulation options
    simulation_options = SimulationOptions(save_to_disk=True)
    execution_options = SimulationExecutionOptions()

    # Create time reversal handler
    tr = TimeReversal(kgrid, medium, sensor)

    # Test invalid simulation operator
    def invalid_operator(*args, **kwargs):
        raise ValueError("Invalid operator")

    with pytest.raises(ValueError):
        tr(invalid_operator, simulation_options, execution_options)

    # Test invalid simulation options
    with pytest.raises(ValueError, match="simulation_options must be provided"):
        tr(kspaceFirstOrder2D, None, execution_options)

    # Test invalid execution options
    with pytest.raises(ValueError, match="execution_options must be provided"):
        tr(kspaceFirstOrder2D, simulation_options, None)


def test_deprecation_warnings():
    """Test deprecation warnings for old time reversal methods."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    # Test property deprecation
    with pytest.warns(DeprecationWarning, match="Use TimeReversal class instead"):
        # Test getter
        _ = sensor.time_reversal_boundary_data

        # Test setter
        sensor.time_reversal_boundary_data = np.random.rand(10, 100)


def test_time_reversal_performance():
    """Test performance of time reversal reconstruction."""
    # Create test data
    kgrid = kWaveGrid([100, 100], [0.1, 0.1])
    kgrid.setTime(100, 1e-6)
    medium = kWaveMedium(sound_speed=1500)
    sensor = kSensor(mask=np.ones((100, 100), dtype=bool))

    # Create simulation options
    simulation_options = SimulationOptions()
    execution_options = SimulationExecutionOptions()

    # Create time reversal handler
    tr = TimeReversal(kgrid, medium, sensor)

    # Mock simulation function
    def mock_simulation(*args, **kwargs):
        return {"p_final": np.random.rand(100, 100)}

    # Set up recorded pressure data
    sensor.recorded_pressure = np.random.rand(100, 100)  # Mock recorded pressure data

    # Measure reconstruction time
    start_time = time.time()
    p0_recon = tr(mock_simulation, simulation_options, execution_options)
    duration = time.time() - start_time

    # Verify reconstruction
    assert p0_recon.shape == (100, 100)
    assert np.all(p0_recon >= 0)  # Check positivity condition
    assert np.all(p0_recon <= 2.0)  # Check compensation factor

    # Check performance (should complete within 1 second for this small grid)
    assert duration < 1.0, f"Reconstruction took {duration:.2f} seconds, expected < 1.0 seconds"
