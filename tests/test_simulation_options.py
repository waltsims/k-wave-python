from kwave.options.simulation_execution_options import SimulationExecutionOptions
import pytest


def test_gpu_validation():
    # mock that cuda is not available
    from kwave.options.simulation_execution_options import cuda

    cuda.is_available = lambda: False
    # Test that the GPU is available
    with pytest.raises(ValueError):
        _ = SimulationExecutionOptions(is_gpu_simulation=True)

    from kwave.options.simulation_execution_options import cuda

    cuda.is_available = lambda: True
    # Test when cuda is available
    _ = SimulationExecutionOptions(is_gpu_simulation=True)
