"""Tests for backward-compatibility bridge."""
import pytest

from kwave.compat import options_to_kwargs
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


class TestSimulationOptions:
    def test_default_options(self):
        opts = SimulationOptions()
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["use_sg"] is True
        assert kwargs["use_kspace"] is True
        assert kwargs["smooth_p0"] is True

    def test_pml_auto(self):
        opts = SimulationOptions(pml_auto=True, pml_inside=False)
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["pml_size"] == "auto"

    def test_pml_per_axis(self):
        opts = SimulationOptions()
        opts.pml_x_size = 10
        opts.pml_y_size = 20
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["pml_size"] == (10, 20)

    def test_pml_alpha(self):
        opts = SimulationOptions(pml_alpha=1.5)
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["pml_alpha"] == 1.5

    def test_save_only(self):
        opts = SimulationOptions(save_to_disk_exit=True)
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["save_only"] is True


class TestExecutionOptions:
    def test_python_backend(self):
        opts = SimulationExecutionOptions(backend="python")
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["backend"] == "python"
        assert kwargs["device"] == "cpu"

    def test_cuda_backend(self):
        opts = SimulationExecutionOptions(is_gpu_simulation=True)
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["backend"] == "cpp"
        assert kwargs["device"] == "gpu"

    def test_omp_backend(self):
        opts = SimulationExecutionOptions(is_gpu_simulation=False)
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["backend"] == "cpp"
        assert kwargs["device"] == "cpu"

    def test_quiet_mode(self):
        opts = SimulationExecutionOptions(verbose_level=0, show_sim_log=False)
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["quiet"] is True

    def test_debug_mode(self):
        opts = SimulationExecutionOptions(verbose_level=2)
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["debug"] is True

    def test_device_num(self):
        opts = SimulationExecutionOptions(device_num=1, is_gpu_simulation=True)
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["device_num"] == 1

    def test_binary_path(self):
        opts = SimulationExecutionOptions(is_gpu_simulation=False, backend="OMP", binary_path="./kspaceFirstOrder-OMP")
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs.get("binary_path") == "./kspaceFirstOrder-OMP"


class TestCombined:
    def test_both_options(self):
        sim_opts = SimulationOptions(pml_auto=True, pml_inside=False, smooth_p0=False)
        exec_opts = SimulationExecutionOptions(is_gpu_simulation=True, verbose_level=2)
        kwargs = options_to_kwargs(sim_opts, exec_opts)
        assert kwargs["pml_size"] == "auto"
        assert kwargs["smooth_p0"] is False
        assert kwargs["backend"] == "cpp"
        assert kwargs["device"] == "gpu"
        assert kwargs["debug"] is True

    def test_none_options(self):
        kwargs = options_to_kwargs()
        assert kwargs == {}
