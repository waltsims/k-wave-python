import sys
from pathlib import Path

import kwave.options.simulation_options as simulation_options_module
from kwave.options.simulation_options import SimulationOptions, resolve_filenames_for_run


def test_dated_filenames_are_enabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KWAVE_USE_DATED_FILENAMES", raising=False)
    options = SimulationOptions(save_to_disk=True, data_path=tmp_path)

    assert Path(options.input_filename).name.endswith("_kwave_input.h5")
    assert Path(options.output_filename).name.endswith("_kwave_output.h5")
    assert Path(options.input_filename).name != "kwave_input.h5"
    assert Path(options.output_filename).name != "kwave_output.h5"


def test_dated_filenames_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KWAVE_USE_DATED_FILENAMES", "0")
    options = SimulationOptions(save_to_disk=True, data_path=tmp_path)

    assert Path(options.input_filename).name == "kwave_input.h5"
    assert Path(options.output_filename).name == "kwave_output.h5"


def test_output_filename_is_derived_from_explicit_input_name(tmp_path):
    options = SimulationOptions(save_to_disk=True, data_path=tmp_path, input_filename="example_input_7.h5")

    assert Path(options.input_filename).name == "example_input_7.h5"
    assert Path(options.output_filename).name == "example_output_7.h5"


def test_output_filename_derivation_appends_output_when_input_token_missing(tmp_path):
    options = SimulationOptions(save_to_disk=True, data_path=tmp_path, input_filename="sample_case.h5")

    assert Path(options.input_filename).name == "sample_case.h5"
    assert Path(options.output_filename).name == "sample_case_output.h5"


def test_default_data_path_for_examples(monkeypatch, tmp_path):
    example_root = tmp_path / "example_runs"
    monkeypatch.setattr(simulation_options_module, "EXAMPLE_OUTPUT_ROOT", example_root)
    monkeypatch.setattr(sys, "argv", ["/repo/examples/us_bmode_phased_array/us_bmode_phased_array.py"])

    options = SimulationOptions(save_to_disk=True)

    assert options.data_path == str(example_root / "us_bmode_phased_array")
    assert Path(options.data_path).exists()
    assert Path(options.input_filename).parent == Path(options.data_path)
    assert Path(options.output_filename).parent == Path(options.data_path)


def test_resolve_filenames_for_run_avoids_overwrites_under_example_root(tmp_path, monkeypatch):
    example_root = tmp_path / "example_runs"
    monkeypatch.setattr(simulation_options_module, "EXAMPLE_OUTPUT_ROOT", example_root)

    output_dir = example_root / "sd_focussed_detector_2D"
    options = SimulationOptions(
        save_to_disk=True,
        data_path=output_dir,
        input_filename="input.h5",
        output_filename="output.h5",
    )

    first_input, first_output = resolve_filenames_for_run(options)
    Path(first_input).touch()
    Path(first_output).touch()

    second_input, second_output = resolve_filenames_for_run(options)

    assert first_input != second_input
    assert first_output != second_output
    assert second_input.endswith("input_1.h5")
    assert second_output.endswith("output_1.h5")


def test_allow_file_overwrite_disables_suffixing(tmp_path, monkeypatch):
    example_root = tmp_path / "example_runs"
    monkeypatch.setattr(simulation_options_module, "EXAMPLE_OUTPUT_ROOT", example_root)

    output_dir = example_root / "checkpointing"
    options = SimulationOptions(
        save_to_disk=True,
        data_path=output_dir,
        input_filename="kwave_input.h5",
        output_filename="kwave_output.h5",
        allow_file_overwrite=True,
    )

    first_input, first_output = resolve_filenames_for_run(options)
    Path(first_input).touch()
    Path(first_output).touch()

    second_input, second_output = resolve_filenames_for_run(options)

    assert second_input == first_input
    assert second_output == first_output


def test_stream_to_disk_relative_path_uses_data_path(tmp_path):
    options = SimulationOptions(save_to_disk=True, data_path=tmp_path, stream_to_disk="harmonic_data.h5")

    assert options.stream_to_disk == str(tmp_path / "harmonic_data.h5")
