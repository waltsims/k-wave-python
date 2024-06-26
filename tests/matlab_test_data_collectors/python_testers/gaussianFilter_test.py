import numpy as np
import os
from pathlib import Path
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader
from kwave.utils.filters import gaussian_filter


def test_gaussianFilter():
    collected_values_folder = os.path.join(Path(__file__).parent, "collectedValues/gaussianFilter")
    record_reader = TestRecordReader(collected_values_folder)

    for _ in range(len(record_reader)):
        fs = record_reader.expected_value_of("fs")
        fc = record_reader.expected_value_of("fc")
        bw = record_reader.expected_value_of("bw")
        input_signal = record_reader.expected_value_of("input_signal")
        output_signal = record_reader.expected_value_of("output_signal")
        local_output = gaussian_filter(input_signal, fs, fc, bw)

        assert np.allclose(output_signal, local_output)
        record_reader.increment()
