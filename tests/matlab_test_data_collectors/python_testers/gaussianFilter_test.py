from scipy.io import loadmat
import numpy as np
import os
from pathlib import Path

from kwave.utils.filters import gaussian_filter


def test_gaussianFilter():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/gaussianFilter')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        fs = float(recorded_data['fs'])
        fc = float(recorded_data['fc'])
        bw = float(recorded_data['bw'])
        input_signal = np.squeeze(recorded_data['input_signal'])
        output_signal = np.squeeze(recorded_data['output_signal'])
        local_output = gaussian_filter(input_signal, fs, fc, bw)

        assert np.allclose(output_signal, local_output)
