from kwave.utils import fourier_shift

from scipy.io import loadmat
import numpy as np
import os
import pytest


@pytest.mark.skip(reason="no way of currently testing this")
def test_fourier_shift():
    collected_values_folder = '/data/code/Work/black_box_testing/collectedValues_fourierShift'
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        # Read recorded data
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        data = recorded_data['data']
        shift = float(recorded_data['shift'])
        if 'shift_dim' in recorded_data:
            shift_dim = int(recorded_data['shift_dim'])
        else:
            shift_dim = None
        expected_shifted_data = recorded_data['shifted_data']

        # Execute implementation
        shifted_data = fourier_shift(data, shift, shift_dim)

        # Check correctness
        assert np.allclose(shifted_data, expected_shifted_data)

    print('fourier_shift(..) works as expected!')


if __name__ == '__main__':
    test_fourier_shift()
