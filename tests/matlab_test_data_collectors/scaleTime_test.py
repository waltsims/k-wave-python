from kwave.utils.conversionutils import scale_time

from scipy.io import loadmat
import numpy as np
import os

collected_values_folder = '/data/code/Work/black_box_testing/collectedValues_scaleTime'
num_collected_values = len(os.listdir(collected_values_folder))


for i in range(num_collected_values):
    filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
    recorded_data = loadmat(filepath)

    seconds = np.squeeze(recorded_data['seconds'])
    expected_time = str(recorded_data['time'][0])

    time = scale_time(seconds)
    assert time == expected_time

print('scale_time(..) works as expected!')
