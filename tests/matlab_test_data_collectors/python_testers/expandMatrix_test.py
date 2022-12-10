import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.matrix import expand_matrix


def test_expand_matrix_test():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/expandMatrix')
    num_collected_values = len(os.listdir(collected_values_folder))
    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)

        matrix = recorded_data['matrix']
        matrix = np.squeeze(matrix)
        input_args = recorded_data['input_args']
        exp_coeff = input_args[0][0]
        if len(input_args[0]) == 2:
            edge_val = float(input_args[0][1])
        else:
            edge_val = None
        expected_expanded_matrix = recorded_data['expanded_matrix']
        expected_expanded_matrix = np.squeeze(expected_expanded_matrix)

        expanded_matrix  = expand_matrix(matrix, exp_coeff=exp_coeff, edge_val=edge_val)

        print(i)
        assert np.allclose(expected_expanded_matrix, expanded_matrix, equal_nan=True)


    print('expanded_matrix(..) works as expected!')
