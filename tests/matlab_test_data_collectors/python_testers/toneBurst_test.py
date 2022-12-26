import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kwave.utils.signals import tone_burst


def camel_to_snake(string):
    # Use a regular expression to match uppercase letters that are not at the beginning of the string and replace them
    # with an underscore followed by the lowercase version of the letter
    return re.sub(r'(?<!^)([A-Z])', r'_\1', string).lower()


def parse_args(input_args, vararg_strings):
    # Initialize variables to store the input arguments and the vararg input arguments
    args = []
    vararg_inputs = {}
    i = 0
    # Iterate through the input arguments
    while i < len(input_args):
        # Check if the current input argument is a string
        if isinstance(input_args[i][0], str):
            # If it is a string, check if it is a vararg string
            if input_args[i][0] in vararg_strings:
                # If it is a vararg string, store the next input argument in the vararg_inputs dictionary
                camel_case_key = camel_to_snake(np.squeeze(input_args[i][0]))
                val = input_args[i + 1]
                if np.squeeze(val).size == 1:
                    vararg_inputs[camel_case_key] = int(val) if str(np.squeeze(val)).isnumeric() else str(
                        np.squeeze(val))
                    input_args = np.concatenate((input_args[:i], input_args[i + 2:]), axis=0)
                else:
                    vararg_inputs[camel_case_key] = np.squeeze(val)
                    input_args = np.concatenate((input_args[:i], input_args[i + 2:]), axis=0)
        else:
            # If it is not a string, it is a regular input argument
            # Convert it to a float if the input argument is an array-like object with length one
            if isinstance(input_args[i], (list, np.ndarray)) and len(input_args[i]) == 1:
                if np.squeeze(input_args[i]).size == 1:
                    args.append(float(np.squeeze(input_args[i])))
                    np.delete(input_args, i, axis=0)
                else:
                    args.append(np.squeeze(input_args[i]))
                    np.delete(input_args, i, axis=0)
            i = i + 1
    return args, vararg_inputs


def test_tone_burst():
    collected_values_folder = os.path.join(Path(__file__).parent, 'collectedValues/toneBurst')
    num_collected_values = len(os.listdir(collected_values_folder))

    for i in range(num_collected_values):
        filepath = os.path.join(collected_values_folder, f'{i:06d}.mat')
        recorded_data = loadmat(filepath)
        params = np.squeeze(recorded_data['params'][0][i])

        args, varargs = parse_args(params, {'Envelope', 'SignalOffset', 'SignalLength'})
        assert len(args) == 3
        local_output = tone_burst(*args, **varargs)
        output_signal = np.squeeze(recorded_data['input_signal'])

        assert np.allclose(output_signal, local_output)
