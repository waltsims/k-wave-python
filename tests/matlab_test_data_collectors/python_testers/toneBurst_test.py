import os
import re
from pathlib import Path

import numpy as np

from kwave.utils.signals import tone_burst
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


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
        if isinstance(input_args[i], str):
            # If it is a string, check if it is a vararg string
            if input_args[i] in vararg_strings:
                # If it is a vararg string, store the next input argument in the vararg_inputs dictionary
                camel_case_key = camel_to_snake(input_args[i])
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
            args.append(input_args[i])
            np.delete(input_args, i, axis=0)
            i = i + 1
    return args, vararg_inputs


def test_tone_burst():
    reader = TestRecordReader(os.path.join(Path(__file__).parent, 'collectedValues/tone_burst.mat'))

    for i in range(len(reader)):
        params = reader.expected_value_of("params")
        expected_signal = reader.expected_value_of("input_signal")

        args, varargs = parse_args(params, {'Envelope', 'SignalOffset', 'SignalLength'})
        assert len(args) == 3
        input_signal = tone_burst(*args, **varargs)

        assert np.allclose(input_signal, expected_signal), "tone_burst did not match expected tone_burst"
        reader.increment()
