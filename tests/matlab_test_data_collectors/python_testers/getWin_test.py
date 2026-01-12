import logging
import os
from pathlib import Path

import numpy as np
import pytest

from kwave.utils.filters import get_win
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_get_win():
    test_data_file = os.path.join(Path(__file__).parent, "collectedValues/getWin.mat")
    reader = TestRecordReader(test_data_file)
    for i in range(len(reader)):
        logging.log(logging.INFO, "i: => %d", i)

        N = reader.expected_value_of("N")
        input_args = reader.expected_value_of("input_args")
        type_ = reader.expected_value_of("type_")

        rotation = bool(input_args[1])
        symmetric = bool(input_args[3])
        square = bool(input_args[5])

        if len(input_args) == 8:
            param = float(input_args[7])
        else:
            param = None
            assert len(input_args) == 6

        N = np.squeeze(N)

        logging.log(logging.INFO, "N=%s, type_=%s, param=%s, rotation=%s, symmetric=%s, square=%s", 
                    N, type_, param, rotation, symmetric, square)

        if (np.isscalar(N) and N > 1) or (isinstance(N, np.ndarray) and (N > 1).all()):
            win_py, cg_py = get_win(N, type_, param=param, rotation=rotation, symmetric=symmetric, square=square)

            cg = reader.expected_value_of("cg")
            win = reader.expected_value_of("win")
            win_py = np.squeeze(win_py)
            assert np.shape(win_py) == np.shape(win)
            assert np.allclose(win_py, win, equal_nan=True)
            assert np.allclose(cg_py, cg, equal_nan=True)

        reader.increment()


if __name__ == "__main__":
    pytest.main([__file__])
