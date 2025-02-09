import numpy as np
from scipy.io import loadmat


class TestRecordReader(object):
    # Will make `pytest` to ignore this class as a test class
    __test__ = False

    def __init__(self, record_filename):
        recorded_data = loadmat(record_filename, simplify_cells=True)
        self._records = recorded_data
        self._total_steps = recorded_data["total_steps"]
        self._step = 0

    def expected_value_of(self, name, squeeze=False):
        record_key = f"step_{self._step}___{name}"
        value = self._records[record_key]
        if squeeze:
            value = np.squeeze(value)
        return value

    def increment(self):
        self._step += 1
        if self._step > self._total_steps:
            raise ValueError("Exceeded total recorded steps. Perhaps something is wrong with logic?")

    def __len__(self):
        return self._total_steps
