import time

import numpy as np
from matplotlib import pyplot as plt

from kwave.utils.conversion import db2neper
from kwave.utils.data import scale_SI, scale_time
from kwave.utils.filters import next_pow2
from kwave.utils.matrix import expand_matrix
from kwave.utils.tictoc import TicToc


def atten_comp(
        signal, dt, c, alpha_0, y,
):
    raise NotImplementedError
