from datetime import datetime

import numpy as np


def get_date_string():
    return datetime.now().strftime("%d-%b-%Y-%H-%M-%S")


def gaussian(x, magnitude=None, mean=0, variance=1):
    if magnitude is None:
        magnitude = np.sqrt(2 * np.pi * variance)
    return magnitude * np.exp(-(x - mean)**2 / (2 * variance))


def ndgrid(*args):
    return np.array(np.meshgrid(*args, indexing='ij'))


def sinc(x):
    return np.sin(x) / x
