from beartype.typing import Union
from nptyping import Int
import numpy as np


INT = Union[Int, int]
NUMERIC = Union[int, float, np.number]
NUMERIC_WITH_COMPLEX = Union[int, float, complex, np.number]
