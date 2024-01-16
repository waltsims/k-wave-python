from beartype.typing import Union
from nptyping import Int, Bool, Float, Complex, NDArray, Shape
import numpy as np


INT = Union[Int, int]
NUMERIC = Union[int, float, np.number]
NUMERIC_WITH_COMPLEX = Union[int, float, complex, np.number]


NP_ARRAY_INT_1D = NDArray[Shape["Dim1"], Int]
NP_ARRAY_FLOAT_1D = NDArray[Shape["Dim1"], Float]
NP_ARRAY_BOOL_1D = NDArray[Shape["Dim1"], Bool]
NP_ARRAY_COMPLEX_1D = NDArray[Shape["Dim1"], Complex]
NP_ARRAY_INT_2D = NDArray[Shape["Dim1, Dim2"], Int]
NP_ARRAY_BOOL_2D = NDArray[Shape["Dim1, Dim2"], Bool]
NP_ARRAY_FLOAT_2D = NDArray[Shape["Dim1, Dim2"], Float]
NP_ARRAY_INT_3D = NDArray[Shape["Dim1, Dim2, Dim3"], Int]
NP_ARRAY_BOOL_3D = NDArray[Shape["Dim1, Dim2, Dim3"], Bool]
NP_ARRAY_FLOAT_3D = NDArray[Shape["Dim1, Dim2, Dim3"], Float]
