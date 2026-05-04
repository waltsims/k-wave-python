import numpy as np
from beartype.typing import Union
from jaxtyping import Bool, Complex, Float, Int, Shaped

# Base array-like types: NumPy arrays + scalars + Python numeric scalars.
_ARRAY_LIKE_TYPES = (
    np.ndarray,  # NumPy array type
    np.bool_,
    np.number,  # NumPy scalar types
    bool,
    int,
    float,
    complex,  # Python scalar types
)

# CuPy arrays are accepted by all array-typed APIs when CuPy is installed (GPU backend).
# When CuPy is missing the tuple is empty and ArrayLike collapses to the NumPy-only form.
# Subscript-time splats (PEP 646) require Python 3.11+, so we build the Union via
# ``Union[tuple]`` which works on 3.10.
try:
    import cupy as _cp

    _ARRAY_LIKE_TYPES = _ARRAY_LIKE_TYPES + (_cp.ndarray,)
except ImportError:
    pass

ArrayLike = Union[_ARRAY_LIKE_TYPES]

ScalarLike = Shaped[ArrayLike, ""]


INT = Union[Int, int]
NUMERIC = Union[int, float, np.number]
NUMERIC_WITH_COMPLEX = Union[int, float, complex, np.number]


NP_ARRAY_INT_1D = Int[np.ndarray, "Dim1"]
NP_ARRAY_FLOAT_1D = Float[np.ndarray, "Dim1"]
NP_ARRAY_BOOL_1D = Bool[np.ndarray, "Dim1"]
NP_ARRAY_COMPLEX_1D = Complex[np.ndarray, "Dim1"]
NP_ARRAY_INT_2D = Int[np.ndarray, "Dim1 Dim2"]
NP_ARRAY_BOOL_2D = Bool[np.ndarray, "Dim1 Dim2"]
NP_ARRAY_FLOAT_2D = Float[np.ndarray, "Dim1 Dim2"]
NP_ARRAY_INT_3D = Int[np.ndarray, "Dim1 Dim2 Dim3"]
NP_ARRAY_BOOL_3D = Bool[np.ndarray, "Dim1 Dim2 Dim3"]
NP_ARRAY_FLOAT_3D = Float[np.ndarray, "Dim1 Dim2 Dim3"]

NP_DOMAIN = Union[Float[np.ndarray, "1"], Float[np.ndarray, "2"], Float[np.ndarray, "3"]]
