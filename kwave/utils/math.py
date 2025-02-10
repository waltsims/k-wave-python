import math
from itertools import compress
from typing import Optional, Tuple, Union, List


import numpy as np
from numpy.fft import ifftshift, fft, ifft
from scipy.spatial.transform import Rotation
from scipy.fft import fftshift
from functools import wraps
import warnings

from kwave.data import Vector
from kwave.utils.deprecation import deprecated


@deprecated("Use scipy.spatial.transform.Rotation instead", "0.5.0")
def Rx(theta: float) -> np.ndarray:
    """Create a rotation matrix for rotation about the x-axis.

    Args:
        theta: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    return Rotation.from_euler("x", theta, degrees=True).as_matrix()


@deprecated("Use scipy.spatial.transform.Rotation instead", "0.5.0")
def Ry(theta: float) -> np.ndarray:
    """Create a rotation matrix for rotation about the y-axis.

    Args:
        theta: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    return Rotation.from_euler("y", theta, degrees=True).as_matrix()


@deprecated("Use scipy.spatial.transform.Rotation instead", "0.5.0")
def Rz(theta: float) -> np.ndarray:
    """Create a rotation matrix for rotation about the z-axis.

    Args:
        theta: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    return Rotation.from_euler("z", theta, degrees=True).as_matrix()


@deprecated("Use make_affine instead", "0.5.0")
def get_affine_matrix(translation: Vector, rotation: Union[float, List[float]], seq: str = "xyz") -> np.ndarray:
    return make_affine(translation, rotation, seq)


def make_affine(translation: Vector, rotation: Union[float, List[float]], seq: str = "xyz") -> np.ndarray:
    """
    Create an affine transformation matrix combining rotation and translation.
    Uses scipy.spatial.transform.Rotation internally.

    Args:
        translation: [dx, dy] or [dx, dy, dz]
        rotation: Single angle (degrees) for 2D or list of angles for 3D
        seq: Rotation sequence for 3D (default: 'xyz')

    Returns:
        3x3 (2D) or 4x4 (3D) affine transformation matrix

    Examples:
        # 2D transform (rotation around z-axis)
        T1 = make_affine([1, 2], 45)

        # 3D transform with xyz Euler angles
        T2 = make_affine([1, 2, 3], [45, 30, 60])

        # 3D transform with custom sequence
        T3 = make_affine([1, 2, 3], [45, 30], 'xy')
    """
    if len(translation) == 2:
        # 2D transformation
        R = Rotation.from_euler("z", rotation, degrees=True).as_matrix()[:2, :2]
        return np.array([[R[0, 0], R[0, 1], translation[0]], [R[1, 0], R[1, 1], translation[1]], [0, 0, 1]])
    else:
        # 3D transformation
        R = Rotation.from_euler(seq, rotation, degrees=True)
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = translation
        return T


def cosd(angle_in_degrees):
    """Compute cosine of angle in degrees."""
    return np.cos(np.radians(angle_in_degrees))


def sind(angle_in_degrees):
    """Compute sine of angle in degrees."""
    return np.sin(np.radians(angle_in_degrees))


def largest_prime_factor(n: int) -> int:
    """
    Finds the largest prime factor of a positive integer.

    Args:
        n: The positive integer to be factored.

    Returns:
        The largest prime factor of n.

    """
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def rwh_primes(n: int) -> List[int]:
    """
    Generates a list of prime numbers less than a given integer.

    Args:
        n: The upper bound for the list of primes.

    Returns:
        A list of prime numbers less than n.

    """
    sieve = bytearray([True]) * (n // 2 + 1)
    for i in range(1, int(n**0.5) // 2 + 1):
        if sieve[i]:
            sieve[2 * i * (i + 1) :: 2 * i + 1] = bytearray((n // 2 - 2 * i * (i + 1)) // (2 * i + 1) + 1)
    return [2, *compress(range(3, n, 2), sieve[1:])]


def primefactors(n: int) -> List[int]:
    """
    Finds the prime factors of a given integer.

    Args:
        n: The integer to factor.

    Returns:
        A list of prime factors of n.

    """
    factors = []
    while n % 2 == 0:
        (factors.append(2),)
        n = n / 2

    # n became odd
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n = n / i

    if n > 2:
        factors.append(n)

    return factors


def next_pow2(n: int) -> int:
    """
    Calculate the next power of 2 that is greater than or equal to `n`.

    This function takes a positive integer `n` and returns the smallest power of 2 that is greater
    than or equal to `n`.

    Args:
        n: The number to find the next power of 2 for.

    Returns:
        The smallest power of 2 that is greater than or equal to `n`.

    """
    # decrement `n` (to handle cases when `n` itself is a power of 2)
    n = n - 1

    # set all bits after the last set bit
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16

    # increment `n` and return
    return np.log2(n + 1)


def phase_shift_interpolate(data: np.ndarray, shift: float, shift_dim: Optional[int] = None) -> np.ndarray:
    """
    Interpolates array data using phase shifts in the Fourier domain.

    This function applies a phase shift in the frequency domain to achieve sub-grid interpolation.
    The phase shift method allows for precise fractional shifts without losing frequency content,
    making it particularly useful for resampling data between staggered and regular grids.

    Example:
        # Move velocity data from staggered to regular grid points
        v_regular = phase_shift_interpolate(v_staggered, shift=0.5)

    Args:
        data: The input array to be interpolated.
        shift: Phase shift amount (in grid points). A shift of 1 means one full grid spacing.
        shift_dim: The dimension along which to apply the phase shift. Default is the last dimension.

    Returns:
        The interpolated array after applying the phase shift.
    """
    # Handle dimension selection
    if shift_dim is None:
        shift_dim = data.ndim - 1 if not (data.ndim == 2 and data.shape[1] == 1) else 0
    else:
        shift_dim = shift_dim - 1
        if not (0 <= shift_dim <= 3):
            raise ValueError("Input dim must be 0, 1, 2 or 3.")
        elif shift_dim >= data.ndim:
            Warning(f"Shift dimension {shift_dim}is greater than the number of dimensions in the input array {data.ndim}.")
            shift_dim = data.ndim - 1

    N = data.shape[shift_dim]

    # Create k-vector and handle even/odd cases
    k_vec = np.arange(-(N // 2), (N + 1) // 2) * (2 * np.pi / N)
    k_vec[N // 2] = 0  # Force middle value to zero

    # Reshape k_vec to match data dimensions for broadcasting
    expand_dims = [1] * data.ndim
    expand_dims[shift_dim] = -1
    k_vec = k_vec.reshape(expand_dims)

    # Apply phase shift in Fourier domain
    return np.real(ifft(fft(data, axis=shift_dim) * ifftshift(np.exp(1j * k_vec * shift)), axis=shift_dim))


@deprecated(
    "This function has been renamed to phase_shift_interpolate to better reflect its functionality. "
    "Please use phase_shift_interpolate instead.",
    "0.5.0",
)
def fourier_shift(data: np.ndarray, shift: float, shift_dim: Optional[int] = None) -> np.ndarray:
    """Wrapper for phase_shift_interpolate. See its documentation for details."""
    return phase_shift_interpolate(data, shift, shift_dim)


def round_even(x):
    """
    Rounds to the nearest even integer.

    Args:
        x: Input value

    Returns:
        Nearest even integer.

    """
    return 2 * round(x / 2)


def round_odd(x):
    """
    Rounds to the nearest odd integer.

    Args:
        x: Input value

    Returns:
        Nearest odd integer.

    """
    return 2 * round((x + 1) / 2) - 1


def find_closest(A: np.ndarray, a: Union[float, int]) -> Tuple[Union[float, int], Tuple[int, ...]]:
    """
    Returns the value and index of the item in A that is closest to the value a.

    This function finds the value and index of the item in the input array A that is closest to the given value a.
    For vectors, the value and index correspond to the closest element in A. For matrices, value and index are row
    vectors corresponding to the closest element from each column. For N-D arrays, the function finds the closest
    value along the first matrix dimension (singleton dimensions are removed before the search). If there is more
    than one element with the closest value, the index of the first one is returned.

    Args:
        A: The array to search.
        a: The value to find.

    Returns:
        A tuple containing the value and index of the closest element in A to a.

    """
    assert isinstance(A, np.ndarray), "A must be an np.array"

    idx = np.unravel_index(np.argmin(abs(A - a)), A.shape)
    return A[idx], idx


def sinc(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculates the sinc function of a given value or array of values.

    Args:
        x: The value or array of values for which to calculate the sinc function.

    Returns:
        The sinc function of x.

    """
    return np.sinc(x / np.pi)


def norm_var(im: np.ndarray) -> float:
    """
    Calculates the normalized variance of an array of values.

    Args:
        im: The input array.

    Returns:
        The normalized variance of im.

    """
    mu = np.mean(im)
    s = np.sum((im - mu) ** 2) / mu
    return s


def gaussian(
    x: Union[int, float, np.ndarray],
    magnitude: Optional[Union[int, float]] = None,
    mean: Optional[float] = 0,
    variance: Optional[float] = 1,
) -> Union[int, float, np.ndarray]:
    """
    Returns a Gaussian distribution f(x) with the specified magnitude, mean, and variance. If these values are not specified,
    the magnitude is normalised and values of variance = 1 and mean = 0 are used. For example running:

        import matplotlib.pyplot as plt
        x = np.arange(-3, 0.05, 3)
        plt.plot(x, gaussian(x))

    will plot a normalised Gaussian distribution.

    Note, the full width at half maximum of the resulting distribution can be calculated by FWHM = 2 * sqrt(2 * log(2) * variance).

    Args:
        x: The input values.
        magnitude: Bell height. Defaults to normalised.
        mean: Mean or expected value. Defaults to 0.
        variance: Variance, or bell width. Defaults to 1.

    Returns:
        A Gaussian distribution.

    """
    if magnitude is None:
        magnitude = (2 * math.pi * variance) ** -0.5

    gauss_distr = magnitude * np.exp(-((x - mean) ** 2) / (2 * variance))

    return gauss_distr


def _compute_direction(start_pos: np.ndarray, end_pos: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute normalized direction vector and magnitude between two points."""
    direction = end_pos - start_pos
    magnitude = np.linalg.norm(direction)
    direction = direction / magnitude
    return direction, magnitude


def _compute_rotation_axis(reference: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute normalized rotation axis and its magnitude."""
    axis = np.cross(reference, direction)
    axis_norm = np.linalg.norm(axis)
    return axis, axis_norm


def _create_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix using Rodrigues' formula."""
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Skew-symmetric matrix of axis
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

    # Outer product
    outer = np.outer(axis, axis)

    return cos_theta * np.eye(3) + sin_theta * skew + (1 - cos_theta) * outer


def compute_rotation_between_vectors(start_pos: np.ndarray, end_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotation matrix between two 3D points.

    Args:
        start_pos: Starting position vector
        end_pos: Ending position vector

    Returns:
        Tuple containing:
        - 3x3 rotation matrix
        - Normalized direction vector
    """
    direction, magnitude = _compute_direction(start_pos, end_pos)

    if np.isclose(magnitude, 0):
        return np.eye(3), np.zeros(3)

    reference = np.array([0.0, 0.0, -1.0])

    axis, axis_norm = _compute_rotation_axis(reference, direction)

    if axis_norm > np.finfo(float).eps:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(reference, direction), -1.0, 1.0))
        rot_mat = _create_rotation_matrix(axis, angle)
    else:
        # Vectors are parallel or anti-parallel
        rot_mat = np.eye(3) if np.dot(reference, direction) > 0 else -np.eye(3)

    return rot_mat, direction


def compute_linear_transform(pos1, pos2, offset=None):
    """
    Compute linear transformation between two 3D points.

    This function computes the linear transformation that maps a vector pointing from
    pos1 to pos2 into the canonical direction [0, 0, -1].

    Args:
        pos1: Starting position (3D point)
        pos2: Ending position (3D point)
        offset: Offset vector (3D point)

    Returns:
        Tuple containing:
        - 3x3 rotation matrix

    """
    rot_mat, direction = compute_rotation_between_vectors(pos1, pos2)
    if offset is not None:
        offset_pos = pos1 + offset * direction
    else:
        offset_pos = 0
    return rot_mat, offset_pos
