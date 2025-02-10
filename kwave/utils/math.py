import math
from itertools import compress
from typing import Optional, Tuple, Union, List


import numpy as np
from numpy.fft import ifftshift, fft, ifft
from scipy.spatial.transform import Rotation

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


def fourier_shift(data: np.ndarray, shift: float, shift_dim: Optional[int] = None) -> np.ndarray:
    """
    Shifts an array along one of its dimensions using Fourier interpolation.

    Args:
        data: The input array.
        shift: The amount of shift to apply.
        shift_dim: The dimension along which to shift the array. Default is the last dimension.

    Returns:
        The shifted array.

    """

    if shift_dim is None:
        shift_dim = data.ndim - 1
        if (shift_dim == 1) and (data.shape[1] == 1):
            # row vector
            shift_dim = 0
    else:
        shift_dim -= 1
        if not (0 <= shift_dim <= 3):
            raise ValueError("Input dim must be 0, 1, 2 or 3.")
        else:
            # subtract 1 in order to keep function interface compatible with matlab
            if shift_dim >= data.ndim:
                Warning(f"Shift dimension {shift_dim}is greater than the number of dimensions in the input array {data.ndim}.")
                shift_dim = data.ndim - 1

    N = data.shape[shift_dim]

    if N % 2 == 0:
        # grid dimension has an even number of points
        k_vec = (2 * np.pi) * (np.arange(-N // 2, N // 2) / N)
    else:
        # grid dimension has an odd number of points
        k_vec = (2 * np.pi) * (np.arange(-(N - 1) // 2, N // 2 + 1) / N)

    # force middle value to be zero in case 1/N is a recurring number and the
    # series doesn't give exactly zero
    k_vec[N // 2] = 0

    reshape_dims_to = [1] * data.ndim
    reshape_dims_to[shift_dim] = -1
    k_vec = np.reshape(k_vec, reshape_dims_to)

    # shift the input using a Fourier interpolant
    phase_shift = ifftshift(np.exp(1j * k_vec * shift))
    kdata = fft(data, axis=shift_dim)
    shifted_kdata = kdata * phase_shift
    result = ifft(shifted_kdata, axis=shift_dim).real
    return result


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
    # return magnitude * norm.pdf(x, loc=mean, scale=variance)
    """ # Former impl. form Farid
        if magnitude is None:
        magnitude = np.sqrt(2 * np.pi * variance)
    return magnitude * np.exp(-(x - mean) ** 2 / (2 * variance))
    """


def cosd(angle_in_degrees):
    # Note:
    #   Using numpy.radians instead math.radians
    #   does not yield the same results as matlab
    angle_in_radians = math.radians(angle_in_degrees)
    return math.cos(angle_in_radians)


def sind(angle_in_degrees):
    # Note:
    #   Using numpy.radians instead math.radians
    #   does not yield the same results as matlab
    angle_in_radians = math.radians(angle_in_degrees)
    return math.sin(angle_in_radians)


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


def _compute_direction_vector(start_pos: np.ndarray, end_pos: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute and normalize the direction vector between two points.

    Args:
        start_pos: Starting position (3D point)
        end_pos: Ending position (3D point)

    Returns:
        Tuple containing:
        - Normalized direction vector
        - Magnitude of the direction vector
    """
    direction = end_pos - start_pos
    magnitude = np.linalg.norm(direction)
    if magnitude > 0:
        direction = direction / magnitude  # Use numpy's broadcasting
    return direction, magnitude


def _compute_rotation_axis(reference: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Compute the normalized rotation axis between two vectors.

    Args:
        reference: Reference direction vector
        direction: Target direction vector

    Returns:
        Normalized rotation axis vector
    """
    # Use numpy's cross product and normalization
    u = np.cross(reference, direction)
    norm = np.linalg.norm(u)
    if norm > 0:
        u = u / norm
    return u


def _compute_rotation_angle(reference: np.ndarray, direction: np.ndarray) -> float:
    """Compute the rotation angle between two vectors.

    Args:
        reference: Reference direction vector
        direction: Target direction vector

    Returns:
        Rotation angle in radians
    """
    # Use numpy's clip to handle numerical precision issues
    cos_theta = np.clip(np.dot(reference, direction), -1.0, 1.0)
    return np.arccos(cos_theta)


def _compute_rodrigues_rotation_matrix(u: np.ndarray, theta: float) -> np.ndarray:
    """Compute rotation matrix using Rodrigues' formula.

    Args:
        u: Rotation axis (normalized)
        theta: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    # Create skew-symmetric matrix of u
    u_skew = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    # Compute outer product u⊗u
    u_outer = np.outer(u, u)

    # Combine terms in the same order as MATLAB
    return np.cos(theta) * np.eye(3) + np.sin(theta) * u_skew + (1 - np.cos(theta)) * u_outer


def compute_rotation_between_vectors(start_pos: np.ndarray, end_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotation matrix between two 3D points.

    This function computes the rotation matrix that transforms a vector pointing from
    start_pos to end_pos into the canonical direction [0, 0, -1].
    Uses Rodrigues' rotation formula to match MATLAB's behavior exactly.

    Args:
        start_pos: Starting position (3D point)
        end_pos: Ending position (3D point)

    Returns:
        Tuple containing:
        - 3x3 rotation matrix
        - Direction vector from start to end position (normalized)
    """
    # Compute and normalize direction vector
    direction, magnitude = _compute_direction_vector(start_pos, end_pos)

    if np.isclose(magnitude, 0):
        return np.eye(3), np.zeros_like(start_pos)

    # Reference direction (canonical vector)
    reference = np.array([0, 0, -1])

    # Find the rotation axis and angle
    u = _compute_rotation_axis(reference, direction)
    theta = _compute_rotation_angle(reference, direction)

    # Compute rotation matrix using Rodrigues' formula
    rotation_matrix = _compute_rodrigues_rotation_matrix(u, theta)

    return rotation_matrix, direction


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
