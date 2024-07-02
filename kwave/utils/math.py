import math
from itertools import compress
from typing import Optional, Tuple, Union, List

import numpy as np
from numpy.fft import ifftshift, fft, ifft

from kwave.data import Vector


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


def Rx(theta):
    """
    3D rotation matrix for rotation about x-axis

    Args:
    theta : float. Angle of rotation (in degrees)

    Returns:
    np.array. 3D rotation matrix
    """
    R = np.array([[1, 0, 0], [0, cosd(theta), -sind(theta)], [0, sind(theta), cosd(theta)]])
    return R


def Ry(theta):
    """
    3D rotation matrix for rotation about y-axis

    Args:
    theta : float. Angle of rotation (in degrees)

    Returns:
    np.array. 3D rotation matrix
    """
    R = np.array([[cosd(theta), 0, sind(theta)], [0, 1, 0], [-sind(theta), 0, cosd(theta)]])
    return R


def Rz(theta):
    """
    3D rotation matrix for rotation about z-axis

    Args:
    theta : float. Angle of rotation (in degrees)

    Returns:
    np.array. 3D rotation matrix
    """
    R = np.array([[cosd(theta), -sind(theta), 0], [sind(theta), cosd(theta), 0], [0, 0, 1]])
    return R


def get_affine_matrix(translation: Vector, rotation: Union[int, float, np.ndarray, Vector]):
    # Check dimensions
    if len(translation) == 2 and isinstance(rotation, (int, float)):
        # Assign the inputs
        dx = translation[0]
        dy = translation[1]
        th = rotation

        # Build affine matrix (counter-clockwise)
        affine = np.array([[cosd(th), -sind(th), dx], [sind(th), cosd(th), dy], [0, 0, 1]])

    elif len(translation) == 3 and isinstance(rotation, (np.ndarray, Vector)) and len(rotation) == 3:
        # Assign the inputs
        dx, dy, dz = translation
        x_th, y_th, z_th = rotation

        # Build the rotation matrices
        x_th_matrix = np.array([[1, 0, 0], [0, cosd(x_th), -sind(x_th)], [0, sind(x_th), cosd(x_th)]])

        y_th_matrix = np.array([[cosd(y_th), 0, sind(y_th)], [0, 1, 0], [-sind(y_th), 0, cosd(y_th)]])

        z_th_matrix = np.array([[cosd(z_th), -sind(z_th), 0], [sind(z_th), cosd(z_th), 0], [0, 0, 1]])

        # Build affine matrix
        affine = np.zeros((4, 4))
        affine[0:3, 0:3] = np.dot(z_th_matrix, np.dot(y_th_matrix, x_th_matrix))
        affine[:, 3] = [dx, dy, dz, 1]

    else:
        raise ValueError("Incorrect size for translation and rotation inputs.")

    return affine


def compute_linear_transform(pos1, pos2, offset=None):
    # Compute vector pointing from pos1 to pos2
    beam_vec = pos2 - pos1

    magnitude = np.linalg.norm(beam_vec)

    #  matlab behaviour is to return nans when positions are the same.
    #  we choose to return the identity matrix and the offset in this case.
    #  TODO: we should open an issue and change our behaviour once matlab is fixed.

    # if np.isclose(magnitude, 0):

    #     # "pos1 and pos2 are the same"
    #     if (shape1 := np.shape(pos1)) == (shape2 := np.shape(pos2)):
    #         raise ValueError(f"pos1 and pos2 must have the same shape. Received shapes: {shape1} and {shape2}")
    #     return np.eye(3), np.zeros_like(pos1) if offset is None else offset * np.ones_like(pos1)

    # Normalise to give unit beam vector
    beam_vec = beam_vec / magnitude

    # Canonical normalised beam_vec (canonical pos1 is [0, 0, 1])
    beam_vec0 = np.array([0, 0, -1])

    # Find the rotation matrix for the bowl
    u = np.cross(beam_vec0, beam_vec)

    # Normalise the rotation matrix if not zero
    if any(u != 0):
        u = u / np.linalg.norm(u)

    # Find the axis-angle transformation between beam_vec and e1
    theta = np.arccos(np.dot(beam_vec0, beam_vec))

    # Convert axis-angle transformation to a rotation matrix
    A = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotMat = np.cos(theta) * np.eye(3) + np.sin(theta) * A + (1 - np.cos(theta)) * np.outer(u, u)

    # Compute an offset for the bowl, where bowl_centre = move from pos1
    # towards focus by radius
    if offset is not None:
        offsetPos = pos1 + offset * beam_vec
    else:
        offsetPos = 0

    return rotMat, offsetPos
