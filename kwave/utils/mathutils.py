from typing import Optional

import numpy as np
from itertools import compress

from numpy.fft import ifftshift, fft, ifft


def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def rwh_primes(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Returns a list of primes < n for n > 2 """
    sieve = bytearray([True]) * (n//2+1)
    for i in range(1, int(n**0.5)//2+1):
        if sieve[i]:
            sieve[2*i*(i+1)::2*i+1] = bytearray((n//2-2*i*(i+1))//(2*i+1)+1)
    return [2, *compress(range(3, n, 2), sieve[1:])]


def check_divisible(number: float, divider: float) -> bool:
    """
        Checks whether number is divisible by divider without any remainder
        Why do we need such a function? -> Because due to floating point precision we
        experience rounding errors while using standard modulo operator with floating point numbers
    Args:
        number: Number that's supposed to be divided
        divider: Divider that should devide the number

    Returns:

    """
    result = number / divider
    after_decimal = result % 1
    return after_decimal == 0


def fourier_shift(
        data: np.ndarray,
        shift: float,
        shift_dim: Optional[int] = None
) -> np.ndarray:
    if shift_dim is None:
        shift_dim = data.ndim - 1
        if (shift_dim == 1) and (data.shape[1] == 1):
            # row vector
            shift_dim = 0
    else:
        # subtract 1 in order to keep function interface compatible with matlab
        shift_dim -= 1

    try:
        N = data.shape[shift_dim]
    except IndexError:
        print('aaa')
    if N % 2 == 0:
        # grid dimension has an even number of points
        k_vec = (2 * np.pi) * ( np.arange(-N // 2, N // 2) / N)
    else:
        # grid dimension has an odd number of points
        k_vec = (2 * np.pi) * ( np.arange(-(N -1) // 2, N // 2 + 1) / N)

    # force middle value to be zero in case 1/N is a recurring number and the
    # series doesn't give exactly zero
    k_vec[N // 2] = 0

    # put the wavenumber vector in the correct orientation for use with bsxfun
    reshape_dims_to = [1] * data.ndim
    if 0 <= shift_dim <= 3:
        reshape_dims_to[shift_dim] = -1
        k_vec = np.reshape(k_vec, reshape_dims_to)
    else:
        raise ValueError('Input dim must be 0, 1, 2 or 3.')

    # shift the input using a Fourier interpolant
    part_1 = ifftshift(np.exp(1j * k_vec * shift))
    part_2 = fft(data, axis=shift_dim)
    part_1_times_2 = part_1 * part_2
    result = ifft(part_1_times_2, axis=shift_dim).real
    return result
