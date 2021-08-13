import numpy as np
from itertools import compress


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
