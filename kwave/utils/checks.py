import logging
import numbers
import platform
from copy import deepcopy
from typing import List, TYPE_CHECKING, Any

import numpy as np
import scipy
import scipy.optimize

if TYPE_CHECKING:
    # Found here: https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium

from .conversion import db2neper
from .math import sinc, primefactors


def enforce_fields(dictionary, *fields):
    """
    Ensures that the given dictionary contains the specified fields.

    Args:
        dictionary: A dictionary to check.
        *fields: The fields that must be present in the dictionary.

    Raises:
        AssertionError: If any of the specified fields are not in the dictionary.

    """

    for f in fields:
        assert f in dictionary.keys(), [f"The field {f} must be defined in the given dictionary"]


def enforce_fields_obj(obj, *fields):
    """
    Enforces that certain fields are not None in the given object.

    Args:
        obj: Object to check the fields of.
        *fields: List of field names to check.

    Raises:
        AssertionError: If any of the given fields are None in the given object.

    """

    for f in fields:
        assert getattr(obj, f) is not None, f"The field {f} must be not None in the given object"


def check_field_names(dictionary, *fields):
    """
    This method checks if the keys of the given dictionary are valid fields.

    Args:
        dictionary: A dictionary where the keys will be checked for validity.
        *fields: A list of valid field names.

    Raises:
        AssertionError: If any of the keys in the dictionary are not in the list of valid fields.

    """

    for k in dictionary.keys():
        assert k in fields, f"The field {k} is not a valid field for the given dictionary"


def check_str_eq(value, target: str) -> bool:
    """
    This method checks whether the given value is a string and is equal to the target string.
    It is useful to avoid FutureWarnings when value is not a string.

    Args:
        value: The value to check.
        target: The target string to compare with.

    Returns:
        True if the value is a string and is equal to the target, False otherwise.

    """

    return isinstance(value, str) and value == target


def check_str_in(value, target: List[str]) -> bool:
    """
    Check if value is in the given list only if the value is string.
    Helps to avoid FutureWarnings when value is not a string.
    Added by @Farid

    Args:
        value: The value to check for inclusion in `target`
        target: A list of strings to check for the presence of `value`

    Returns:
        True if `value` is a string and is present in `target`, otherwise False

    """

    return isinstance(value, str) and value in target


def is_number(value: Any) -> bool:
    """
    Check if the given value is a numeric type.

    Args:
        value: The value to check.

    Returns:
        True if the value is numeric, False otherwise.

    """

    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return False
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number)
    return np.issubdtype(np.array(value), np.number)


def is_unix() -> bool:
    """
    Check whether the current platform is a Unix-like system.

    Returns:
        True if the current platform is a Unix-like system, False otherwise.

    """
    return platform.system() in ["Linux", "Darwin"]


def _evaluate_absorbing_dt_stability_limit(kmax, c_ref, medium: "kWaveMedium", xtol=1e-12) -> float:
    # convert the absorption coefficient to nepers.(rad/s)^-y.m^-1
    alpha_coeff = db2neper(medium.alpha_coeff, medium.alpha_power)

    # calculate the absorption constant
    # calculate the absorption constant
    if medium.alpha_mode != "no_absorption":
        absorb_tau = -2.0 * alpha_coeff * medium.sound_speed ** (medium.alpha_power - 1.0)
    else:
        absorb_tau = np.array([0])

    # calculate the dispersion constant
    if medium.alpha_mode != "no_dispersion":
        absorb_eta = 2.0 * alpha_coeff * medium.sound_speed ** (medium.alpha_power) * np.tan(np.pi * medium.alpha_power / 2.0)
    else:
        absorb_eta = np.array([0])

    # estimate the timestep required for stability in the absorbing case by
    # assuming the k-space correction factor, kappa = 1 (note that
    # absorb_tau and absorb_eta are negative quantities)
    medium.sound_speed = np.atleast_1d(medium.sound_speed)

    temp1 = 1 - absorb_eta.min() * kmax ** (medium.alpha_power - 1)

    def kappa(dt):
        return sinc(c_ref * kmax * dt / 2.0)

    def temp2(dt):
        return medium.sound_speed.max() * absorb_tau.min() * kappa(dt) * kmax ** (medium.alpha_power - 1)

    def func_to_solve(dt):
        return (temp2(dt) + np.sqrt((temp2(dt)) ** 2.0 + 4.0 * temp1)) / (temp1 * kmax * kappa(dt) * medium.sound_speed.max())

    dt_start = func_to_solve(0)

    dt_stability_limit = scipy.optimize.fixed_point(func_to_solve, dt_start, xtol=xtol)
    return dt_stability_limit


def _evaluate_non_absorbing_dt_stability_limit(kmax, c_ref, medium: "kWaveMedium") -> float:
    if c_ref >= medium.sound_speed.max():
        # set the timestep to Inf when the model is unconditionally stable
        dt_stability_limit = float("inf")
    else:
        # set the timestep required for stability when c_ref~=max(medium.sound_speed(:))
        dt_stability_limit = 2.0 / (c_ref * kmax) * np.arcsin(c_ref / medium.sound_speed.max())
    return dt_stability_limit


def check_stability(kgrid: "kWaveGrid", medium: "kWaveMedium") -> float:
    """
    Calculates the maximum time step for which the k-space
    propagation models are stable.

    These models are unconditionally
    stable when the reference sound speed is equal to or greater than the
    maximum sound speed in the medium and there is no absorption.
    However, when the reference sound speed is less than the maximum
    sound speed the model is only stable for sufficiently small time
    steps. The criterion is more stringent (the time step is smaller) in
    the absorbing case.

    The time steps given are accurate when the medium properties are
    homogeneous. For a heterogeneous media they give a useful, but not
    exact, estimate.

    Args:
        kgrid: simulation grid
        medium: medium properties

    Returns:
         The maximum time step for which the models are stable. This is set to Inf when the model is unconditionally stable.

    """

    # why? : this function was migrated from Matlab.
    # Matlab would treat the 'medium' as a "pass by value" argument.
    # In python argument is passed by reference and changes in this function will cause original data to be changed.
    # Instead of making significant changes to the function, we make a deep copy of the argument
    medium = deepcopy(medium)

    # find the maximum wavenumber
    kmax = kgrid.k.max()

    # calculate the reference sound speed for the fluid code, using the
    # maximum by default which ensures the model is unconditionally stable
    reductions = {"min": np.min, "max": np.max, "mean": np.mean}

    # TODO: move this logic to medium
    if medium.sound_speed_ref is not None:
        ss_ref = medium.sound_speed_ref
        if isinstance(ss_ref, numbers.Number):
            c_ref = ss_ref
        else:
            try:
                c_ref = reductions[ss_ref](medium.sound_speed)
            except KeyError:
                raise NotImplementedError(f"Unknown value of {ss_ref} for medium.sound_speed_ref.")
    else:
        c_ref = reductions["max"](medium.sound_speed)

    medium.sound_speed = np.atleast_1d(medium.sound_speed)
    # calculate the timesteps required for stability
    if medium.alpha_coeff is None or np.all(medium.alpha_coeff == 0):
        dt_stability_limit = _evaluate_non_absorbing_dt_stability_limit(kmax, c_ref, medium)
    else:
        dt_stability_limit = _evaluate_absorbing_dt_stability_limit(kmax, c_ref, medium)

    return dt_stability_limit


def check_factors(min_number: int, max_number: int) -> None:
    """
    Return the maximum prime factor for a range of numbers.

    checkFactors loops through the given range of numbers and finds the
    numbers with the smallest maximum prime factors. This allows suitable
    grid sizes to be selected to maximise the speed of the FFT (this is
    fastest for FFT lengths with small prime factors). The output is
    printed to the command line.

    Args:
        min_number: integer specifying the lower bound of values to test
        max_number: integer specifying the upper bound of values to test

    """

    # compute the factors and maximum prime factors for each number in the range
    factors = {}
    for n in range(min_number, max_number):
        factors[n] = {"factors": primefactors(n), "max_prime_factor": max(primefactors(n))}

    # print the numbers that match each maximum prime factor
    for factor in [2, 3, 5, 7]:
        logging.log(logging.INFO, f"Numbers with a maximum prime factor of {factor}:")
        for n in range(min_number, max_number):
            if factors[n]["max_prime_factor"] == factor:
                logging.log(logging.INFO, n)


def check_divisible(number: float, divider: float) -> bool:
    """
    Checks whether number is divisible by divider without any remainder
    Why do we need such a function? -> Because due to floating point precision we
    experience rounding errors while using standard modulo operator with floating point numbers

    Args:
        number: Number that's supposed to be divided
        divider: Divider that should devide the number

    Returns:
        True if number is divisible by divider, False otherwise

    """

    result = number / divider
    after_decimal = result % 1
    return after_decimal == 0
