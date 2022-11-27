import platform
from copy import deepcopy
from typing import List

import numpy as np

from .conversion import db2neper
from .math import sinc, primefactors


def enforce_fields(dictionary, *fields):
    # from kwave
    for f in fields:
        assert f in dictionary.keys(), [f'The field {f} must be defined in the given dictionary']


def enforce_fields_obj(obj, *fields):
    # from kwave
    for f in fields:
        assert getattr(obj, f) is not None, f'The field {f} must be not None in the given object'


def check_field_names(dictionary, *fields):
    # from kwave
    for k in dictionary.keys():
        assert k in fields, f'The field {k} is not a valid field for the given dictionary'


def num_dim(x):
    # get the size collapsing any singleton dimensions
    return len(x.squeeze().shape)


def num_dim2(x: np.ndarray):
    # get the size collapsing any singleton dimensions
    sz = np.squeeze(x).shape

    if len(sz) > 2:
        return len(sz)
    else:
        return np.sum(np.array(sz) > 1)


def check_str_eq(value, target: str):
    """
        String equality check only if the value is string. Helps to avoid FutureWarnings when value is not a string.
        Added by @Farid
    Args:
        value:
        target:

    Returns:

    """
    return isinstance(value, str) and value == target


def check_str_in(value, target: List[str]):
    """
        Check if value is in the given list only if the value is string.
        Helps to avoid FutureWarnings when value is not a string.
        Added by @Farid
    Args:
        value:
        target:

    Returns:

    """
    # added by Farid
    return isinstance(value, str) and value in target


def is_number(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return False
    if value.dtype in [np.float32, np.float64]:
        return True
    return np.issubdtype(np.array(value), np.number)


def is_unix():
    return platform.system() in ['Linux', 'Darwin']


def check_stability(kgrid, medium):
    """
          checkStability calculates the maximum time step for which the k-space
          propagation models kspaceFirstOrder1D, kspaceFirstOrder2D and
          kspaceFirstOrder3D are stable. These models are unconditionally
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
        kgrid: k-Wave grid object return by kWaveGrid
        medium: structure containing the medium properties

    Returns: the maximum time step for which the models are stable.
            This is set to Inf when the model is unconditionally stable.
    """
    # why? : this function was migrated from Matlab.
    # Matlab would treat the 'medium' as a "pass by value" argument.
    # In python argument is passed by reference and changes in this function will cause original data to be changed.
    # Instead of making significant changes to the function, we make a deep copy of the argument
    medium = deepcopy(medium)

    # define literals
    FIXED_POINT_ACCURACY = 1e-12

    # find the maximum wavenumber
    kmax = kgrid.k.max()

    # calculate the reference sound speed for the fluid code, using the
    # maximum by default which ensures the model is unconditionally stable
    reductions = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean
    }

    if medium.sound_speed_ref is not None:
        ss_ref = medium.sound_speed_ref
        if np.isscalar(ss_ref):
            c_ref = ss_ref
        else:
            try:
                c_ref = reductions[ss_ref](medium.sound_speed)
            except KeyError:
                raise NotImplementedError('Unknown input for medium.sound_speed_ref.')
    else:
        c_ref = reductions['max'](medium.sound_speed)

    # calculate the timesteps required for stability
    if medium.alpha_coeff is None or np.all(medium.alpha_coeff == 0):

        # =====================================================================
        # NON-ABSORBING CASE
        # =====================================================================

        medium.sound_speed = np.atleast_1d(medium.sound_speed)
        if c_ref >= medium.sound_speed.max():
            # set the timestep to Inf when the model is unconditionally stable
            dt_stability_limit = float('inf')

        else:
            # set the timestep required for stability when c_ref~=max(medium.sound_speed(:))
            dt_stability_limit = 2 / (c_ref * kmax) * np.asin(c_ref / medium.sound_speed.max())

    else:

        # =====================================================================
        # ABSORBING CASE
        # =====================================================================

        # convert the absorption coefficient to nepers.(rad/s)^-y.m^-1
        medium.alpha_coeff = db2neper(medium.alpha_coeff, medium.alpha_power)

        # calculate the absorption constant
        if medium.alpha_mode == 'no_absorption':
            absorb_tau = -2 * medium.alpha_coeff * medium.sound_speed ** (medium.alpha_power - 1)
        else:
            absorb_tau = np.array([0])

        # calculate the dispersion constant
        if medium.alpha_mode == 'no_dispersion':
            absorb_eta = 2 * medium.alpha_coeff * medium.sound_speed ** medium.alpha_power * np.tan(
                np.pi * medium.alpha_power / 2)
        else:
            absorb_eta = np.array([0])

        # estimate the timestep required for stability in the absorbing case by
        # assuming the k-space correction factor, kappa = 1 (note that
        # absorb_tau and absorb_eta are negative quantities)
        medium.sound_speed = np.atleast_1d(medium.sound_speed)

        temp1 = medium.sound_speed.max() * absorb_tau.min() * kmax ** (medium.alpha_power - 1)
        temp2 = 1 - absorb_eta.min() * kmax ** (medium.alpha_power - 1)
        dt_estimate = (temp1 + np.sqrt(temp1 ** 2 + 4 * temp2)) / (temp2 * kmax * medium.sound_speed.max())

        # use a fixed point iteration to find the correct timestep, assuming
        # now that kappa = kappa(dt), using the previous estimate as a starting
        # point

        # first define the function to iterate
        def kappa(dt):
            return sinc(c_ref * kmax * dt / 2)

        def temp3(dt):
            return medium.sound_speed.max() * absorb_tau.min() * kappa(dt) * kmax ** (medium.alpha_power - 1)

        def func_to_solve(dt):
            return (temp3(dt) + np.sqrt((temp3(dt)) ** 2 + 4 * temp2)) / (
                    temp2 * kmax * kappa(dt) * medium.sound_speed.max())

        # run the fixed point iteration
        dt_stability_limit = dt_estimate
        dt_old = 0
        while abs(dt_stability_limit - dt_old) > FIXED_POINT_ACCURACY:
            dt_old = dt_stability_limit
            dt_stability_limit = func_to_solve(dt_stability_limit)

    return dt_stability_limit


def check_factors(min_number, max_number):
    """
        Return the maximum prime factor for a range of numbers.

        checkFactors loops through the given range of numbers and finds the
        numbers with the smallest maximum prime factors. This allows suitable
        grid sizes to be selected to maximise the speed of the FFT (this is
        fastest for FFT lengths with small prime factors). The output is
        printed to the command line, and a plot of the factors is generated.

    Args:
        min_number: integer specifying the lower bound of values to test
        max_number: integer specifying the upper bound of values to test

    Returns:

    """

    # extract factors
    facs = np.zeros(1, max_number - min_number)
    fac_max = facs
    for index in range(min_number, max_number):
        facs[index - min_number + 1] = len(primefactors(index))
        fac_max[index - min_number + 1] = max(primefactors(index))

    # compute best factors in range
    print('Numbers with a maximum prime factor of 2')
    ind = min_number + np.argwhere(fac_max == 2)
    print(ind)
    print('Numbers with a maximum prime factor of 3')
    ind = min_number + np.argwhere(fac_max == 3)
    print(ind)
    print('Numbers with a maximum prime factor of 5')
    ind = min_number + np.argwhere(fac_max == 5)
    print(ind)
    print('Numbers with a maximum prime factor of 7')
    ind = min_number + np.argwhere(fac_max == 7)
    print(ind)
    print('Numbers to avoid (prime numbers)')
    nums = np.arange(min_number, max_number)
    print(nums[fac_max == nums])


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
