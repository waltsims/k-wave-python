import logging
import math

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.conversion import db2neper


def create_absorption_variables(kgrid: kWaveGrid, medium: kWaveMedium, equation_of_state):
    # define the lossy derivative operators and proportionality coefficients
    if equation_of_state == "absorbing":
        return create_absorbing_medium_variables(kgrid.k, medium)
    elif equation_of_state == "stokes":
        return create_stokes_medium_variables(medium)
    else:
        raise NotImplementedError


def create_absorbing_medium_variables(kgrid_k, medium: kWaveMedium):
    # convert the absorption coefficient to nepers.(rad/s)^-y.m^-1
    medium.alpha_coeff = db2neper(medium.alpha_coeff, medium.alpha_power)

    absorb_nabla1, absorb_tau = get_absorbtion(kgrid_k, medium)
    absorb_nabla2, absorb_eta = get_dispersion(kgrid_k, medium)

    absorb_nabla1, absorb_nabla2 = apply_alpha_filter(medium, absorb_nabla1, absorb_nabla2)

    return absorb_nabla1, absorb_nabla2, absorb_tau, absorb_eta


def create_stokes_medium_variables(medium: kWaveMedium):
    # convert the absorption coefficient to nepers.(rad/s)^-2.m^-1
    medium.alpha_coeff = db2neper(medium.alpha_coeff, 2)

    # compute the absorbing coefficient
    absorb_tau = compute_absorbing_coeff(medium)
    return None, None, absorb_tau, None


def get_absorbtion(kgrid_k, medium):
    # compute the absorbing fractional Laplacian operator and coefficient
    if medium.alpha_mode == "no_absorption":
        return 0, 0

    nabla1 = kgrid_k ** (medium.alpha_power - 2)
    nabla1[np.isinf(nabla1)] = 0
    nabla1 = np.fft.ifftshift(nabla1)
    tau = compute_absorbing_coeff(medium)
    return nabla1, tau


def get_dispersion(kgrid_k, medium):
    # compute the dispersive fractional Laplacian operator and coefficient
    if medium.alpha_mode == "no_dispersion":
        return 0, 0
    nabla2 = kgrid_k ** (medium.alpha_power - 1)
    nabla2[np.isinf(nabla2)] = 0
    nabla2 = np.fft.ifftshift(nabla2)
    eta = compute_dispersive_coeff(medium)
    return nabla2, eta


def compute_absorbing_coeff(medium):
    tau = -2 * medium.alpha_coeff * (medium.sound_speed ** (medium.alpha_power - 1))

    # modify the sign of the absorption operator if alpha_sign is defined
    # (this is used for time-reversal photoacoustic image reconstruction
    # with absorption compensation)
    if medium.alpha_sign is not None:
        tau = np.sign(medium.alpha_sign[0]) * tau
    return tau


def compute_dispersive_coeff(medium):
    eta = 2 * medium.alpha_coeff * medium.sound_speed ** (medium.alpha_power) * math.tan(math.pi * medium.alpha_power / 2)

    # modify the sign of the absorption operator if alpha_sign is defined
    # (this is used for time-reversal photoacoustic image reconstruction
    # with absorption compensation)
    if medium.alpha_sign is not None:
        eta = np.sign(medium.alpha_sign[1]) * eta
    return eta


def apply_alpha_filter(medium, nabla1, nabla2):
    # pre-filter the absorption parameters if alpha_filter is defined (this
    # is used for time-reversal photoacoustic image reconstruction
    # with absorption compensation)
    if medium.alpha_filter is None:
        return nabla1, nabla2

    # update command line status
    logging.log(logging.INFO, "  filtering absorption variables...")

    # frequency shift the absorption parameters
    nabla1 = np.fft.fftshift(nabla1)
    nabla2 = np.fft.fftshift(nabla2)

    # apply the filter
    nabla1 = nabla1 * medium.alpha_filter
    nabla2 = nabla2 * medium.alpha_filter

    # shift the parameters back
    nabla1 = np.fft.ifftshift(nabla1)
    nabla2 = np.fft.ifftshift(nabla2)
    return nabla1, nabla2
