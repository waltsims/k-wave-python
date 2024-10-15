import logging

from kwave.data import Vector
from kwave.kgrid import kWaveGrid

import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator

def kspaceLineRecon(
    p,
    dy,
    dt,
    c,
    data_order='ty',
    interp='nearest',
    pos_cond=False
):
    p = p.copy()

    # reorder the data if needed (p_ty)
    if data_order == 'yt':
        p = p.T

    # mirror the time domain data about t = 0 to allow the cosine transform to
    # be computed using an FFT (p_ty)
    p = np.vstack((np.flipud(p), p[1:]))

    # extract the size of mirrored input data
    Nt, Ny = p.shape
    
    # update command line status
    logging.log(logging.INFO, "Running k-Wave line reconstruction...\n"
                              f"grid size: {Ny} by {(Nt + 1) / 2} grid points\n"
                              f"interpolation mode: {interp}")

    # create a computational grid that is evenly spaced in w and ky, where
    # Nx = Nt and dx = dt*c
    N = Vector([Nt, Ny])
    d = Vector([dt * c, dy])
    kgrid = kWaveGrid(N, d)

    # from the grid for kx, create a computational grid for w using the
    # relation dx = dt*c; this represents the initial sampling of p(w, ky)
    w = c * kgrid.kx

    # remap the computational grid for kx onto w using the dispersion
    # relation w/c = (kx^2 + ky^2)^1/2. This gives an w grid that is
    # evenly spaced in kx. This is used for the interpolation from p(w, ky)
    # to p(kx, ky). Only real w is taken to force kx (and thus x) to be
    # symmetrical about 0 after the interpolation.
    w_new = c * kgrid.k

    # calculate the scaling factor using the value of kx, where
    # kx = sqrt( (w/c).^2 - kgrid.ky.^2 ) and then manually
    # replacing the DC value with its limit (otherwise NaN results)
    with np.errstate(divide='ignore', invalid='ignore'):
        sf = c ** 2 * np.emath.sqrt((w / c) ** 2 - kgrid.ky ** 2) / (2 * w)
    sf[(w == 0) & (kgrid.ky == 0)] = c / 2

    # compute the FFT of the input data p(t, y) to yield p(w, ky) and scale
    p = sf * fftshift(fftn(ifftshift(p)))

    # exclude the inhomogeneous part of the wave
    p[np.abs(w) < np.abs(c * kgrid.ky)] = 0

    # compute the interpolation from p(w, ky) to p(kx, ky)and then force to be
    # symmetrical
    interp_func = RegularGridInterpolator(
        (w[:, 0], kgrid.ky[0]),
        p, bounds_error=False, fill_value=0, method=interp
    )
    query_points = np.stack((w_new, kgrid.ky), axis=-1)
    p = interp_func(query_points)

    # compute the inverse FFT of p(kx, ky) to yield p(x, y)
    p = np.real(fftshift(ifftn(ifftshift(p))))

    # remove the left part of the mirrored data which corresponds to the
    # negative part of the mirrored time data
    p = p[(Nt // 2):, :]

    # correct the scaling - the forward FFT is computed with a spacing of dt
    # and the reverse requires a spacing of dy = dt*c, the reconstruction
    # assumes that p0 is symmetrical about y, and only half the plane collects
    # data (first approximation to correcting the limited view problem)
    p = 2 * 2 * p / c

    # enforce positivity condition
    if pos_cond:
        logging.log(logging.INFO, 'applying positivity condition...')
        p[p < 0] = 0

    return p
