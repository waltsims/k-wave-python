import logging

from kwave.data import Vector
from kwave.kgrid import kWaveGrid

import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator


def kspacePlaneRecon(
        p,
        dy,
        dz,
        dt,
        c,
        data_order='tyz',
        interp='nearest',
        pos_cond=False
):
    p = p.copy()

    # reorder the data to p(t, y, z) if needed
    if data_order == 'yzt':
        p = np.transpose(p, (2, 0, 1))

    # mirror the time domain data about t = 0 to allow the cosine transform in
    # the t direction to be computed using an FFT
    p = np.concatenate((np.flip(p, axis=0), p[1:, :, :]), axis=0)

    # extract the size of mirrored input data
    Nt, Ny, Nz = p.shape

    # update command line status
    logging.log(logging.INFO, "Running k-Wave planar reconstruction...\n"
                              f"grid size: {(Nt + 1) // 2} by {Ny} by {Nz} grid points\n"
                              f"interpolation mode: {interp}")

    # create a computational grid that is evenly spaced in w, ky, and kz, where
    # Nx = Nt and dx = dt*c
    N = Vector([Nt, Ny, Nz])
    d = Vector([dt * c, dy, dz])
    kgrid = kWaveGrid(N, d)

    # from the grid for kx, create a computational grid for w using the
    # relation dx = dt*c; this represents the initial sampling of p(w, ky, kz)
    w = c * kgrid.kx

    # remap the computational grid for kx onto w using the dispersion
    # relation w/c = (kx^2 + ky^2 + kz^2)^1/2. This gives an w grid that is
    # evenly spaced in kx. This is used for the interpolation from p(w, ky, kz)
    # to p(kx, ky, kz). Only real w is taken to force kx (and thus x) to be
    # symmetrical about 0 after the interpolation.
    w_new = c * kgrid.k

    # calculate the scaling factor using the value of kx, where
    # kx = sqrt( (w/c)^2 - kgrid.ky^2 - kgrid.kz^2 ) and then manually
    # replacing the DC value with its limit (otherwise NaN results)
    with np.errstate(divide='ignore', invalid='ignore'):
        sf = c ** 2 * np.emath.sqrt((w / c) ** 2 - kgrid.ky ** 2 - kgrid.kz ** 2) / (2 * w)
    sf[(w == 0) & (kgrid.ky == 0) & (kgrid.kz == 0)] = c / 2

    # compute the FFT of the input data p(t, y, z) to yield p(w, ky, kz) and scale
    p = sf * fftshift(fftn(ifftshift(p)))

    # exclude the inhomogeneous part of the wave
    p[np.abs(w) < (c * np.sqrt(kgrid.ky ** 2 + kgrid.kz ** 2))] = 0

    # compute the interpolation from p(w, ky, kz) to p(kx, ky, kz)
    interp_func = RegularGridInterpolator(
        (w[:, 0, 0], kgrid.ky[0,:, 0], kgrid.kz[0, 0, :]),
        p, bounds_error=False, fill_value=0, method=interp)
    query_points = np.stack((w_new, kgrid.ky, kgrid.kz), axis=-1)
    p = interp_func(query_points)

    # compute the inverse FFT of p(kx, ky, kz) to yield p(x, y, z)
    p = np.real(fftshift(ifftn(ifftshift(p))))

    # remove the left part of the mirrored data which corresponds to the
    # negative part of the mirrored time data
    p = p[(Nt // 2):,]

    # correct the scaling - the forward FFT is computed with a spacing of dt
    # and the reverse requires a spacing of dz = dt*c, the reconstruction
    # assumes that p0 is symmetrical about z, and only half the plane collects
    # data (first approximation to correcting the limited view problem)
    p = 2 * 2 * p / c

    # enforce positivity condition
    if pos_cond:
        logging.log(logging.INFO, 'applying positivity condition...')
        p[p < 0] = 0

    return p