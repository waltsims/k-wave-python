import numpy as np
from scipy.interpolate import interpn


def calculate_rho0_sgx_sgy(rho0, kgrid, flags, numDim_func):
    if numDim_func(rho0) == 2 and flags['use_sg']:
        rho0_sgx = interpn((kgrid['x'], kgrid['y']), rho0, (kgrid['x'] + kgrid['dx'] / 2, kgrid['y']), method='linear')
        rho0_sgy = interpn((kgrid['x'], kgrid['y']), rho0, (kgrid['x'], kgrid['y'] + kgrid['dy'] / 2), method='linear')

        rho0_sgx[np.isnan(rho0_sgx)] = rho0[np.isnan(rho0_sgx)]
        rho0_sgy[np.isnan(rho0_sgy)] = rho0[np.isnan(rho0_sgy)]
    else:
        rho0_sgx = rho0
        rho0_sgy = rho0

    rho0_sgx_inv = 1.0 / rho0_sgx
    rho0_sgy_inv = 1.0 / rho0_sgy

    return rho0_sgx, rho0_sgy, rho0_sgx_inv, rho0_sgy_inv
