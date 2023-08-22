import numpy as np
from scipy.interpolate import interpn


def numDim(arr):
    return len(arr.shape)


if numDim(rho0) == 2 and flags.use_sg:
    rho0_sgx = interpn((kgrid.x, kgrid.y), rho0, (kgrid.x + kgrid.dx / 2, kgrid.y), method='linear')
    rho0_sgy = interpn((kgrid.x, kgrid.y), rho0, (kgrid.x, kgrid.y + kgrid.dy / 2), method='linear')

    rho0_sgx[np.isnan(rho0_sgx)] = rho0[np.isnan(rho0_sgx)]
    rho0_sgy[np.isnan(rho0_sgy)] = rho0[np.isnan(rho0_sgy)]

else:
    rho0_sgx = rho0
    rho0_sgy = rho0

rho0_sgx_inv = 1. / rho0_sgx
rho0_sgy_inv = 1. / rho0_sgy
