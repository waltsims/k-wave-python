import math
import sys
from dataclasses import dataclass

import numpy as np

from kwave.data import Vector, FlexibleVector
from kwave.enums import DiscreteCosine, DiscreteSine
from kwave.utils import matlab
from kwave.utils.math import largest_prime_factor


@dataclass
class kWaveGrid(object):
    """
    kWaveGrid is the grid class used across the k-Wave Toolbox. An object
    of the kWaveGrid class contains the grid coordinates and wavenumber
    matrices used within the simulation and reconstruction functions in
    k-Wave. The grid matrices are indexed as: (x, 1) in 1D; (x, y) in
    2D; and (x, y, z) in 3D. The grid is assumed to be a regularly spaced
    Cartesian grid, with grid spacing given by dx, dy, dz (typically the
    grid spacing in each direction is constant).
    """

    # default CFL number
    CFL_DEFAULT = 0.3

    # machine precision
    MACHINE_PRECISION = 100 * sys.float_info.epsilon

    def __init__(self, N, spacing):
        """
        Args:
            N: grid size in each dimension [grid points]
            spacing: grid point spacing in each direction [m]
        """
        N, spacing = np.atleast_1d(N), np.atleast_1d(spacing)  # if inputs are lists
        assert N.ndim == 1 and spacing.ndim == 1  # ensure no multidimensional lists
        assert (1 <= N.size <= 3) and (1 <= spacing.size <= 3)  # ensure valid dimensionality
        assert N.size == spacing.size, "Size list N and spacing list do not have the same size."

        self.N = N.astype(int)  #: grid size in each dimension [grid points]
        self.spacing = spacing  #: grid point spacing in each direction [m]
        self.dim = self.N.size  #: Number of dimensions (1, 2 or 3)

        self.nonuniform = False  #: flag that indicates grid non-uniformity
        self.dt = "auto"  #: size of time step [s]
        self.Nt = "auto"  #: number of time steps [s]

        # originally there was [xn_vec, yn_vec, zn_vec]
        self.n_vec = FlexibleVector([0] * self.dim)  #: position vectors for the grid points in [0, 1]
        # originally there was [xn_vec_sgx, yn_vec_sgy, zn_vec_sgz]
        self.n_vec_sg = FlexibleVector([0] * self.dim)  #: position vectors for the staggered grid points in [0, 1]

        # originally there was [dxudxn, dyudyn, dzudzn]
        self.dudn = FlexibleVector([0] * self.dim)  #: transformation gradients between uniform and staggered grids
        # originally there was [dxudxn_sgx, dyudyn_sgy, dzudzn_sgz]
        self.dudn_sg = FlexibleVector([0] * self.dim)  #: transformation gradients between uniform and staggered grids

        # assign the grid parameters for the x spatial direction
        # originally kx_vec
        self.k_vec = FlexibleVector([self.makeDim(self.Nx, self.dx)])  #: Nx x 1 vector of wavenumber components in the x-direction [rad/m]

        if self.dim == 1:
            # define the scalar wavenumber based on the wavenumber components
            self.k = abs(self.k_vec.x)  #: scalar wavenumber

        if self.dim >= 2:
            # assign the grid parameters for the x and y spatial directions
            # Ny x 1 vector of wavenumber components in the y-direction [rad/m]
            self.k_vec = self.k_vec.append(self.makeDim(self.Ny, self.dy))

            if self.dim == 2:
                # define the wavenumber based on the wavenumber components
                self.k = np.zeros((self.Nx, self.Ny))
                self.k = np.reshape(self.k_vec.x, (-1, 1)) ** 2 + self.k
                self.k = np.reshape(self.k_vec.y, (1, -1)) ** 2 + self.k
                self.k = np.sqrt(self.k)  #: scalar wavenumber

        if self.dim == 3:
            # assign the grid parameters for the x, y, and z spatial directions
            # Nz x 1 vector of wavenumber components in the z-direction [rad/m]
            self.k_vec = self.k_vec.append(self.makeDim(self.Nz, self.dz))

            # define the wavenumber based on the wavenumber components
            self.k = np.zeros((self.Nx, self.Ny, self.Nz))
            self.k = np.reshape(self.k_vec.x, (-1, 1, 1)) ** 2 + self.k
            self.k = np.reshape(self.k_vec.y, (1, -1, 1)) ** 2 + self.k
            self.k = np.reshape(self.k_vec.z, (1, 1, -1)) ** 2 + self.k
            self.k = np.sqrt(self.k)  #: scalar wavenumber

    @property
    def t_array(self):
        """
        time array [s]
        """
        # TODO (walter): I would change this functionality to return a time array even if Nt or dt are not yet set
        #  (e.g. if they are still 'auto')

        if self.Nt == "auto" or self.dt == "auto":
            return "auto"
        else:
            t_array = np.arange(0, self.Nt) * self.dt
            # TODO: adding this extra dimension seems unnecessary
            # This leads to an extra squeeze when plotting e.g. in example "array as sensor" on lines 110 and 111
            return np.expand_dims(t_array, axis=0)

    @t_array.setter
    def t_array(self, t_array):
        # check for 'auto' input
        if t_array == "auto":
            # set values to auto
            self.Nt = "auto"
            self.dt = "auto"

        else:
            # extract property values
            Nt_temp = t_array.size
            dt_temp = t_array[1] - t_array[0]

            # check the time array begins at zero
            assert t_array[0] == 0, "t_array must begin at zero."

            # check the time array is evenly spaced
            assert (t_array[1:] - t_array[0:-1] - dt_temp).max() < self.MACHINE_PRECISION, "t_array must be evenly spaced."

            # check the time steps are increasing
            assert dt_temp > 0, "t_array must be monotonically increasing."

            # assign values
            self.Nt = Nt_temp
            self.dt = dt_temp

    def setTime(self, Nt, dt) -> None:
        """
        Set Nt and dt based on user input

        Args:
            Nt:
            dt:

        Returns: None
        """
        # check the value for Nt
        assert (
            isinstance(Nt, int) or np.issubdtype(Nt, np.int64) or np.issubdtype(Nt, np.int32)
        ) and Nt > 0, "Nt must be a positive integer."

        # check the value for dt
        assert dt > 0, "dt must be positive."

        # assign values
        self.Nt = Nt
        self.dt = dt

    @property
    def Nx(self):
        """
        grid size in x-direction [grid points]
        """
        return self.N[0]

    @property
    def Ny(self):
        """
        grid size in y-direction [grid points]
        """
        return self.N[1] if self.N.size >= 2 else 0

    @property
    def Nz(self):
        """
        grid size in z-direction [grid points]
        """
        return self.N[2] if self.N.size == 3 else 0

    @property
    def dx(self):
        """
        grid point spacing in x-direction [m]
        """
        return self.spacing[0]

    @property
    def dy(self):
        """
        grid point spacing in y-direction [m]
        """
        return self.spacing[1] if self.spacing.size >= 2 else 0

    @property
    def dz(self):
        """
        grid point spacing in z-direction [m]
        """
        return self.spacing[2] if self.spacing.size == 3 else 0

    @property
    def x_vec(self):
        """
        Nx x 1 vector of the grid coordinates in the x-direction [m]
        """
        # calculate x_vec based on kx_vec
        return self.size[0] * self.k_vec.x * self.dx / (2 * np.pi)

    @property
    def y_vec(self):
        """
        Ny x 1 vector of the grid coordinates in the y-direction [m]
        """
        # calculate y_vec based on ky_vec
        if self.dim < 2:
            return np.nan
        return self.size[1] * self.k_vec.y * self.dy / (2 * np.pi)

    @property
    def z_vec(self):
        """
        Nz x 1 vector of the grid coordinates in the z-direction [m]
        """
        # calculate z_vec based on kz_vec
        if self.dim < 3:
            return np.nan
        return self.size[2] * self.k_vec.z * self.dz / (2 * np.pi)

    @property
    def x(self):
        """
        Nx x Ny x Nz grid containing repeated copies of the grid coordinates in the x-direction [m]
        """
        return self.size[0] * self.kx * self.dx / (2 * math.pi)

    @property
    def y(self):
        """
        Nx x Ny x Nz grid containing repeated copies of the grid coordinates in the y-direction [m]
        """
        if self.dim < 2:
            return 0
        return self.size[1] * self.ky * self.dy / (2 * math.pi)

    @property
    def z(self):
        """
        Nx x Ny x Nz grid containing repeated copies of the grid coordinates in the z-direction [m]
        """
        if self.dim < 3:
            return 0
        return self.size[2] * self.kz * self.dz / (2 * math.pi)

    @property
    def xn(self):
        """
        3D plaid non-uniform spatial grids

        Returns:
            plaid xn matrix
        """
        if self.dim == 1:
            return self.n_vec.x if self.nonuniform else 0
        elif self.dim == 2:
            return np.tile(self.n_vec.x, (1, self.Ny)) if self.nonuniform else 0
        else:
            return np.tile(self.n_vec.x, (1, self.Ny, self.Nz)) if self.nonuniform else 0

    @property
    def yn(self):
        """
        3D plaid non-uniform spatial grids

        Returns:
            plaid yn matrix
        """
        if self.dim < 2:
            return np.nan

        n_vec_y = np.array(self.n_vec.y).T

        if self.dim == 2:
            return np.tile(n_vec_y, (self.Nx, 1)) if self.nonuniform else 0
        else:
            return np.tile(n_vec_y, (self.Nx, 1, self.Nz)) if self.nonuniform else 0

    @property
    def zn(self):
        """
        3D plaid non-uniform spatial grids
        Returns:
            plaid zn matrix
        """
        if self.dim < 3:
            return np.nan
        n_vec_z = np.atleast_1d(np.squeeze(self.n_vec.z))[None, None, :]
        return np.tile(n_vec_z, (self.Nx, self.Ny, 1)) if self.nonuniform else 0

    @property
    def size(self):
        """
        Size of grid in the all directions [m]
        """
        return Vector(self.N * self.spacing)

    @property
    def total_grid_points(self) -> np.ndarray:
        """
        Total number of grid points (equal to Nx * Ny * Nz)
        """
        return np.prod(self.N)

    @property
    def kx(self):
        """
        Nx x Ny x Nz grid containing repeated copies of the wavenumber components in the x-direction [rad/m]

        Returns:
            plaid xn matrix
        """
        if self.dim == 1:
            return self.k_vec.x
        elif self.dim == 2:
            return np.tile(self.k_vec.x, (1, self.Ny))
        else:
            return np.tile(self.k_vec.x[:, :, None], (1, self.Ny, self.Nz))

    @property
    def ky(self):
        """
        Nx x Ny x Nz grid containing repeated copies of the wavenumber components in the y-direction [rad/m]

        Returns:
            plaid yn matrix
        """
        if self.dim == 2:
            return np.tile(self.k_vec.y.T, (self.Nx, 1))
        elif self.dim == 3:
            return np.tile(self.k_vec.y[None, :, :], (self.Nx, 1, self.Nz))
        return np.nan

    @property
    def kz(self):
        """
        Nx x Ny x Nz grid containing repeated copies of the wavenumber components in the z-direction [rad/m]

        Returns:
            plaid zn matrix
        """
        if self.dim == 3:
            return np.tile(self.k_vec.z.T[None, :, :], (self.Nx, self.Ny, 1))
        else:
            return np.nan

    @property
    def x_size(self):
        """
        Size of grid in the x-direction [m]
        """
        return self.Nx * self.dx

    @property
    def y_size(self):
        """
        Size of grid in the y-direction [m]
        """
        return self.Ny * self.dy

    @property
    def z_size(self):
        """
        Size of grid in the z-direction [m]
        """
        return self.Nz * self.dz

    @property
    def k_max(self):  # added by us, not the same as kWave k_max (see k_max_all for KwaveGrid.k_max)
        """
        Maximum supported spatial frequency in the 3 directions [rad/m]

        Returns:
            Vector of 3 elements each in [rad/m]. Value for higher dimensions set to NaN
        """
        #
        kx_max = np.abs(self.k_vec.x).max()
        ky_max = np.abs(self.k_vec.y).max() if self.dim >= 2 else np.nan
        kz_max = np.abs(self.k_vec.z).max() if self.dim == 3 else np.nan
        return Vector([kx_max, ky_max, kz_max])

    @property
    def k_max_all(self):
        """
        Maximum supported spatial frequency in all directions [rad/m]
        Originally k_max in kWave.kWaveGrid!

        Returns:
            Scalar in [rad/m]
        """
        #
        return np.nanmin(self.k_max)

    ########################################
    # functions that can only be accessed by class members
    ########################################
    @staticmethod
    # TODO (walter): convert this name to snake case
    def makeDim(num_points, spacing):
        """
        Create the grid parameters for a single spatial direction

        Args:
            num_points:
            spacing:

        Returns:

        """
        # define the discretisation of the spatial dimension such that there is always a DC component
        if num_points % 2 == 0:
            # grid dimension has an even number of points
            nx = np.arange(-num_points / 2, num_points / 2) / num_points
        else:
            # grid dimension has an odd number of points
            nx = np.arange(-(num_points - 1) / 2, (num_points - 1) / 2 + 1) / num_points
        nx = np.array(nx).T

        # force middle value to be zero in case 1/Nx is a recurring
        # number and the series doesn't give exactly zero
        nx[int(num_points // 2)] = 0

        # define the wavenumber vector components
        res = (2 * math.pi / spacing) * nx
        return res[:, None]

    def highest_prime_factors(self, axisymmetric=None) -> np.ndarray:
        """
        calculate the highest prime factors

        Args:
            axisymmetric: Axisymmetric code or None

        Returns:
            Vector of three elements
        """
        # import statement place here in order to avoid circular dependencies
        if axisymmetric is not None:
            if axisymmetric == "WSWA":
                prime_facs = [largest_prime_factor(self.Nx), largest_prime_factor(self.Ny * 4), largest_prime_factor(self.Nz)]
            elif axisymmetric == "WSWS":
                prime_facs = [largest_prime_factor(self.Nx), largest_prime_factor(self.Ny * 2 - 2), largest_prime_factor(self.Nz)]
            else:
                raise ValueError("Unknown axisymmetric symmetry.")
        else:
            prime_facs = [largest_prime_factor(self.Nx), largest_prime_factor(self.Ny), largest_prime_factor(self.Nz)]
        return np.array(prime_facs)

    # TODO (walter): convert this name to snake case
    def makeTime(self, c, cfl=CFL_DEFAULT, t_end=None):
        """
        Compute Nt and dt based on the cfl number and grid size, where
        the number of time-steps is chosen based on the time it takes to
        travel from one corner of the grid to the geometrically opposite
        corner. Note, if c is given as a matrix, the calculation for dt
        is based on the maximum value, and the calculation for t_end
        based on the minimum value.

        Args:
            c: sound speed
            cfl: convergence condition by Courant–Friedrichs–Lewy
            t_end: final time step

        Returns:
            Nothing
        """
        # if c is a matrix, find the minimum and maximum values
        c = np.array(c)
        c_min, c_max = np.min(c), np.max(c)

        # check for user define t_end, otherwise set the simulation
        # length based on the size of the grid diagonal and the maximum
        # sound speed in the medium
        if t_end is None:
            t_end = np.linalg.norm(self.size, ord=2) / c_min

        # extract the smallest grid spacing
        min_grid_dim = np.min(self.spacing)

        # assign time step based on CFL stability criterion
        self.dt = cfl * min_grid_dim / c_max

        # assign number of time steps based on t_end
        self.Nt = int(np.floor(t_end / self.dt) + 1)

        # catch case where dt is a recurring number
        if (np.floor(t_end / self.dt) != np.ceil(t_end / self.dt)) and (matlab.rem(t_end, self.dt) == 0):
            self.Nt = self.Nt + 1

        return self.t_array, self.dt

    ##################################################
    ####
    #### FUNCTIONS BELOW WERE NOT TESTED FOR CORRECTNESS!
    ####
    ##################################################
    def kx_vec_dtt(self, dtt_type):
        """
        Compute the DTT wavenumber vector in the x-direction

        Args:
            dtt_type:

        Returns:

        """
        kx_vec_dtt, M = self.makeDTTDim(self.Nx, self.dx, dtt_type)
        return kx_vec_dtt, M

    def ky_vec_dtt(self, dtt_type):
        """
        Compute the DTT wavenumber vector in the y-direction

        Args:
            dtt_type:

        Returns:

        """
        ky_vec_dtt, M = self.makeDTTDim(self.Ny, self.dy, dtt_type)
        return ky_vec_dtt, M

    def kz_vec_dtt(self, dtt_type):
        """
        Compute the DTT wavenumber vector in the z-direction

        Args:
            dtt_type:

        Returns:

        """
        kz_vec_dtt, M = self.makeDTTDim(self.Nz, self.dz, dtt_type)
        return kz_vec_dtt, M

    @staticmethod
    # TODO (walter): convert this name to snake case
    def makeDTTDim(Nx, dx, dtt_type):
        """
        Create the DTT grid parameters for a single spatial direction

        Args:
            Nx:
            dx:
            dtt_type:

        Returns:

        """

        # compute the implied period of the input function
        if dtt_type == DiscreteCosine.TYPE_1:
            M = 2 * (Nx - 1)
        elif dtt_type == DiscreteSine.TYPE_1:
            M = 2 * (Nx + 1)
        else:
            M = 2 * Nx

        # calculate the wavenumbers
        if dtt_type == DiscreteCosine.TYPE_1:
            # whole-wavenumber DTT
            # WSWS / DCT-I
            n = np.arange(0, M // 2 + 1).T
            kx_vec = 2 * math.pi * n / (M * dx)
        elif dtt_type == DiscreteCosine.TYPE_2:
            # whole-wavenumber DTT
            # HSHS / DCT-II
            n = np.arange(0, M // 2).T
            kx_vec = 2 * math.pi * n / (M * dx)
        elif dtt_type == DiscreteSine.TYPE_1:
            # whole-wavenumber DTT
            # WAWA / DST-I
            n = np.arange(1, M // 2).T
            kx_vec = 2 * math.pi * n / (M * dx)
        elif dtt_type == DiscreteSine.TYPE_2:
            # whole-wavenumber DTT
            # HAHA / DST-II
            n = np.arange(1, M // 2 + 1).T
            kx_vec = 2 * math.pi * n / (M * dx)
        elif dtt_type in [DiscreteCosine.TYPE_3, DiscreteCosine.TYPE_4, DiscreteSine.TYPE_3, DiscreteSine.TYPE_4]:
            # half-wavenumber DTTs
            # WSWA / DCT-III
            # HSHA / DCT-IV
            # WAWS / DST-III
            # HAHS / DST-IV
            n = np.arange(0, M // 2).T
            kx_vec = 2 * math.pi * (n + 0.5) / (M * dx)
        else:
            raise ValueError

        return kx_vec, M

    ########################################
    # functions for non-uniform grids
    ########################################
    # TODO (walter): convert this name to snake case
    def setNUGrid(self, dim, n_vec, dudn, n_vec_sg, dudn_sg):
        """
        Function to set non-uniform grid parameters in specified dimension

        Args:
            dim:
            n_vec:
            dudn:
            n_vec_sg:
            dudn_sg:

        Returns:

        """

        # check the dimension to set the nonuniform grid is appropriate
        assert dim <= self.dim, f"Cannot set nonuniform parameters for dimension {dim} of {self.dim}-dimensional grid."

        # force non-uniform grid spacing to be column vectors, and the
        # gradients to be in the correct direction for use with bsxfun
        n_vec = np.reshape(n_vec, (-1, 1), order="F")
        n_vec_sg = np.reshape(n_vec_sg, (-1, 1), order="F")

        if dim == 1:
            dudn = np.reshape(dudn, (-1, 1), order="F")
            dudn_sg = np.reshape(dudn_sg, (-1, 1), order="F")
        elif dim == 2:
            dudn = np.reshape(dudn, (1, -1), order="F")
            dudn_sg = np.reshape(dudn_sg, (1, -1), order="F")
        elif dim == 3:
            dudn = np.reshape(dudn, (1, 1, -1), order="F")
            dudn_sg = np.reshape(dudn_sg, (1, 1, -1), order="F")

        self.n_vec.assign_dim(self.dim, n_vec)
        self.n_vec_sg.assign_dim(self.dim, n_vec_sg)

        self.dudn.assign_dim(self.dim, dudn)
        self.dudn_sg.assign_dim(self.dim, dudn_sg)

        # set non-uniform flag
        self.nonuniform = True

    def k_dtt(self, dtt_type):  # Not tested for correctness!
        """
        compute the individual wavenumber vectors, where dtt_type is the
        type of discrete trigonometric transform, which corresponds to
        the assumed input symmetry of the input function, where:

        1. DCT-I    WSWS
        2. DCT-II   HSHS
        3. DCT-III  WSWA
        4. DCT-IV   HSHA
        5. DST-I    WAWA
        6. DST-II   HAHA
        7. DST-III  WAWS
        8. DST-IV   HAHS

        Args:
            dtt_type:

        Returns:

        """
        # check dtt_type is a scalar or a vector the same size self.dim
        dtt_type = np.array(dtt_type)
        assert dtt_type.size in [1, self.dim], f"dtt_type must be a scalar, or {self.dim}D vector"
        if self.dim == 1:
            k, M = self.kx_vec_dtt(dtt_type[0])
            return k, M
        elif self.dim == 2:
            # assign the grid parameters for the x and y spatial directions
            kx_vec_dtt, Mx = self.kx_vec_dtt(dtt_type[0])
            ky_vec_dtt, My = self.ky_vec_dtt(dtt_type[-1])

            # define the wavenumber based on the wavenumber components
            k = np.zeros((self.Nx, self.Ny))
            # assert len(kx_vec_dtt.shape) == 3
            k += np.reshape(kx_vec_dtt, (-1, 1)) ** 2
            k += np.reshape(ky_vec_dtt, (1, -1)) ** 2
            k = np.sqrt(k)

            # define product of implied period
            M = Mx * My
            return k, M
        elif self.dim == 3:
            # assign the grid parameters for the x, y, and z spatial directions
            kx_vec_dtt, Mx = self.kx_vec_dtt(dtt_type[0])
            ky_vec_dtt, My = self.ky_vec_dtt(dtt_type[len(dtt_type) // 2])
            kz_vec_dtt, Mz = self.kz_vec_dtt(dtt_type[-1])

            # define the wavenumber based on the wavenumber components
            k = np.zeros((self.Nx, self.Ny, self.Nz))
            k = np.reshape(kx_vec_dtt, (-1, 1, 1)) ** 2 + k
            k = np.reshape(ky_vec_dtt, (1, -1, 1)) ** 2 + k
            k = np.reshape(kz_vec_dtt, (1, 1, -1)) ** 2 + k
            k = np.sqrt(k)

            # define product of implied period
            M = Mx * My * Mz
            return k, M
