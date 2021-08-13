import numpy as np

def get_pml(Nx, dx, dt, c, pml_size, pml_alpha, staggered, dimension, axisymmetric=False):
    """
        getPML returns a 1D perfectly matched layer variable based on the given size and absorption coefficient.
    Args:
        Nx:
        dx:
        dt:
        c:
        pml_size:
        pml_alpha:
        staggered:
        dimension:
        axisymmetric:

    Returns:

    """
    # define x-axis
    Nx = int(Nx)
    pml_size = int(pml_size)
    x = np.arange(1, pml_size + 1)

    # create absorption profile
    if staggered:

        # calculate the varying components of the pml using a staggered grid
        pml_left  = pml_alpha * (c / dx) * (( ((x + 0.5) - pml_size - 1) / (0 - pml_size) ) ** 4)
        pml_right = pml_alpha * (c / dx) * (( (x + 0.5) / pml_size ) ** 4)

    else:

        # calculate the varying components of the pml using a regular grid
        pml_left  = pml_alpha * (c / dx) * (( (x - pml_size - 1) / (0 - pml_size) ) ** 4)
        pml_right = pml_alpha * (c / dx) * (( x / pml_size ) ** 4)

    # exponentiation
    pml_left  = np.exp(-pml_left * dt / 2)
    pml_right = np.exp(-pml_right * dt / 2)

    # add the components of the pml to the total function, not adding the axial
    # side of the radial PML if axisymmetric
    pml = np.ones((1, Nx))
    if not axisymmetric:
        pml[:, :pml_size] = pml_left

    pml[:, Nx - pml_size:] = pml_right

    # reshape the pml vector to be in the desired direction
    if dimension == 1:
        pml = pml.T
    elif dimension == 3:
        pml = np.reshape(pml, (1, 1, Nx))
    return pml

    # ------------
    # Other forms:
    # ------------
    # Use this to include an extra unity point:
    # pml_left = pml_alpha*(c/dx)* ( (x - pml_size) ./ (1 - pml_size) ).^2;
    # pml_right = pml_alpha*(c/dx)* ( (x - 1) ./ (pml_size - 1) ).^2;
    # Staggered grid equivalents:
    # pml_left = pml_alpha*(c/dx)* ( ((x + 0.5) - pml_size) ./ (1 - pml_size) ).^2;
    # pml_right = pml_alpha*(c/dx)* ( ((x + 0.5) - 1) ./ (pml_size - 1) ).^2;


def getOptimalPMLSize(grid_sz, pml_range=None, axisymmetric=None):
    """
        %     getOptimalPMLSize finds the size of the perfectly matched layer (PML)
        %     that gives an overall grid size with the smallest prime factors when
        %     using the first-order simulation functions in k-Wave with the
        %     optional input 'PMLInside', false. Choosing grid sizes with small
        %     prime factors can have a significant impact on the computational
        %     speed, as the code computes spatial gradients using the fast Fourier
        %     transform (FFT).
    Args:
        grid_sz: Grid size defined as a one (1D), two (2D), or three (3D) element vector. Alternatively, can be an
                    object of the kWaveGrid class defining the Cartesian and k-space grid fields.
        pml_range: Two element vector specifying the minimum and maximum PML size (default = [10, 40]).
        axisymmetric: If using the axisymmetric code, string specifying the radial symmetry. Allowable inputs are 'WSWA'
                        and 'WSWS' (default = ''). This is important as the axisymmetric code only applies to the
                        PML to the outside edge in the radial dimension.
    Returns: PML size that gives the overall grid with the smallest prime factors.

    """
    # check if grid size is given as kgrid, and extract grid size
    from kwave.kgrid import kWaveGrid
    if isinstance(grid_sz, kWaveGrid):
        if grid_sz.dim == 1:
            grid_sz = [grid_sz.Nx]
        if grid_sz.dim == 2:
            grid_sz = [grid_sz.Nx, grid_sz.Ny]
        if grid_sz.dim == 3:
            grid_sz = [grid_sz.Nx, grid_sz.Ny, grid_sz.Nz]

    # assign grid size
    grid_dim = len(grid_sz)

    # check grid size is 1, 2, or 3
    assert 1 <= grid_dim <= 3, 'Grid dimensions must be given as a 1, 2, or 3 element vector.'

    # check for pml_range input
    if pml_range is None:
        pml_range = [10, 40]

    # force integer
    pml_range = np.round(pml_range).astype(int)

    # check for positive values
    assert np.all(pml_range >= 0), 'Optional input pml_range must be positive.'

    # check for correct length
    assert len(pml_range) == 2, 'Optional input pml_range must be a two element vector.'

    # check for monotonic
    assert pml_range[1] > pml_range[0], 'The second value for pml_range must be greater than the first.'

    # check for axisymmetric input
    if axisymmetric is None:
        axisymmetric = False

    # check for correct string
    assert not isinstance(axisymmetric, str) or axisymmetric.startswith(('WSWA', 'WSWS')), \
        "Optional input axisymmetric must be set to ''WSWA'' or ''WSWS''."

    # check for correct dimensions
    if isinstance(axisymmetric, str) and grid_dim != 2:
        raise ValueError('Optional input axisymmetric is only valid for 2D grid sizes.')

    # create array of PML values to search
    pml_size = np.arange(pml_range[0], pml_range[1] + 1)

    # extract largest prime factor for each dimension for each pml size
    facs = np.zeros((grid_dim, len(pml_size)))
    from kwave.utils import largest_prime_factor
    for dim in range(0, grid_dim):
        for index in range(0, len(pml_size)):
            if isinstance(axisymmetric, str) and dim == 2:
                if axisymmetric == 'WSWA':
                    facs[dim, index] = largest_prime_factor((grid_sz[dim] + pml_size[index]) * 4)
                if axisymmetric == 'WSWS':
                    facs[dim, index] = largest_prime_factor((grid_sz[dim] + pml_size[index]) * 2 - 2)
            else:
                facs[dim, index] = largest_prime_factor(grid_sz[dim] + 2 * pml_size[index])

    # get best dimension size
    ind_opt = np.argmin(facs, 1)

    # assign output
    pml_sz = np.zeros((1, grid_dim))
    pml_sz[0] = pml_size[ind_opt[0]]
    if grid_dim > 1:
        pml_sz[1] = pml_size[ind_opt[1]]
    if grid_dim > 2:
        pml_sz[2] = pml_size[ind_opt[2]]

    return pml_sz