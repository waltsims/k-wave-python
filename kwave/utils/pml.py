import numpy as np

from kwave.utils.math import largest_prime_factor


def get_pml(
    Nx: int, dx: float, dt: float, c: float, pml_size: int, pml_alpha: float, staggered: bool, dimension: int, axisymmetric: bool = False
) -> np.ndarray:
    """
    Returns a 1D perfectly matched layer variable based on the given size and absorption coefficient.

    This function calculates a 1D perfectly matched layer (PML) variable based on the specified size and absorption coefficient.
    It uses the given parameters to create an absorption profile, which is then exponentiated and reshaped in the desired direction.
    If the axisymmetric argument is set to True, the axial side of the radial PML will not be added.

    Args:
        Nx: The number of grid points in the x direction.
        dx: The spacing between grid points in the x direction.
        dt: The time step size.
        c: The wave speed in the medium.
        pml_size: The size of the PML layer in grid points.
        pml_alpha: The absorption coefficient of the PML layer.
        staggered: Whether to use a staggered grid for calculating the varying components of the PML.
        dimension: The dimension of the PML (1, 2, or 3).
        axisymmetric: Whether to use axisymmetry when calculating the PML. Defaults to False.

    Returns:
        A 1D numpy array representing the PML variable.
    """
    # define x-axis
    Nx = int(Nx)
    pml_size = int(pml_size)
    x = np.arange(1, pml_size + 1)

    # create absorption profile
    if staggered:
        pml_left = pml_alpha * (c / dx) * ((((x + 0.5) - pml_size - 1) / (0 - pml_size)) ** 4)
        pml_right = pml_alpha * (c / dx) * (((x + 0.5) / pml_size) ** 4)
    else:
        pml_left = pml_alpha * (c / dx) * (((x - pml_size - 1) / (0 - pml_size)) ** 4)
        pml_right = pml_alpha * (c / dx) * ((x / pml_size) ** 4)

    # exponentiate and add the components of the pml to the total function
    pml_left = np.exp(-pml_left * dt / 2)
    pml_right = np.exp(-pml_right * dt / 2)
    pml = np.ones((1, Nx))
    if not axisymmetric:
        pml[:, :pml_size] = pml_left
    pml[:, Nx - pml_size :] = pml_right

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


def get_optimal_pml_size(grid_size, pml_range=None, axisymmetric=None):
    """
     get_optimal_pml_size finds the size of the perfectly matched layer (PML)
     that gives an overall grid size with the smallest prime factors when
     using the first-order simulation functions in k-Wave with the
     optional input 'PMLInside', false. Choosing grid sizes with small
     prime factors can have a significant impact on the computational
     speed, as the code computes spatial gradients using the fast Fourier
     transform (FFT).
    Args:
        grid_size:      Grid size defined as a one (1D), two (2D), or three (3D) element vector. Alternatively, can be an
                        object of the kWaveGrid class defining the Cartesian and k-space grid fields.
        pml_range:      Two element vector specifying the minimum and maximum PML size (default = [10, 40]).
        axisymmetric:   If using the axisymmetric code, string specifying the radial symmetry. Allowable inputs are 'WSWA'
                        and 'WSWS' (default = ''). This is important as the axisymmetric code only applies to the
                        PML to the outside edge in the radial dimension.

    Returns:
         PML size that gives the overall grid with the smallest prime factors.

    """
    # check if grid size is given as kgrid, and extract grid size
    from kwave.kgrid import kWaveGrid

    if isinstance(grid_size, kWaveGrid):
        grid_size = grid_size.N

    # assign grid size
    grid_dim = len(grid_size)

    # check grid size is 1, 2, or 3
    assert 1 <= grid_dim <= 3, "Grid dimensions must be given as a 1, 2, or 3 element vector."

    # check for pml_range input
    if pml_range is None:
        pml_range = [10, 40]

    # force integer
    pml_range = np.round(pml_range).astype(int)

    # check for positive values
    assert np.all(pml_range >= 0), "Optional input pml_range must be positive."

    # check for correct length
    assert len(pml_range) == 2, "Optional input pml_range must be a two element vector."

    # check for monotonic
    assert pml_range[1] > pml_range[0], "The second value for pml_range must be greater than the first."

    # check for axisymmetric input
    if axisymmetric is None:
        axisymmetric = False

    # check for correct string
    assert not isinstance(axisymmetric, str) or axisymmetric.startswith(
        ("WSWA", "WSWS")
    ), "Optional input axisymmetric must be set to ''WSWA'' or ''WSWS''."

    # check for correct dimensions
    if isinstance(axisymmetric, str) and grid_dim != 2:
        raise ValueError("Optional input axisymmetric is only valid for 2D grid sizes.")

    # create array of PML values to search
    pml_size = np.arange(pml_range[0], pml_range[1] + 1)

    # extract the largest prime factor for each dimension for each pml size
    facs = np.zeros((grid_dim, len(pml_size)))
    for dim in range(0, grid_dim):
        for index in range(0, len(pml_size)):
            if isinstance(axisymmetric, str) and dim == 2:
                if axisymmetric == "WSWA":
                    facs[dim, index] = largest_prime_factor((grid_size[dim] + pml_size[index]) * 4)
                if axisymmetric == "WSWS":
                    facs[dim, index] = largest_prime_factor((grid_size[dim] + pml_size[index]) * 2 - 2)
            else:
                facs[dim, index] = largest_prime_factor(grid_size[dim] + 2 * pml_size[index])

    # get best dimension size
    ind_opt = np.argmin(facs, 1)

    # assign output
    pml_sz = pml_size[ind_opt]

    return pml_sz
