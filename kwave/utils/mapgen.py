import math
import warnings
from math import floor
from typing import Tuple, Optional, Union, List, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize

from .conversion import db2neper, neper2db
from .data import scale_SI
from .matlab import matlab_assign, matlab_find, ind2sub, sub2ind
from .matrix import max_nd
from .tictoc import TicToc
from ..data import Vector


def get_spaced_points(start: float, stop: float, n: int = 100, spacing: str = 'linear') -> np.ndarray:
    """
    Generate a row vector of either logarithmically or linearly spaced points between `start` and `stop`.

    When `spacing` is set to 'linear', the function is identical to the inbuilt `np.linspace` function.
    When `spacing` is set to 'log', the function is similar to the inbuilt `np.logspace` function, except
    that `start` and `stop` define the start and end numbers, not decades. For logarithmically spaced
    points, `start` must be > 0. If `n` < 2, `stop` is returned.

    Args:
        start: start value for the spaced points
        stop: end value for the spaced points
        n: number of points to generate
        spacing: type of spacing to use, either 'linear' or 'log'

    Returns:
        points: row vector of spaced points

    Raises:
        ValueError: if `stop` <= `start` or `spacing` is not 'linear' or 'log'

    """

    if stop <= start:
        raise ValueError('`stop` must be larger than `start`.')

    if spacing == 'linear':
        return np.linspace(start, stop, num=n)
    elif spacing == 'log':
        return np.geomspace(start, stop, num=n)
    else:
        raise ValueError(f"`spacing` {spacing} is not a valid argument. Choose from 'linear' or 'log'.")


def fit_power_law_params(a0: float, y: float, c0: float, f_min: float, f_max: float, plot_fit: bool = False) -> Tuple[
    float, float]:
    """
    Calculate absorption parameters that fit a power law over a given frequency range.

    This function calculates the absorption parameters that should be defined in the simulation functions
    to achieve the desired power law absorption behavior defined by `a0` and `y`. This takes into account
    the actual absorption behavior exhibited by the fractional Laplacian wave equation.

    This fitting is required when using large absorption values or high frequencies, as the fractional
    Laplacian wave equation solved in `kspaceFirstOrderND` and `kspaceSecondOrder` no longer encapsulates
    absorption of the form `a = a0*f^y`.

    The returned values should be used to define `medium.alpha_coeff` and `medium.alpha_power` within the
    simulation functions. The absorption behavior over the frequency range `f_min`:`f_max` will then
    follow the power law defined by `a0` and `y`.

    Args:
        a0: coefficient in the power law absorption equation
        y: exponent in the power law absorption equation
        c0: speed of sound in the medium
        f_min: minimum frequency in the range to fit the power law
        f_max: maximum frequency in the range to fit the power law
        plot_fit: whether to plot the fit

    Returns:
        A tuple of the absorption coefficient and fitted exponent of the power law absorption equation.

    """

    # define frequency axis
    f = get_spaced_points(f_min, f_max, 200)
    w = 2 * np.pi * f
    # convert user defined a0 to Nepers/((rad/s)^y m)
    a0_np = db2neper(a0, y)

    desired_absorption = a0_np * w ** y

    def abs_func(trial_vals):
        """Second-order absorption error"""
        a0_np_trial, y_trial = trial_vals

        actual_absorption = a0_np_trial * w ** y_trial / (1 - (y_trial + 1) *
                                                          a0_np_trial * c0 * np.tan(np.pi * y_trial / 2) * w ** (
                                                                  y_trial - 1))

        absorption_error = np.sqrt(np.sum((desired_absorption - actual_absorption) ** 2))

        return absorption_error

    a0_np_fit, y_fit = optimize.fmin(abs_func, [a0_np, y])

    a0_fit = neper2db(a0_np_fit, y_fit)

    if plot_fit:
        raise NotImplementedError

    return a0_fit, y_fit


def power_law_kramers_kronig(w: np.ndarray, w0: float, c0: float, a0: float, y: float) -> np.ndarray:
    """
    Compute the variation in sound speed for an attenuating medium using the Kramers-Kronig for power law attenuation.

    This function computes the variation in sound speed for an attenuating medium using the Kramers-Kronig
    formula for power law attenuation, where `att = a0 * w^y`. The power law parameters must be in Nepers/m,
    with the frequency in rad/s. The variation is given about the sound speed `c0` at a reference frequency `w0`.

    Args:
        w: input frequency array [rad/s]
        w0: reference frequency [rad/s]
        c0: sound speed at w0 [m/s]
        a0: power law coefficient [Nepers/((rad/s)^y m)]
        y: power law exponent, where 0 < y < 3

    Returns:
        Variation of sound speed with w [m/s]

    """

    if 0 >= y or y >= 3:
        warnings.warn("y must be within the interval (0,3)", UserWarning)
        c_kk = c0 * np.ones_like(w)
    elif y == 1:
        # Kramers-Kronig for y = 1
        c_kk = 1 / (1 / c0 - 2 * a0 * np.log(w / w0) / np.pi)
    else:
        # Kramers-Kronig for 0 < y < 1 and 1 < y < 3
        c_kk = 1 / (1 / c0 + a0 * np.tan(y * np.pi / 2) * (w ** (y - 1) - w0 ** (y - 1)))

    return c_kk


def water_absorption(f, temp):
    """
    Calculates the ultrasonic absorption in distilled
    water at a given temperature and frequency using a 7 th order
    polynomial fitted to the data given by np.pinkerton(1949).


    Args:
        f:   f frequency value [MHz]
        T:   water temperature value [degC]

    Returns:
        abs:  absorption [dB / cm]

    Examples:
        >>> abs = waterAbsorption(f, T)

    References:
        [1] np.pinkerton(1949) "The Absorption of Ultrasonic Waves in Liquids
        and its Relation to Molecular Constitution, " Proceedings of the
        Physical Society.Section B, 2, 129 - 141

    """

    NEPER2DB = 8.686
    # check temperature is within range
    if not 0 <= temp <= 60:
        raise Warning("Temperature outside range of experimental data")

    # conversion factor between Nepers and dB NEPER2DB = 8.686;
    # coefficients for 7th order polynomial fit
    a = [56.723531840522710, -2.899633796917384, 0.099253401567561, -0.002067402501557, 2.189417428917596e-005,
         -6.210860973978427e-008, -6.402634551821596e-010, 3.869387679459408e-012]

    # compute absorption
    a_on_fsqr = (a[0] + a[1] * temp + a[2] * temp ** 2 + a[3] * temp ** 3 + a[4] * temp ** 4 + a[5] * temp ** 5 + a[
        6] * temp ** 6 + a[7] * temp ** 7) * 1e-17

    abs = NEPER2DB * 1e12 * f ** 2 * a_on_fsqr
    return abs


def water_sound_speed(temp: float) -> float:
    """
    Calculate the sound speed in distilled water with temperature.

    Args:
        temp: The temperature of the water in degrees Celsius.

    Returns:
        c: The sound speed in distilled water in m/s.

    Raises:
        ValueError: if `temp` is not between 0 and 95

    References:
        Marczak, R. (1997). The sound velocity in water as a function of temperature. Journal of Research of the National Institute of Standards and Technology, 102(6), 561-567.

    """

    # check limits
    if not (0 <= temp <= 95):
        raise ValueError("`temp` must be between 0 and 95.")

    # find value
    p = [2.787860e-9, -1.398845e-6, 3.287156e-4, -5.779136e-2, 5.038813, 1.402385e3]
    c = np.polyval(p, temp)
    return c


def water_density(temp: float) -> float:
    """
    Calculate the density of air-saturated water with temperature.

    This function calculates the density of air-saturated water at a given temperature using the 4th order polynomial
    given by Jones [1].

    Args:
        temp: water temperature in the range 5 to 40 [degC]

    Returns:
        density: density of water [kg/m^3]

    Raises:
        ValueError: if `temp` is not between 5 and 40

    References:
        [1] F.E. Jones and G.L. Harris (1992) "ITS-90 Density of Water Formulation for Volumetric Standards Calibration,"
        J. Res. Natl. Inst.Stand.Technol., 97(3), 335-340.

    """

    # check limits
    if not (5 <= temp <= 40):
        raise ValueError("`temp` must be between 5 and 40.")

    # calculate density of air-saturated water
    density = 999.84847 + 6.337563e-2 * temp - 8.523829e-3 * temp ** 2 + 6.943248e-5 * temp ** 3 - 3.821216e-7 * temp ** 4
    return density


def water_non_linearity(temp: float) -> float:
    """
     Calculates the parameter of nonlinearity B/A at a
     given temperature using a fourth-order polynomial fitted to the data
     given by Beyer (1960).

    Args:
        temp: water temperature in the range 0 to 100 [degC]

    Returns:
        BonA: parameter of nonlinearity

    Examples:
         >>> BonA = waterNonlinearity(T)

     References:
         [1] R. T Beyer (1960) "Parameter of nonlinearity in fluids," J.
         Acoust. Soc. Am., 32(6), 719-721.

    """

    # check limits
    assert 0 <= temp <= 100, "Temp must be between 0 and 100."

    # find value
    p = [-4.587913769504693e-08, 1.047843302423604e-05, -9.355518377254833e-04, 5.380874771364909e-2, 4.186533937275504]
    BonA = np.polyval(p, temp)
    return BonA


def make_ball(grid_size: Vector, ball_center: Vector, radius: int, plot_ball: bool = False,
              binary: bool = False) -> np.ndarray:
    """
    Creates a binary map of a filled ball within a 3D grid.

    Args:
        grid_size: size of the 3D grid in [grid points].
        ball_center: centre of the ball in [grid points]
        radius: ball radius [grid points].
        plot_ball: whether to plot the ball using voxelPlot (default = False).
        binary: whether to return the ball map as a double precision matrix (False) or a logical matrix (True) (default = False).

    Returns:
        ball: 3D binary map of a filled ball.

    """

    # define literals
    MAGNITUDE = 1
    assert grid_size.shape == (3,), "grid_size must be a 3 element vector"
    assert ball_center.shape == (3,), "ball_center must be a 3 element vector"

    # force integer values
    grid_size = cast(Vector, grid_size.astype(int))
    ball_center = cast(Vector, ball_center.astype(int))

    # check for zero values
    for i in range(3):
        if ball_center[i] == 0:
            ball_center[i] = int(floor(grid_size[i] / 2)) + 1

    # create empty matrix
    ball = np.zeros(grid_size).astype(np.bool if binary else np.float32)

    # define np.pixel map
    r = make_pixel_map(grid_size, shift=[0, 0, 0])

    # create ball
    ball[r <= radius] = MAGNITUDE

    # shift centre
    ball_center = ball_center - np.ceil(grid_size / 2).astype(int)
    ball = np.roll(ball, ball_center, axis=(0, 1, 2))

    # plot results
    if plot_ball:
        raise NotImplementedError
        # voxelPlot(double(ball))
    return ball


def make_cart_sphere(radius: float, num_points: int, center_pos: Vector = Vector([0, 0, 0]),
                     plot_sphere: bool = False) -> Union[
    List[Tuple[float, float, float]], Tuple[List[Tuple[float, float, float]], Any]]:
    """
    Cart_sphere creates a set of points in Cartesian coordinates defining a sphere.

    Args:
        radius: the radius of the sphere.
        num_points: the number of points to be generated.
        center_pos: the coordinates of the center of the sphere. Defaults to (0, 0, 0).
        plot_sphere: whether to plot the sphere. Defaults to False.

    Returns:
        The points on the sphere.

    """

    cx, cy, cz = center_pos

    # generate angle functions using the Golden Section Spiral method
    inc = np.pi * (3 - np.sqrt(5))
    off = 2 / num_points
    k = np.arange(0, num_points)
    y = k * off - 1 + (off / 2)
    r = np.sqrt(1 - (y ** 2))
    phi = k * inc

    # create the sphere
    sphere = radius * np.concatenate([np.cos(phi) * r[np.newaxis, :], y[np.newaxis, :], np.sin(phi) * r[np.newaxis, :]])

    # offset if needed
    sphere[:] = sphere[:] + center_pos[None, :]

    # plot results
    if plot_sphere:
        # select suitable axis scaling factor
        [x_sc, scale, prefix, _] = scale_SI(np.max(sphere))

        # create the figure
        plt.figure()
        plt.style.use('seaborn-poster')
        ax = plt.axes(projection='3d')
        ax.plot3D(sphere[0, :] * scale, sphere[1, :] * scale, sphere[2, :] * scale, '.')
        ax.set_xlabel(f"[{prefix} m]")
        ax.set_ylabel(f"[{prefix} m]")
        ax.set_zlabel(f"[{prefix} m]")
        ax.axis('auto')
        ax.grid()
        plt.show()

    return sphere.squeeze()


def make_cart_circle(radius: float, num_points: int, center_pos: Vector = Vector([0, 0]),
                     arc_angle: float = 2 * np.pi, plot_circle: bool = False) -> np.ndarray:
    """
    Create a set of points in cartesian coordinates defining a circle or arc.

    This function creates a set of points in cartesian coordinates defining a circle or arc.

    Args:
        radius: radius of the circle or arc
        num_points: number of points to generate
        center_pos: center position of the circle or arc
        arc_angle: arc angle in radians
        plot_circle: whether to plot the circle or arc

    Returns:
        2 x `num_points` array of cartesian coordinates

    """

    # check for arc_angle input
    if arc_angle == 2 * np.pi:
        full_circle = True
    else:
        full_circle = False

    n_steps = num_points if full_circle else num_points - 1

    # create angles
    angles = np.arange(0, num_points) * arc_angle / n_steps + np.pi / 2

    # create cartesian grid
    circle = np.concatenate([radius * np.cos(angles[np.newaxis, :]), radius * np.sin(-angles[np.newaxis])])

    # offset if needed
    circle = circle + center_pos[:, None]

    # plot results
    if plot_circle:
        # select suitable axis scaling factor
        [_, scale, prefix, _] = scale_SI(np.max(abs(circle)))

        # create the figure
        plt.figure()
        plt.plot(circle[1, :] * scale, circle[0, :] * scale, 'b.')
        plt.xlabel([f"y-position [{prefix} m]"])
        plt.ylabel([f"x-position [{prefix} m]"])
        plt.axis('equal')
        plt.show()

    return np.squeeze(circle)


def make_disc(Nx, Ny, cx, cy, radius, plot_disc=False):
    """
    Create a binary map of a filled disc within a 2D grid.

    This function creates a binary map of a filled disc within a two-dimensional grid. The disc position is denoted by 1's
    in the matrix with 0's elsewhere. A single grid point is taken as the disc centre, so the total diameter of the disc
    will always be an odd number of grid points. If used within a k-Wave grid where dx != dy, the disc will appear oval
    shaped. If part of the disc overlaps the grid edge, the rest of the disc will wrap to the grid edge on the opposite
    side.

    Args:
        Nx: The number of grid points along the x-axis.
        Ny: The number of grid points along the y-axis.
        cx: The x-coordinate of the disc centre.
        cy: The y-coordinate of the disc centre.
        radius: The radius of the disc.
        plot_disc: If set to True, the disc will be plotted using Matplotlib.

    Returns:
        A binary map of the disc in the 2D grid.

    """

    # define literals
    MAGNITUDE = 1

    # force integer values
    Nx = int(round(Nx))
    Ny = int(round(Ny))
    cx = int(round(cx))
    cy = int(round(cy))

    # check for zero values
    if cx == 0:
        cx = int(floor(Nx / 2)) + 1

    if cy == 0:
        cy = int(floor(Ny / 2)) + 1

    # check the inputs
    assert (0 <= cx < Nx) and (0 <= cy < Ny), 'Disc center must be within grid.'

    # create empty matrix
    disc = np.zeros((Nx, Ny))

    # define np.pixel map
    r = make_pixel_map(Vector([Nx, Ny]), shift=[0, 0])

    # create disc
    disc[r <= radius] = MAGNITUDE

    # shift centre
    cx = cx - int(math.ceil(Nx / 2))
    cy = cy - int(math.ceil(Ny / 2))
    disc = np.roll(disc, (cx, cy), axis=(0, 1))

    # create the figure
    if plot_disc:
        raise NotImplementedError
    return disc


def make_circle(Nx: int, Ny: int, cx: int, cy: int, radius: int, arc_angle: Optional[float] = None,
                plot_circle: bool = False) -> np.ndarray:
    """
    Create a binary map of a circle within a 2D grid.

    This function creates a binary map of a circle (or arc) using the midpoint circle algorithm within a two-dimensional grid.
    The circle position is denoted by 1's in the matrix with 0's elsewhere. A single grid point is taken as the circle
    centre, so the total diameter will always be an odd number of grid points. The centre of the circle and the radius
    are not constrained by the grid dimensions, so it is possible to create sections of circles or a blank image if none
    of the circle intersects the grid.

    Args:
        Nx: The number of grid points along the x-axis.
        Ny: The number of grid points along the y-axis.
        cx: The x-coordinate of the circle centre.
        cy: The y-coordinate of the circle centre.
        radius: The radius of the circle.
        arc_angle: The angle of the circular arc in degrees. If set to None, a full circle will be created.
        plot_circle: If set to True, the circle will be plotted using Matplotlib.

    Returns:
        A binary map of the circle in the 2D grid.

    """

    # define literals
    MAGNITUDE = 1

    if arc_angle is None:
        arc_angle = 2 * np.pi
    elif arc_angle > 2 * np.pi:
        arc_angle = 2 * np.pi
    elif arc_angle < 0:
        arc_angle = 0

    # force integer values
    Nx = int(round(Nx))
    Ny = int(round(Ny))
    cx = int(round(cx))
    cy = int(round(cy))
    radius = int(round(radius))

    # check for zero values
    if cx == 0:
        cx = int(floor(Nx / 2)) + 1

    if cy == 0:
        cy = int(floor(Ny / 2)) + 1

    # create empty matrix
    circle = np.zeros((Nx, Ny), dtype=int)

    # initialise loop variables
    x = 0
    y = radius
    d = 1 - radius

    if (cx >= 1) and (cx <= Nx) and ((cy - y) >= 1) and ((cy - y) <= Ny):
        circle[cx - 1, cy - y - 1] = MAGNITUDE

    # draw the remaining cardinal points
    px = [cx, cx + y, cx - y]
    py = [cy + y, cy, cy]
    for point_index, (px_i, py_i) in enumerate(zip(px, py)):
        # check whether the point is within the arc made by arc_angle, and lies
        # within the grid
        if (np.arctan2(px_i - cx, py_i - cy) + np.pi) <= arc_angle:
            if (px_i >= 1) and (px_i <= Nx) and (py_i >= 1) and (
                    py_i <= Ny):
                circle[px_i - 1, py_i - 1] = MAGNITUDE

    # loop through the remaining points using the midpoint circle algorithm
    while x < (y - 1):

        x = x + 1
        if d < 0:
            d = d + x + x + 1
        else:
            y = y - 1
            a = x - y + 1
            d = d + a + a

        # setup point indices (break coding standard for readability)
        px = [x + cx, y + cx, y + cx, x + cx, -x + cx, -y + cx, -y + cx, -x + cx]
        py = [y + cy, x + cy, -x + cy, -y + cy, -y + cy, -x + cy, x + cy, y + cy]

        # loop through each point
        for point_index, (px_i, py_i) in enumerate(zip(px, py)):

            # check whether the point is within the arc made by arc_angle, and
            # lies within the grid
            if (np.arctan2(px_i - cx, py_i - cy) + np.pi) <= arc_angle:
                if (px_i >= 1) and (px_i <= Nx) and (py_i >= 1) and (py_i <= Ny):
                    circle[px_i - 1, py_i - 1] = MAGNITUDE

    if plot_circle:
        plt.imshow(circle, cmap='gray_r')
        plt.ylabel('x-position [grid points]')
        plt.xlabel('y-position [grid points]')
        plt.show()

    return circle


def make_pixel_map(grid_size: Vector, shift=None, origin_size='single') -> np.ndarray:
    """
    Generates a matrix with values of the distance of each pixel from the center of a grid.

    This function generates a matrix populated with values of how far each pixel in a grid is from the center. The center
    can be a single pixel or a double pixel, and the optional input parameter 'OriginSize' controls this. For grids where
    the dimension size and center pixel size are not both odd or even, the optional input parameter 'Shift' can be used to
    control the location of the center point.

    Args:
        Nx: number of pixels in the x-dimension
        Ny: number of pixels in the y-dimension
        Nz: number of pixels in the z-dimension
        *args: additional optional arguments

    Returns:
        r: pixel-radius

    Examples:

        Single pixel origin size for odd and even (with 'Shift' = [1 1] and
        [0 0], respectively) grid sizes:

         x x x       x x x x         x x x x
         x 0 x       x x x x         x 0 x x
         x x x       x x 0 x         x x x x
                     x x x x         x x x x

         Double pixel origin size for even and odd (with 'Shift' = [1 1] and
         [0 0], respectively) grid sizes:

         x x x x      x x x x x        x x x x x
         x 0 0 x      x x x x x        x 0 0 x x
         x 0 0 x      x x 0 0 x        x 0 0 x x
         x x x x      x x 0 0 x        x x x x x
                      x x x x x        x x x x x

         By default, a single pixel centre is used which is shifted towards
         the final row and column.

    """
    assert len(grid_size) == 2 or len(grid_size) == 3, 'Grid size must be a 2 or 3 element vector.'

    # define defaults
    shift_def = 1

    Nx = grid_size[0]
    Ny = grid_size[1]
    Nz = None
    if len(grid_size) == 3:
        Nz = grid_size[2]

    # detect whether the inputs are for two or three dimensions
    map_dimension = 2 if Nz is None else 3
    if shift is None:
        shift = [shift_def] * map_dimension

    # catch input errors
    assert origin_size in ['single', 'double'], 'Unknown setting for optional input Center.'

    assert len(
        shift) == map_dimension, f'Optional input Shift must have {map_dimension} elements for {map_dimension} dimensional input parameters.'

    if map_dimension == 2:
        # create the maps for each dimension
        nx = create_pixel_dim(Nx, origin_size, shift[0])
        ny = create_pixel_dim(Ny, origin_size, shift[1])

        # create plaid grids
        r_x, r_y = np.meshgrid(nx, ny, indexing='ij')

        # extract the pixel radius
        r = np.sqrt(r_x ** 2 + r_y ** 2)
    if map_dimension == 3:
        # create the maps for each dimension
        nx = create_pixel_dim(Nx, origin_size, shift[0])
        ny = create_pixel_dim(Ny, origin_size, shift[1])
        nz = create_pixel_dim(Nz, origin_size, shift[2])

        # create plaid grids
        r_x, r_y, r_z = np.meshgrid(nx, ny, nz, indexing='ij')

        # extract the pixel radius
        r = np.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2)
    return r


def create_pixel_dim(Nx: int, origin_size: float, shift: float) -> Tuple[np.ndarray, float]:
    """
    Create an array of pixel dimensions and a pixel size.

    Args:
        Nx: The number of pixels in the x-dimension.
        origin_size: The size of the origin in the x-dimension.
        shift: The shift of the pixels in the x-dimension.

    Returns:
        The pixel dimensions.

    """

    # Nested function to create the pixel radius variable

    # grid dimension has an even number of points
    if Nx % 2 == 0:

        # pixel numbering has a single centre point
        if origin_size == 'single':

            # centre point is shifted towards the final pixel
            if shift == 1:
                nx = np.arange(-Nx / 2, Nx / 2 - 1 + 1, 1)

            # centre point is shifted towards the first pixel
            else:
                nx = np.arange(-Nx / 2 + 1, Nx / 2 + 1, 1)

        # pixel numbering has a double centre point
        else:
            nx = np.hstack([np.arange(-Nx / 2 + 1, 0 + 1, 1), np.arange(0, -Nx / 2 - 1 + 1, 1)])

    # grid dimension has an odd number of points
    else:

        # pixel numbering has a single centre point
        if origin_size == 'single':
            nx = np.arange(-(Nx - 1) / 2, (Nx - 1) / 2 + 1, 1)

        # pixel numbering has a double centre point
        else:

            # centre point is shifted towards the final pixel
            if shift == 1:
                nx = np.hstack([np.arange(-(Nx - 1) / 2, 0 + 1, 1), np.arange(0, (Nx - 1) / 2 - 1 + 1, 1)])

            # centre point is shifted towards the first pixel
            else:
                nx = np.hstack([np.arange(-(Nx - 1) / 2 + 1, 0 + 1, 1), np.arange(0, (Nx - 1) / 2 + 1, 1)])
    return nx


def make_line(
        Nx: int,
        Ny: int,
        startpoint: Tuple[int, int],
        endpoint: Optional[Tuple[int, int]] = None,
        angle: Optional[float] = None,
        length: Optional[int] = None
) -> np.ndarray:
    """
    Generate a line shape with a given start and end point, angle, or length.

    Args:
        Nx: The number of pixels in the x-dimension.
        Ny: The number of pixels in the y-dimension.
        startpoint: The start point of the line, given as a tuple of x and y coordinates.
        endpoint: The end point of the line, given as a tuple of x and y coordinates. If not specified, the line is drawn from the start point at a given angle and length.
        angle: The angle of the line in radians, measured counterclockwise from the x-axis. If not specified, the line is drawn from the start point to the end point.
        length: The length of the line in pixels. If not specified, the line is drawn from the start point to the end point.

    Returns:
        line: A 2D array of the same size as the input parameters, with a value of 1 for pixels that are part of the line and 0 for pixels that are not.
    """

    startpoint = np.array(startpoint, dtype=int)
    if endpoint is not None:
        endpoint = np.array(endpoint, dtype=int)

    if len(startpoint) != 2:
        raise ValueError('startpoint should be a two-element vector.')

    if np.any(startpoint < 1) or startpoint[0] > Nx or startpoint[1] > Ny:
        ValueError('The starting point must lie within the grid, between [1 1] and [Nx Ny].')

    # =========================================================================
    # LINE BETWEEN TWO POINTS OR ANGLED LINE?
    # =========================================================================

    if endpoint is not None:
        linetype = 'AtoB'
        a, b = startpoint, endpoint

        # Addition => Fix Matlab2Python indexing
        a -= 1
        b -= 1
    else:
        linetype = 'angled'
        angle, linelength = angle, length

    # =========================================================================
    # MORE INPUT CHECKING
    # =========================================================================

    if linetype == 'AtoB':

        # a and b must be different points
        if np.all(a == b):
            raise ValueError('The first and last points cannot be the same.')

        # end point must be a two-element row vector
        if len(b) != 2:
            raise ValueError('endpoint should be a two-element vector.')

        # a and b must be within the grid
        xx = np.array([a[0], b[0]], dtype=int)
        yy = np.array([a[1], b[1]], dtype=int)
        if np.any(a < 0) or np.any(b < 0) or np.any(xx > Nx - 1) or np.any(yy > Ny - 1):
            raise ValueError('Both the start and end points must lie within the grid.')

    if linetype == 'angled':

        # angle must lie between -np.pi and np.pi
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle = angle - (2 * np.pi)
        elif angle < -np.pi:
            angle = angle + (2 * np.pi)

    # =========================================================================
    # CALCULATE A LINE FROM A TO B
    # =========================================================================

    if linetype == 'AtoB':

        # define an empty grid to hold the line
        line = np.zeros((Nx, Ny))

        # find the equation of the line
        m = (b[1] - a[1]) / (b[0] - a[0])  # gradient of the line
        c = a[1] - m * a[0]  # where the line crosses the y axis

        if abs(m) < 1:

            # start at the end with the smallest value of x
            if a[0] < b[0]:
                x, y = a
                x_end = b[0]
            else:
                x, y = b
                x_end = a[0]

            # fill in the first point
            line[x, y] = 1

            while x < x_end:
                # next points to try are
                poss_x = [x, x, x + 1, x + 1, x + 1]
                poss_y = [y - 1, y + 1, y - 1, y, y + 1]

                # find the point closest to the line
                true_y = m * poss_x + c
                diff = (poss_y - true_y) ** 2
                index = matlab_find(diff == min(diff))[0]

                # the next point
                x = poss_x[index[0] - 1]
                y = poss_y[index[0] - 1]

                # add the point to the line
                line[x - 1, y - 1] = 1

        elif not np.isinf(abs(m)):

            # start at the end with the smallest value of y
            if a[1] < b[1]:
                x = a[0]
                y = a[1]
                y_end = b[1]
            else:
                x = b[0]
                y = b[1]
                y_end = a[1]

            # fill in the first point
            line[x, y] = 1

            while y < y_end:
                # next points to try are
                poss_y = [y, y, y + 1, y + 1, y + 1]
                poss_x = [x - 1, x + 1, x - 1, x, x + 1]

                # find the point closest to the line
                true_x = (poss_y - c) / m
                diff = (poss_x - true_x) ** 2
                index = matlab_find(diff == min(diff))[0]

                # the next point
                x = poss_x[index[0] - 1]
                y = poss_y[index[0] - 1]

                # add the point to the line
                line[x, y] = 1

        else:  # m = +-Inf

            # start at the end with the smallest value of y
            if a[1] < b[1]:
                x = a[0]
                y = a[1]
                y_end = b[1]
            else:
                x = b[0]
                y = b[1]
                y_end = a[1]

            # fill in the first point
            line[x, y] = 1

            while y < y_end:
                # next point
                y = y + 1

                # add the point to the line
                line[x, y] = 1

    # =========================================================================
    # CALCULATE AN ANGLED LINE
    # =========================================================================

    elif linetype == 'angled':

        # define an empty grid to hold the line
        line = np.zeros((Nx, Ny))

        # start at the atart
        x, y = startpoint

        # fill in the first point
        line[x - 1, y - 1] = 1

        # initialise the current length of the line
        line_length = 0

        if abs(angle) == np.pi:

            while line_length < linelength:

                # next point
                y = y + 1

                # stop the points incrementing at the edges
                if y > Ny:
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif (angle < np.pi) and (angle > np.pi / 2):

            # define the equation of the line
            m = -np.tan(angle - np.pi / 2)  # gradient of the line
            c = y - m * x  # where the line crosses the y axis

            while line_length < linelength:

                # next points to try are
                poss_x = np.array([x - 1, x - 1, x])
                poss_y = np.array([y, y + 1, y + 1])

                # find the point closest to the line
                true_y = m * poss_x + c
                diff = (poss_y - true_y) ** 2
                index = matlab_find(diff == min(diff))[0]

                # the next point
                x = poss_x[index[0] - 1]
                y = poss_y[index[0] - 1]

                # stop the points incrementing at the edges
                if (x < 0) or (y > Ny - 1):
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif angle == np.pi / 2:

            while line_length < linelength:

                # next point
                x = x - 1

                # stop the points incrementing at the edges
                if x < 1:
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif (angle < np.pi / 2) and (angle > 0):

            # define the equation of the line
            m = np.tan(np.pi / 2 - angle)  # gradient of the line
            c = y - m * x  # where the line crosses the y axis

            while line_length < linelength:

                # next points to try are
                poss_x = np.array([x - 1, x - 1, x])
                poss_y = np.array([y, y - 1, y - 1])

                # find the point closest to the line
                true_y = m * poss_x + c
                diff = (poss_y - true_y) ** 2
                index = matlab_find(diff == min(diff))[0]

                # the next point
                x = poss_x[index[0] - 1]
                y = poss_y[index[0] - 1]

                # stop the points incrementing at the edges
                if (x < 1) or (y < 1):
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif angle == 0:

            while line_length < linelength:

                # next point
                y = y - 1

                # stop the points incrementing at the edges
                if y < 1:
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif (angle < 0) and (angle > -np.pi / 2):

            # define the equation of the line
            m = -np.tan(np.pi / 2 + angle)  # gradient of the line
            c = y - m * x  # where the line crosses the y axis

            while line_length < linelength:

                # next points to try are
                poss_x = np.array([x + 1, x + 1, x])
                poss_y = np.array([y, y - 1, y - 1])

                # find the point closest to the line
                true_y = m * poss_x + c
                diff = (poss_y - true_y) ** 2
                index = matlab_find(diff == min(diff))[0]

                # the next point
                x = poss_x[index[0] - 1]
                y = poss_y[index[0] - 1]

                # stop the points incrementing at the edges
                if (x > Nx) or (y < 1):
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif angle == -np.pi / 2:

            while line_length < linelength:

                # next point
                x = x + 1

                # stop the points incrementing at the edges
                if x > Nx:
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

        elif (angle < -np.pi / 2) and (angle > -np.pi):

            # define the equation of the line
            m = np.tan(-angle - np.pi / 2)  # gradient of the line
            c = y - m * x  # where the line crosses the y axis

            while line_length < linelength:

                # next points to try are
                poss_x = np.array([x + 1, x + 1, x])
                poss_y = np.array([y, y + 1, y + 1])

                # find the point closest to the line
                true_y = m * poss_x + c
                diff = (poss_y - true_y) ** 2
                index = matlab_find(diff == min(diff))[0]

                # the next point
                x = poss_x[index[0] - 1]
                y = poss_y[index[0] - 1]

                # stop the points incrementing at the edges
                if (x > Nx) or (y > Ny):
                    break

                # add the point to the line
                line[x - 1, y - 1] = 1

                # calculate the current length of the line
                line_length = np.sqrt((x - startpoint[0]) ** 2 + (y - startpoint[1]) ** 2)

    return line


def make_arc(grid_size: np.ndarray, arc_pos: np.ndarray, radius: float, diameter: float,
             focus_pos: np.ndarray) -> np.ndarray:
    """
    Generates an arc shape with a given radius, diameter, and focus position.

    Args:
        grid_size: The size of the grid, given as a 1D array with the number of pixels in each dimension.
        arc_pos: The position of the arc, given as a 1D array with the coordinates in each dimension.
        radius: The radius of the arc.
        diameter: The diameter of the arc.
        focus_pos: The position of the focus, given as a 1D array with the coordinates in each dimension.

    Returns:
        np.ndarray: A 2D array with the arc shape.
    """
    # force integer input values
    grid_size = grid_size.round().astype(int)
    arc_pos = arc_pos.round().astype(int)
    diameter = int(round(diameter))
    focus_pos = focus_pos.round().astype(int)

    try:
        radius = int(radius)
    except OverflowError:
        radius = float(radius)

    # check the input ranges
    if np.any(grid_size < 1):
        raise ValueError('The grid size must be positive.')
    if radius <= 0:
        raise ValueError('The radius must be positive.')

    if diameter <= 0:
        raise ValueError('The diameter must be positive.')

    if np.any(arc_pos < 1) or np.any(arc_pos > grid_size):
        raise ValueError('The centre of the arc must be within the grid.')

    if diameter > 2 * radius:
        raise ValueError('The diameter of the arc must be less than twice the radius of curvature.')

    if diameter % 2 != 1:
        raise ValueError('The diameter must be an odd number of grid points.')

    if np.all(arc_pos == focus_pos):
        raise ValueError('The focus_pos must be different to the arc_pos.')

    # assign variable names to vector components
    Nx, Ny = grid_size
    ax, ay = arc_pos
    fx, fy = focus_pos

    # =========================================================================
    # CREATE ARC
    # =========================================================================

    if not np.isinf(radius):

        # find half the arc angle
        half_arc_angle = np.arcsin(diameter / 2 / radius)

        # find centre of circle on which the arc lies
        distance_cf = np.sqrt((ax - fx) ** 2 + (ay - fy) ** 2)
        cx = round(radius / distance_cf * (fx - ax) + ax)
        cy = round(radius / distance_cf * (fy - ay) + ay)
        c = np.array([cx, cy])

        # create circle
        arc = make_circle(Nx, Ny, cx, cy, radius)

        # form vector from the geometric arc centre to the arc midpoint
        v1 = arc_pos - c

        # calculate length of vector
        l1 = np.sqrt(sum((arc_pos - c) ** 2))

        # extract all points that form part of the arc
        arc_ind = matlab_find(arc, mode='eq', val=1)

        # loop through the arc points
        for arc_ind_i in arc_ind:

            # extract the indices of the current point
            x_ind, y_ind = ind2sub([Nx, Ny], arc_ind_i)
            p = np.array([x_ind, y_ind])

            # form vector from the geometric arc centre to the current point
            v2 = p - c

            # calculate length of vector
            l2 = np.sqrt(sum((p - c) ** 2))

            # find the angle between the two vectors using the dot product,
            # normalised using the vector lengths
            theta = np.arccos(sum(v1 * v2 / (l1 * l2)))

            # if the angle is greater than the half angle of the arc, remove
            # it from the arc
            if theta > half_arc_angle:
                arc = matlab_assign(arc, arc_ind_i - 1, 0)
    else:

        # calculate arc direction angle, then rotate by 90 degrees
        ang = np.arctan((fx - ax) / (fy - ay)) + np.pi / 2

        # draw lines to create arc with infinite radius
        arc = np.logical_or(
            make_line(Nx, Ny, arc_pos, endpoint=None, angle=ang, length=(diameter - 1) // 2),
            make_line(Nx, Ny, arc_pos, endpoint=None, angle=(ang + np.pi), length=(diameter - 1) // 2)
        )
    return arc


def make_pixel_map_point(grid_size: np.ndarray, centre_pos: np.ndarray) -> np.ndarray:
    """
    Generates a map of the distance of each pixel from a given centre position.

    Args:
        grid_size: The size of the grid, given as a 1D array with the number of pixels in each dimension.
        centre_pos: The position of the centre, given as a 1D array with the coordinates in each dimension.

    Returns:
        np.ndarray: A 2D array with the distance of each pixel from the given centre position.

    Raises:
        ValueError: If `grid_size` and `centre_pos` do not have the same number of dimensions.
    """
    # check for number of dimensions
    num_dim = len(grid_size)

    # check that centre_pos has the same dimensions
    if len(grid_size) != len(centre_pos):
        raise ValueError('The inputs centre_pos and grid_size must have the same number of dimensions.')

    if num_dim == 2:
        # assign inputs and force to be integers
        Nx, Ny = grid_size.astype(int)
        cx, cy = centre_pos.astype(int)

        # generate index vectors in each dimension
        nx = np.arange(0, Nx) - cx + 1
        ny = np.arange(0, Ny) - cy + 1

        # combine index matrices
        pixel_map = np.zeros((Nx, Ny))
        pixel_map += (nx ** 2)[:, None]
        pixel_map += (ny ** 2)[None, :]
        pixel_map = np.sqrt(pixel_map)

    elif num_dim == 3:

        # assign inputs and force to be integers
        Nx, Ny, Nz = grid_size.astype(int)
        cx, cy, cz = centre_pos.astype(int)

        # generate index vectors in each dimension
        nx = np.arange(0, Nx) - cx + 1
        ny = np.arange(0, Ny) - cy + 1
        nz = np.arange(0, Nz) - cz + 1

        # combine index matrices
        pixel_map = np.zeros((Nx, Ny, Nz))
        pixel_map += (nx ** 2)[:, None, None]
        pixel_map += (ny ** 2)[None, :, None]
        pixel_map += (nz ** 2)[None, None, :]
        pixel_map = np.sqrt(pixel_map)

    else:
        # throw error
        raise ValueError('Grid size must be 2 or 3D.')

    return pixel_map


def make_pixel_map_plane(grid_size: np.ndarray, normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Generates a pixel map of a plane with given normal vector and point.

    Args:
        grid_size: The size of the grid as a NumPy array [Nx, Ny, Nz].
        normal: The normal vector of the plane as a NumPy array [nx, ny, nz].
        point: A point on the plane as a NumPy array [px, py, pz].

    Returns:
        pixel_map: A 2D array with the distance of each pixel from the given plane.

    Raises:
        ValueError: If the normal vector is zero.
    """  # error checking
    if np.all(normal == 0):
        raise ValueError('Normal vector should not be zero.')

    # check for number of dimensions
    num_dim = len(grid_size)

    if num_dim == 2:
        # assign inputs and force to be integers
        Nx = round(grid_size[0])
        Ny = round(grid_size[1])

        # create coordinate meshes
        [px, py] = np.meshgrid(Nx, Ny)
        [pointx, pointy] = np.meshgrid(np.ones((1, Nx)) * point[0], np.ones(1, Ny) * point[1])
        [nx, ny] = np.meshgrid(np.ones((1, Nx)) * normal[0], np.ones(1, Ny) * normal[2])

        # calculate distance according to Eq. (6) at
        # http://mathworld.wolfram.com/Point-PlaneDistance.html
        pixel_map = np.abs((px - pointx) * nx + (py - pointy) * ny) / np.sqrt(sum(normal ** 2))

    elif num_dim == 3:

        # assign inputs and force to be integers
        Nx = np.round(grid_size[0])
        Ny = np.round(grid_size[1])
        Nz = np.round(grid_size[2])

        # create coordinate meshes
        px, py, pz = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1), indexing='ij')
        pointx, pointy, pointz = np.meshgrid(np.ones(Nx) * point[0], np.ones(Ny) * point[1], np.ones(Nz) * point[2],
                                             indexing='ij')
        nx, ny, nz = np.meshgrid(np.ones(Nx) * normal[0], np.ones(Ny) * normal[1], np.ones(Nz) * normal[2],
                                 indexing='ij')

        # calculate distance according to Eq. (6) at
        # http://mathworld.wolfram.com/Point-PlaneDistance.html
        pixel_map = np.abs((px - pointx) * nx + (py - pointy) * ny + (pz - pointz) * nz) / np.sqrt(sum(normal ** 2))

    else:
        # throw error
        raise ValueError('Grid size must be 2 or 3D.')

    return pixel_map


def make_bowl(grid_size: Tuple[int, int, int], bowl_pos: Tuple[int, int, int], radius: int, diameter: int,
              focus_pos: Tuple[int, int, int], binary: bool = False, remove_overlap: bool = False) -> np.ndarray:
    """
    Generate a matrix representing a bowl-shaped object in 3D space.

    This function generates a 3D matrix representing a bowl-shaped object with the specified radius and diameter. The position
    of the bowl and the focus point can be specified, as well as whether to return a binary matrix or a matrix with
    continuous values. The optional parameter 'remove_overlap' can be used to remove any overlap with the surrounding grid.

    Args:
        grid_size: size of the grid in each dimension
        bowl_pos: position of the bowl in the grid
        radius: radius of the bowl
        diameter: diameter of the bowl
        focus_pos: position of the focus point in the grid
        binary: whether to return a binary matrix
        remove_overlap: whether to remove overlap with the surrounding grid

    Returns:
        matrix: 3D matrix representing the bowl-shaped object

    Raises:
        ValueError: if any of the input arguments are outside the valid range
    """
    # =========================================================================
    # DEFINE LITERALS
    # =========================================================================

    # threshold used to find the closest point to the radius
    THRESHOLD = 0.5

    # number of grid points to expand the bounding box compared to
    # sqrt(2)*diameter
    BOUNDING_BOX_EXP = 2

    # =========================================================================
    # INPUT CHECKING
    # =========================================================================

    # force integer input values
    grid_size = np.round(grid_size).astype(int)
    bowl_pos = np.round(bowl_pos).astype(int)
    focus_pos = np.round(focus_pos).astype(int)
    diameter = np.round(diameter)
    radius = np.round(radius)

    # check the input ranges
    if np.any(grid_size < 1):
        raise ValueError('The grid size must be positive.')
    if np.any(bowl_pos < 1) or np.any(bowl_pos > grid_size):
        raise ValueError('The centre of the bowl must be within the grid.')
    if radius <= 0:
        raise ValueError('The radius must be positive.')
    if diameter <= 0:
        raise ValueError('The diameter must be positive.')
    if diameter > (2 * radius):
        raise ValueError('The diameter of the bowl must be less than twice the radius of curvature.')
    if diameter % 2 == 0:
        raise ValueError('The diameter must be an odd number of grid points.')
    if np.all(bowl_pos == focus_pos):
        raise ValueError('The focus_pos must be different to the bowl_pos.')

    # =========================================================================
    # BOUND THE GRID TO SPEED UP CALCULATION
    # =========================================================================

    # create bounding box slightly larger than bowl diameter * sqrt(2)
    Nx = np.round(np.sqrt(2) * diameter).astype(int) + BOUNDING_BOX_EXP
    Ny = Nx
    Nz = Nx
    grid_size_sm = np.array([Nx, Ny, Nz])

    # set the bowl position to be the centre of the bounding box
    bx = np.ceil(Nx / 2).astype(int)
    by = np.ceil(Ny / 2).astype(int)
    bz = np.ceil(Nz / 2).astype(int)
    bowl_pos_sm = np.array([bx, by, bz])

    # set the focus position to be in the direction specified by the user
    fx = bx + (focus_pos[0] - bowl_pos[0])
    fy = by + (focus_pos[1] - bowl_pos[1])
    fz = bz + (focus_pos[2] - bowl_pos[2])
    focus_pos_sm = [fx, fy, fz]

    # preallocate storage variable
    if binary:
        bowl_sm = np.zeros((Nx, Ny, Nz), dtype=bool)
    else:
        bowl_sm = np.zeros((Nx, Ny, Nz))

    # =========================================================================
    # CREATE DISTANCE MATRIX
    # =========================================================================

    if not np.isinf(radius):

        # find half the arc angle
        half_arc_angle = np.arcsin(diameter / (2 * radius))

        # find centre of sphere on which the bowl lies
        distance_cf = np.sqrt((bx - fx) ** 2 + (by - fy) ** 2 + (bz - fz) ** 2)
        cx = round(radius / distance_cf * (fx - bx) + bx)
        cy = round(radius / distance_cf * (fy - by) + by)
        cz = round(radius / distance_cf * (fz - bz) + bz)
        c = np.array([cx, cy, cz])

        # generate matrix with distance from the centre
        pixel_map = make_pixel_map_point(grid_size_sm, c)

        # set search radius to bowl radius
        search_radius = radius

    else:

        # generate matrix with distance from the centre
        pixel_map = make_pixel_map_plane(grid_size_sm, bowl_pos_sm - focus_pos_sm, bowl_pos_sm)

        # set search radius to 0 (the disc is flat)
        search_radius = 0

    # calculate distance from search radius
    pixel_map = np.abs(pixel_map - search_radius)

    # =========================================================================
    # DIMENSION 1
    # =========================================================================

    # find the grid point that corresponds to the outside of the bowl in the
    # first dimension in both directions (the index gives the distance along
    # this dimension)
    value_forw, index_forw = pixel_map.min(axis=0), pixel_map.argmin(axis=0)
    value_back, index_back = np.flip(pixel_map, axis=0).min(axis=0), np.flip(pixel_map, axis=0).argmin(axis=0)

    # extract the linear index in the y-z plane of the values that lie on the
    # bowl surface
    yz_ind_forw = matlab_find(value_forw < THRESHOLD)
    yz_ind_back = matlab_find(value_back < THRESHOLD)

    # use these subscripts to extract the x-index of the grid points that lie
    # on the bowl surface
    x_ind_forw = index_forw.flatten(order='F')[yz_ind_forw - 1] + 1
    x_ind_back = index_back.flatten(order='F')[yz_ind_back - 1] + 1

    # convert the linear index to equivalent subscript values
    y_ind_forw, z_ind_forw = ind2sub([Ny, Nz], yz_ind_forw)
    y_ind_back, z_ind_back = ind2sub([Ny, Nz], yz_ind_back)

    # combine x-y-z indices into a linear index
    linear_index_forw = sub2ind([Nx, Ny, Nz], x_ind_forw - 1, y_ind_forw - 1, z_ind_forw - 1) + 1
    linear_index_back = sub2ind([Nx, Ny, Nz], Nx - x_ind_back, y_ind_back - 1, z_ind_back - 1) + 1

    # assign these values to the bowl
    bowl_sm = matlab_assign(bowl_sm, linear_index_forw - 1, 1)
    bowl_sm = matlab_assign(bowl_sm, linear_index_back - 1, 1)

    # set existing bowl values to a distance of zero in the pixel map (this
    # avoids problems with overlapping pixels)
    pixel_map[bowl_sm == 1] = 0

    # =========================================================================
    # DIMENSION 2
    # =========================================================================

    # find the grid point that corresponds to the outside of the bowl in the
    # second dimension in both directions (the pixel map is first re-ordered to
    # [X, Y, Z] -> [Y, Z, X])
    pixel_map_temp = np.transpose(pixel_map, (1, 2, 0))
    value_forw, index_forw = pixel_map_temp.min(axis=0), pixel_map_temp.argmin(axis=0)
    value_back, index_back = np.flip(pixel_map_temp, axis=0).min(axis=0), np.flip(pixel_map_temp, axis=0).argmin(axis=0)
    del pixel_map_temp

    # extract the linear index in the y-z plane of the values that lie on the
    # bowl surface
    zx_ind_forw = matlab_find(value_forw < THRESHOLD)
    zx_ind_back = matlab_find(value_back < THRESHOLD)

    # use these subscripts to extract the y-index of the grid points that lie
    # on the bowl surface
    y_ind_forw = index_forw.flatten(order='F')[zx_ind_forw - 1] + 1
    y_ind_back = index_back.flatten(order='F')[zx_ind_back - 1] + 1

    # convert the linear index to equivalent subscript values
    z_ind_forw, x_ind_forw = ind2sub([Nz, Nx], zx_ind_forw)
    z_ind_back, x_ind_back = ind2sub([Nz, Nx], zx_ind_back)

    # combine x-y-z indices into a linear index
    linear_index_forw = sub2ind([Nx, Ny, Nz], x_ind_forw - 1, y_ind_forw - 1, z_ind_forw - 1) + 1
    linear_index_back = sub2ind([Nx, Ny, Nz], x_ind_back - 1, Ny - y_ind_back, z_ind_back - 1) + 1

    # assign these values to the bowl
    bowl_sm = matlab_assign(bowl_sm, linear_index_forw - 1, 1)
    bowl_sm = matlab_assign(bowl_sm, linear_index_back - 1, 1)

    # set existing bowl values to a distance of zero in the pixel map (this
    # avoids problems with overlapping pixels)
    pixel_map[bowl_sm == 1] = 0

    # =========================================================================
    # DIMENSION 3
    # =========================================================================

    # find the grid point that corresponds to the outside of the bowl in the
    # third dimension in both directions (the pixel map is first re-ordered to
    # [X, Y, Z] -> [Z, X, Y])
    pixel_map_temp = np.transpose(pixel_map, (2, 0, 1))
    value_forw, index_forw = pixel_map_temp.min(axis=0), pixel_map_temp.argmin(axis=0)
    value_back, index_back = np.flip(pixel_map_temp, axis=0).min(axis=0), np.flip(pixel_map_temp, axis=0).argmin(axis=0)
    del pixel_map_temp

    # extract the linear index in the y-z plane of the values that lie on the
    # bowl surface
    xy_ind_forw = matlab_find(value_forw < THRESHOLD)
    xy_ind_back = matlab_find(value_back < THRESHOLD)

    # use these subscripts to extract the z-index of the grid points that lie
    # on the bowl surface
    z_ind_forw = index_forw.flatten(order='F')[xy_ind_forw - 1] + 1
    z_ind_back = index_back.flatten(order='F')[xy_ind_back - 1] + 1

    # convert the linear index to equivalent subscript values
    x_ind_forw, y_ind_forw = ind2sub([Nx, Ny], xy_ind_forw)
    x_ind_back, y_ind_back = ind2sub([Nx, Ny], xy_ind_back)

    # combine x-y-z indices into a linear index
    linear_index_forw = sub2ind([Nx, Ny, Nz], x_ind_forw - 1, y_ind_forw - 1, z_ind_forw - 1) + 1
    linear_index_back = sub2ind([Nx, Ny, Nz], x_ind_back - 1, y_ind_back - 1, Nz - z_ind_back) + 1

    # assign these values to the bowl
    bowl_sm = matlab_assign(bowl_sm, linear_index_forw - 1, 1)
    bowl_sm = matlab_assign(bowl_sm, linear_index_back - 1, 1)

    # =========================================================================
    # RESTRICT SPHERE TO BOWL
    # =========================================================================

    # remove grid points within the sphere that do not form part of the bowl
    if not np.isinf(radius):

        # form vector from the geometric bowl centre to the back of the bowl
        v1 = bowl_pos_sm - c

        # calculate length of vector
        l1 = np.sqrt(sum((bowl_pos_sm - c) ** 2))

        # loop through the non-zero elements in the bowl matrix
        bowl_ind = matlab_find(bowl_sm == 1)[:, 0]
        for bowl_ind_i in bowl_ind:

            # extract the indices of the current point
            x_ind, y_ind, z_ind = ind2sub([Nx, Ny, Nz], bowl_ind_i)
            p = np.array([x_ind, y_ind, z_ind])

            # form vector from the geometric bowl centre to the current point
            # on the bowl
            v2 = p - c

            # calculate length of vector
            l2 = np.sqrt(sum((p - c) ** 2))

            # find the angle between the two vectors using the dot product,
            # normalised using the vector lengths
            theta = np.arccos(sum(v1 * v2 / (l1 * l2)))

            #         # alternative calculation normalised using radius of curvature
            #         theta2 = acos(sum( v1 .* v2 ./ radius**2 ))

            # if the angle is greater than the half angle of the bowl, remove
            # it from the bowl
            if theta > half_arc_angle:
                bowl_sm = matlab_assign(bowl_sm, bowl_ind_i - 1, 0)

    else:

        # form a distance map from the centre of the disc
        pixelMapPoint = make_pixel_map_point(grid_size_sm, bowl_pos_sm)

        # set all points in the disc greater than the diameter to zero
        bowl_sm[pixelMapPoint > (diameter / 2)] = 0

    # =========================================================================
    # REMOVE OVERLAPPED POINTS
    # =========================================================================

    if remove_overlap:

        # define the shapes that capture the overlapped points, along with the
        # corresponding mask of which point to delete
        overlap_shapes = []
        overlap_delete = []

        shape = np.zeros((3, 3, 3))
        shape[0, 0, :] = 1
        shape[1, 1, :] = 1
        shape[2, 2, :] = 1
        shape[0, 1, 1] = 1
        shape[1, 2, 1] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[0, 1, 1] = 1
        delete[0, 2, 1] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0, 0, :] = 1
        shape[1, 1, :] = 1
        shape[2, 2, :] = 1
        shape[0, 1, 1] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[0, 1, 1] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0:2, 0, :] = 1
        shape[2, 1, :] = 1
        shape[1, 1, 1] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[1, 1, 1] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0, 0, :] = 1
        shape[1, 1, :] = 1
        shape[2, 2, :] = 1
        shape[0, 1, 0] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[0, 1, 0] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0:2, 1, :] = 1
        shape[2, 2, :] = 1
        shape[2, 1, 0] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[2, 1, 0] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0, :, 2] = 1
        shape[1, :, 1] = 1
        shape[1, :, 0] = 1
        shape[2, 1, 0] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[2, 1, 0] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0, 2, :] = 1
        shape[1, 0:2, :] = 1
        shape[2, 0, 0] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[2, 0, 0] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[:, :, 1] = 1
        shape[0, 0, 0] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[0, 0, 0] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[0, :, 0] = 1
        shape[1, :, 1] = 1
        shape[1, :, 2] = 1
        shape[1, 1, 0] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[1, 1, 0] = 1
        overlap_delete.append(delete)

        shape = np.zeros((3, 3, 3))
        shape[1:3, 2, 0] = 1
        shape[0, 2, 1:3] = 1
        shape[0, 1, 2] = 1
        shape[1, 1, 1] = 1
        shape[2, 1, 0] = 1
        shape[1:3, 0, 1] = 1
        shape[1, 0, 2] = 1
        overlap_shapes.append(shape)

        delete = np.zeros((3, 3, 3))
        delete[1, 0, 1] = 1
        overlap_delete.append(delete)

        # set loop flag
        points_remaining = True

        # initialise deleted point counter
        deleted_points = 0

        # set list of possible permutations
        perm_list = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 2, 0],
            [1, 0, 2],
            [2, 0, 1],
            [2, 1, 0]
        ]

        while points_remaining:

            # get linear index of non-zero bowl elements
            index_mat = matlab_find(bowl_sm > 0)[:, 0]

            # set Boolean delete variable
            delete_point = False

            # loop through all points on the bowl, and find the all the points with
            # more than 8 neighbours
            index = 0
            for index, index_mat_i in enumerate(index_mat):

                # extract subscripts for current point
                cx, cy, cz = ind2sub([Nx, Ny, Nz], index_mat_i)

                # ignore edge points
                if (cx > 1) and (cx < Nx) and (cy > 1) and (cy < Ny) and (cz > 1) and (cz < Nz):

                    # extract local region around current point
                    region = bowl_sm[cx - 1:cx + 1, cy - 1:cy + 1, cz - 1:cz + 1]  # FARID might not work

                    # if there's more than 8 neighbours, check the point for
                    # deletion
                    if (region.sum() - 1) > 8:

                        # loop through the different shapes
                        for shape_index in range(len(overlap_shapes)):

                            # check every permutation of the shape, and apply the
                            # deletion mask if the pattern matches

                            # loop through possible shape permutations
                            for ind1 in range(len(perm_list)):

                                # get shape and delete mask
                                overlap_s = overlap_shapes[shape_index]
                                overlap_d = overlap_delete[shape_index]

                                # permute
                                overlap_s = np.transpose(overlap_s, perm_list[ind1])
                                overlap_d = np.transpose(overlap_d, perm_list[ind1])

                                # loop through possible shape reflections
                                for ind2 in range(1, 8):

                                    # flipfunc the shape
                                    if ind2 == 2:
                                        overlap_s = np.flip(overlap_s, axis=0)
                                        overlap_d = np.flip(overlap_d, axis=0)
                                    elif ind2 == 3:
                                        overlap_s = np.flip(overlap_s, axis=1)
                                        overlap_d = np.flip(overlap_d, axis=1)
                                    elif ind2 == 4:
                                        overlap_s = np.flip(overlap_s, axis=2)
                                        overlap_d = np.flip(overlap_d, axis=2)
                                    elif ind2 == 5:
                                        overlap_s = np.flip(np.flip(overlap_s, axis=0), axis=1)
                                        overlap_d = np.flip(np.flip(overlap_d, axis=0), axis=1)
                                    elif ind2 == 6:
                                        overlap_s = np.flip(np.flip(overlap_s, axis=0), axis=2)
                                        overlap_d = np.flip(np.flip(overlap_d, axis=0), axis=2)
                                    elif ind2 == 7:
                                        overlap_s = np.flip(np.flip(overlap_s, axis=1), axis=2)
                                        overlap_d = np.flip(np.flip(overlap_d, axis=1), axis=2)

                                    # rotate the shape 4 x 90 degrees
                                    for ind3 in range(4):

                                        # check if the shape matches
                                        if np.all(overlap_s == region):
                                            delete_point = True

                                        # break from loop if a match is found
                                        if delete_point:
                                            break

                                        # rotate shape
                                        overlap_s = np.rot90(overlap_s)
                                        overlap_d = np.rot90(overlap_d)

                                    # break from loop if a match is found
                                    if delete_point:
                                        break

                                # break from loop if a match is found
                                if delete_point:
                                    break

                            # remove point from bowl if required, and update
                            # counter
                            if delete_point:
                                bowl_sm[cx - 1:cx + 1, cy - 1:cy + 1, cz - 1:cz + 1] = bowl_sm[cx - 1:cx + 1,
                                                                                       cy - 1:cy + 1,
                                                                                       cz - 1:cz + 1] * np.bitwise_not(
                                    overlap_d).astype(float)  # Farid won't work probably
                                deleted_points = deleted_points + 1
                                break

                # break from loop if a match is found
                if delete_point:
                    break

            # break from while loop if the outer for loop has completed
            # without deleting a point
            if index == (len(index_mat) - 1):
                points_remaining = False

        # display status
        if deleted_points:
            print('{deleted_points} overlapped points removed from bowl')

    # =========================================================================
    # PLACE BOWL WITHIN LARGER GRID
    # =========================================================================

    # preallocate storage variable
    if binary:
        bowl = np.zeros(grid_size, dtype=bool)
    else:
        bowl = np.zeros(grid_size)

    # calculate position of bounding box within larger grid
    x1 = bowl_pos[0] - bx
    x2 = x1 + Nx
    y1 = bowl_pos[1] - by
    y2 = y1 + Ny
    z1 = bowl_pos[2] - bz
    z2 = z1 + Nz

    # truncate bounding box if it falls outside the grid
    if x1 < 0:
        bowl_sm = bowl_sm[abs(x1):, :, :]
        x1 = 0
    if y1 < 0:
        bowl_sm = bowl_sm[:, abs(y1):, :]
        y1 = 0
    if z1 < 0:
        bowl_sm = bowl_sm[:, :, abs(z1):]
        z1 = 0
    if x2 >= grid_size[0]:
        to_delete = x2 - grid_size[0]
        bowl_sm = bowl_sm[:-to_delete, :, :]
        x2 = grid_size[0]
    if y2 >= grid_size[1]:
        to_delete = y2 - grid_size[1]
        bowl_sm = bowl_sm[:, :-to_delete, :]
        y2 = grid_size[1]
    if z2 >= grid_size[2]:
        to_delete = z2 - grid_size[2]
        bowl_sm = bowl_sm[:, :, :-to_delete]
        z2 = grid_size[2]

    # place bowl into grid
    bowl[x1:x2, y1:y2, z1:z2] = bowl_sm

    return bowl


def make_multi_bowl(grid_size: int, bowl_pos: List[Tuple[int, int]], radius: int, diameter: int,
                    focus_pos: Tuple[int, int], binary: bool = False, remove_overlap: bool = False) -> Tuple[
    np.ndarray, List[np.ndarray]]:
    """
    Generates a multi-bowl mask for an image given the size of the grid, the positions of the bowls, the radius of each bowl, the diameter of the bowls, and the position of the focus.

    Args:
        grid_size: The size of the grid (assumed to be square).
        bowl_pos: A list of tuples containing the (x, y) coordinates of the center of each bowl.
        radius: The radius of each bowl.
        diameter: The diameter of the bowls.
        focus_pos: The (x, y) coordinates of the focus.
        binary: Whether to return a binary mask (default: False).
        remove_overlap: Whether to remove overlap between the bowls (default: False).

    Returns:
        bowls:
        bowls_labeled:
    """

    # check inputs
    if bowl_pos.shape[-1] != 3:
        raise ValueError('bowl_pos should contain 3 columns, with [bx, by, bz] in each row.')

    if len(radius) != 1 and len(radius) != bowl_pos.shape[0]:
        raise ValueError('The number of rows in bowl_pos and radius does not match.')

    if len(diameter) != 1 and len(diameter) != bowl_pos.shape[0]:
        raise ValueError('The number of rows in bowl_pos and diameter does not match.')

    # force integer grid size values
    grid_size = np.round(grid_size).astype(int)
    bowl_pos = np.round(bowl_pos).astype(int)
    focus_pos = np.round(focus_pos).astype(int)
    diameter = np.round(diameter)
    radius = np.round(radius)

    # =========================================================================
    # CREATE BOWLS
    # =========================================================================

    # preallocate output matrices
    if binary:
        bowls = np.zeros(grid_size, dtype=bool)
    else:
        bowls = np.zeros(grid_size)

    bowls_labelled = np.zeros(grid_size)

    # loop for calling make_bowl
    for bowl_index in range(bowl_pos.shape[0]):

        # update command line status
        if bowl_index == 1:
            TicToc.tic()
        else:
            TicToc.toc(reset=True)
        print(f'Creating bowl {bowl_index} of {bowl_pos.shape[0]} ... ')

        # get parameters for current bowl
        if bowl_pos.shape[0] > 1:
            bowl_pos_k = bowl_pos[bowl_index]
        else:
            bowl_pos_k = bowl_pos

        if len(radius) > 1:
            radius_k = radius[bowl_index]
        else:
            radius_k = radius

        if len(diameter) > 1:
            diameter_k = diameter[bowl_index]
        else:
            diameter_k = diameter

        if focus_pos.shape[0] > 1:
            focus_pos_k = focus_pos[bowl_index]
        else:
            focus_pos_k = focus_pos

        # create new bowl
        new_bowl = make_bowl(
            grid_size, bowl_pos_k, radius_k, diameter_k, focus_pos_k,
            remove_overlap=remove_overlap, binary=binary
        )

        # add bowl to bowl matrix
        bowls = bowls + new_bowl

        # add new bowl to labelling matrix
        bowls_labelled[new_bowl == 1] = bowl_index

    TicToc.toc()

    # check if any of the bowls are overlapping
    max_nd_val, _ = max_nd(bowls)
    if max_nd_val > 1:
        # display warning
        print(f'WARNING: {max_nd_val - 1} bowls are overlapping')

        # force the output to be binary
        bowls[bowls != 0] = 1

    return bowls, bowls_labelled


def make_multi_arc(grid_size: np.ndarray, arc_pos: np.ndarray, radius: Union[int, np.ndarray],
                   diameter: Union[int, np.ndarray], focus_pos: np.ndarray) -> np.ndarray:
    """
    Generates a multi-arc mask for an image given the size of the grid, the positions and properties of the arcs, and the position of the focus.

    Args:
        grid_size: The size of the grid (assumed to be square).
        arc_pos: An array containing the (x, y) coordinates of the center of each arc.
        radius: The radius of each arc. Can be a single value or an array with one value for each arc.
        diameter: The diameter of the arcs. Can be a single value or an array with one value for each arc.
        focus_pos: The (x, y) coordinates of the focus.

    Returns:
        arcs: A binary mask of the arcs.
        arcs_labelled: A labelled mask of the arcs.

    Raises:
        ValueError: If the shape of arc_pos is not (N, 2), if the number of rows in arc_pos and radius do not match, or if the number of rows in arc_pos and diameter do not match.
    """
    # check inputs
    if arc_pos.shape[-1] != 2:
        raise ValueError('arc_pos should contain 2 columns, with [ax, ay] in each row.')

    if len(radius) != 1 and len(radius) != arc_pos.shape[0]:
        raise ValueError('The number of rows in arc_pos and radius does not match.')

    if len(diameter) != 1 and len(diameter) != arc_pos.shape[0]:
        raise ValueError('The number of rows in arc_pos and diameter does not match.')

    # force integer grid size values
    grid_size = grid_size.round().astype(int)
    arc_pos = arc_pos.round().astype(int)
    diameter = diameter.round()
    focus_pos = focus_pos.round().astype(int)
    radius = radius.round()

    # =========================================================================
    # CREATE ARCS
    # =========================================================================

    # create empty matrix
    arcs = np.zeros(grid_size)
    arcs_labelled = np.zeros(grid_size)

    # loop for calling make_arc
    for k in range(arc_pos.shape[0]):

        # get parameters for current arc
        if arc_pos.shape[0] > 1:
            arc_pos_k = arc_pos[k]
        else:
            arc_pos_k = arc_pos

        if len(radius) > 1:
            radius_k = radius[k]
        else:
            radius_k = radius

        if len(diameter) > 1:
            diameter_k = diameter[k]
        else:
            diameter_k = diameter

        if focus_pos.shape[0] > 1:
            focus_pos_k = focus_pos[k]
        else:
            focus_pos_k = focus_pos

        # create new arc
        new_arc = make_arc(grid_size, arc_pos_k, radius_k, diameter_k, focus_pos_k)

        # add arc to arc matrix
        arcs = arcs + new_arc

        # add new arc to labelling matrix
        arcs_labelled[new_arc == 1] = k

    # check if any of the arcs are overlapping
    max_nd_val, _ = max_nd(arcs)
    if max_nd_val > 1:
        # display warning
        print(f'WARNING: {max_nd_val - 1} arcs are overlapping')

        # force the output to be binary
        arcs[arcs != 0] = 1

    return arcs, arcs_labelled


def make_sphere(Nx: int, Ny: int, Nz: int, radius: float, plot_sphere: bool = False,
                binary: bool = False) -> np.ndarray:
    """
    Generates a sphere mask for a 3D grid given the dimensions of the grid, the radius of the sphere, and optional flags to plot the sphere and/or return a binary mask.

    Args:
        Nx: The number of grid points in the x-dimension.
        Ny: The number of grid points in the y-dimension.
        Nz: The number of grid points in the z-dimension.
        radius: The radius of the sphere.
        plot_sphere: Whether to plot the sphere (default: False).
        binary: Whether to return a binary mask (default: False).

    Returns:
        sphere: The sphere mask as a NumPy array.
    """
    # enforce a centered sphere
    cx = floor(Nx / 2) + 1
    cy = floor(Ny / 2) + 1
    cz = floor(Nz / 2) + 1

    # preallocate the storage variable
    if binary:
        sphere = np.zeros((Nx, Ny, Nz), dtype=bool)
    else:
        sphere = np.zeros((Nx, Ny, Nz))

    # create a guide circle from which the individal radii can be extracted
    guide_circle = make_circle(Ny, Nx, cy, cx, radius)

    # step through the guide circle points and create partially filled discs
    centerpoints = np.arange(cx - radius, cx + 1)
    reflection_offset = np.arange(len(centerpoints), 1, -1)
    for centerpoint_index in range(len(centerpoints)):

        # extract the current row from the guide circle
        row_data = guide_circle[:, centerpoints[centerpoint_index] - 1]

        # add an index to the grid points in the current row
        row_index = row_data * np.arange(1, len(row_data) + 1)

        # calculate the radius
        swept_radius = (row_index.max() - row_index[row_index != 0].min()) / 2

        # create a circle to add to the sphere
        circle = make_circle(Ny, Nz, cy, cz, swept_radius)

        # make an empty fill matrix
        if binary:
            circle_fill = np.zeros((Ny, Nz), dtype=bool)
        else:
            circle_fill = np.zeros((Ny, Nz))

        # fill in the circle line by line
        fill_centerpoints = np.arange(cz - swept_radius, cz + swept_radius + 1).astype(int)
        for fill_centerpoints_i in fill_centerpoints:

            # extract the first row
            row_data = circle[:, fill_centerpoints_i - 1]

            # add an index to the grid points in the current row
            row_index = row_data * np.arange(1, len(row_data) + 1)

            # calculate the diameter
            start_index = row_index[row_index != 0].min()
            stop_index = row_index.max()

            # count how many points on the line
            num_points = sum(row_data)

            # fill in the line
            if start_index != stop_index and (stop_index - start_index) >= num_points:
                circle_fill[(start_index + num_points // 2) - 1:stop_index - (num_points // 2),
                fill_centerpoints_i - 1] = 1

        # remove points from the filled circle that existed in the previous
        # layer
        if centerpoint_index == 0:
            sphere[centerpoints[centerpoint_index] - 1, :, :] = circle + circle_fill
            prev_circle = circle + circle_fill
        else:
            prev_circle_alt = circle + circle_fill
            circle_fill = circle_fill - prev_circle
            circle_fill[circle_fill < 0] = 0
            sphere[centerpoints[centerpoint_index] - 1, :, :] = circle + circle_fill
            prev_circle = prev_circle_alt

        # create the other half of the sphere at the same time
        if centerpoint_index != len(centerpoints) - 1:
            sphere[cx + reflection_offset[centerpoint_index] - 2, :, :] = sphere[centerpoints[centerpoint_index] - 1, :,
                                                                          :]

    # plot results
    if plot_sphere:
        raise NotImplementedError
    return sphere


def make_spherical_section(radius: float, height: float, width: float = None, plot_section: bool = False,
                           binary: bool = False) -> np.ndarray:
    """
    Generates a spherical section mask given the radius and height of the section and optional parameters to specify the width and/or plot and return a binary mask.

    Args:
        radius: The radius of the spherical section.
        height: The height of the spherical section.
        width: The width of the spherical section (default: height).
        plot_section: Whether to plot the spherical section (default: False).
        binary: Whether to return a binary mask (default: False).

    Returns:
        ss: The spherical section mask as a NumPy array.
        dist_map: The distance map of the spherical section mask as a NumPy array.

    Raises:
        ValueError: If the width is not an odd number.
        NotImplementedError: Plotting not currently supported.
    """
    use_spherical_sections = True

    # force inputs to be integers
    radius = int(radius)
    height = int(height)

    use_width = (width is not None)
    if use_width:
        width = int(width)
        if width % 2 == 0:
            raise ValueError('Input width must be an odd number.')

    # calculate minimum grid dimensions to fit entire sphere
    Nx = 2 * radius + 1

    # create sphere
    ss = make_sphere(Nx, Nx, Nx, radius, False, binary)

    # truncate to given height
    if use_spherical_sections:
        ss = ss[:height, :, :]
    else:
        ss = np.transpose(ss[:, :height, :], [1, 2, 0])

    # flatten transducer and store the maximum and indices
    mx = np.squeeze(np.max(ss, axis=0))

    # calculate the total length/width of the transducer
    length = mx[(len(mx) + 1) // 2].sum()

    # truncate transducer grid based on length (removes empty rows and columns)
    offset = int((Nx - length) / 2)
    ss = ss[:, offset:-offset, offset:-offset]

    # also truncate to given width if defined by user
    if use_width:

        # check the value is appropriate
        if width > length:
            raise ValueError('Input for width must be less than or equal to transducer length.')

        # calculate offset
        offset = int((length - width) / 2)

        # truncate transducer grid
        ss = ss[:, offset:-offset, :]

    # compute average distance between each grid point and its contiguous

    # calculate x-index of each grid point in the spherical section, create
    # mask and remove singleton dimensions
    mx, mx_ind = np.max(ss, axis=0), ss.argmax(axis=0) + 1
    mask = np.squeeze(mx != 0)
    mx_ind = np.squeeze(mx_ind) * mask

    # double check there there is only one value of spherical section in
    # each matrix column
    if mx.sum() != ss.sum():
        raise ValueError(
            'mean neighbour distance cannot be calculated uniquely due to overlapping points in the x-direction')

    # calculate average distance to grid point neighbours in the flat case
    x_dist = np.tile([1, 0, 1], [3, 1])
    y_dist = x_dist.T
    flat_dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
    flat_dist = np.mean(flat_dist)

    # compute distance map
    dist_map = np.zeros(mx_ind.shape)
    sz = mx_ind.shape
    for m in range(sz[0]):
        for n in range(sz[1]):

            # clear map
            local_heights = np.zeros((3, 3))

            # extract the height (x-distance) of the 8 neighbouring grid
            # points
            if m == 0 and n == 0:
                local_heights[1:3, 1:3] = mx_ind[m:m + 2, n:n + 2]
            elif m == (sz[0] - 1) and n == (sz[1] - 1):
                local_heights[0:2, 0:2] = mx_ind[m - 1:m + 1, n - 1:n + 1]
            elif m == 0 and n == (sz[1] - 1):
                local_heights[1:3, 0:2] = mx_ind[m:m + 2, n - 1:n + 1]
            elif m == (sz[0] - 1) and n == 0:
                local_heights[0:2, 1:3] = mx_ind[m - 1:m + 1, n:n + 2]
            elif m == 0:
                local_heights[1:3, :] = mx_ind[m:m + 2, n - 1:n + 2]
            elif m == (sz[0] - 1):
                local_heights[0:2, :] = mx_ind[m - 1:m + 1, n - 1:n + 2]
            elif n == 0:
                local_heights[:, 1:3] = mx_ind[m - 1:m + 2, n:n + 2]
            elif n == (sz[1] - 1):
                local_heights[:, 0:2] = mx_ind[m - 1:m + 2, n - 1:n + 1]
            else:
                local_heights = mx_ind[m - 1:m + 2, n - 1:n + 2]

            # compute average variation from center
            local_heights_var = abs(local_heights - local_heights[1, 1])

            # threshold no neighbours
            local_heights_var[local_heights == 0] = 0

            # calculate total distance from centre
            dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + local_heights_var ** 2)

            # average and store as a ratio
            dist_map[m, n] = 1 + (np.mean(dist) - flat_dist) / flat_dist

    # threshold out the non-transducer grid points
    dist_map[mask != 1] = 0

    # plot if required
    if plot_section:
        raise NotImplementedError

    return ss, dist_map


def focused_bowl_oneil(radius: float, diameter: float, velocity: float, frequency: float, sound_speed: float,
                       density: float, axial_positions: Union[np.ndarray, float, list] = None,
                       lateral_positions: Union[np.ndarray, float, list] = None) -> [float, float]:
    """
    Calculates O'Neil's solution for the axial and lateral pressure amplitude generated by a focused bowl transducer.

    Args:
        radius: The radius of the transducer.
        diameter: The diameter of the transducer.
        velocity: The normal surface velocity of the transducer.
        frequency: The driving frequency of the sinusoid.
        sound_speed: The sound speed in the medium.
        density: The density of the medium.
        axial_positions: The positions along the beam axis where the pressure is evaluated (0 corresponds to the transducer surface). Set to [] to return only lateral pressure.
        lateral_positions: The lateral positions through the geometric focus where the pressure is evaluated (0 corresponds to the beam axis). Set to [] to return only axial pressure.

    Returns:
       p-axial: The axial pressure amplitude.
       p-lateral: The lateral pressure amplitude.

    Example:
        # define transducer parameters
        radius = 140e-3  # [m]
        diameter = 120e-3  # [m]
        velocity = 100e-3  # [m / s]
        frequency = 1e6  # [Hz]
        sound_speed = 1500  # [m / s]
        density = 1000  # [kg / m ^ 3]

        # define position vectors
        axial_position = np.arange(0, 250e-3 + 1e-4, 1e-4)  # [m]
        lateral_position = np.arange(-15e-3, 15e-3 + 1e-4, 1e-4)  # [m]

        # evaluate pressure
        [p_axial, p_lateral] = focused_bowl_oneil(radius, diameter,
                                                  velocity, frequency, sound_speed, density,
                                                  axial_position, lateral_position)

    References:
        O'Neil, H. (1949). Theory of focusing radiators. J. Acoust. Soc. Am., 21(5), 516-526.
    """

    float_eps = np.finfo(float).eps

    def calculate_axial_pressure() -> float:
        # calculate distances
        B = np.sqrt((axial_positions - h) ** 2 + (diameter / 2) ** 2)
        d = B - axial_positions
        E = 2 / (1 - axial_positions / radius)

        # compute pressure
        P = E * np.sin(k * d / 2)

        # replace values where axial_position is equal to the radius with limit
        P[np.abs(axial_positions - radius) < float_eps] = k * h

        # calculate magnitude of the on - axis pressure
        axial_pressure = density * sound_speed * velocity * np.abs(P)
        return axial_pressure

    def calculate_lateral_pressure() -> float:
        # calculate magnitude of the lateral pressure at the geometric focus
        Z = k * lateral_positions * diameter / (2 * radius)
        lateral_pressure = 2. * density * sound_speed * velocity * k * h * scipy.special.jv(1, Z) / Z

        # replace origin with limit
        lateral_pressure[lateral_positions == 0] = density * sound_speed * velocity * k * h
        return lateral_pressure

    # wave number
    k = 2 * np.pi * frequency / sound_speed

    # height of rim
    h = radius - np.sqrt(radius ** 2 - (diameter / 2) ** 2)

    p_axial = None
    p_lateral = None

    if lateral_positions is not None:
        p_lateral = calculate_lateral_pressure()
    if axial_positions is not None:
        p_axial = calculate_axial_pressure()

    return p_axial, p_lateral

    return [p_axial, p_lateral]


def ndgrid(*args):
    return np.array(np.meshgrid(*args, indexing='ij'))
