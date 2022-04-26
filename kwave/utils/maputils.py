import math
from math import floor

import numpy as np


def makeBall(Nx, Ny, Nz, cx, cy, cz, radius, plot_ball=False, binary=False):
    """
    %MAKEBALL Create a binary map of a filled ball within a 3D grid.
    %
    % DESCRIPTION:
    %     makeBall creates a binary map of a filled ball within a
    %     three-dimensional grid (the ball position is denoted by 1's in the
    %     matrix with 0's elsewhere). A single grid point is taken as the ball
    %     centre thus the total diameter of the ball will always be an odd
    %     number of grid points.
    Args:
        Nx: size of the 3D grid in x-dimension [grid points]
        Ny: size of the 3D grid in y-dimension [grid points]
        Nz: size of the 3D grid in z-dimension [grid points]
        cx: centre of the ball in x-dimension [grid points]
        cy: centre of the ball in y-dimension [grid points]
        cz: centre of the ball in z-dimension [grid points]
        radius: ball radius [grid points]
        plot_ball: Boolean controlling whether the ball is plotted using voxelPlot (default = false)
        binary: Boolean controlling whether the ball map is returned as a double precision matrix (false)
                or a logical matrix (true) (default = false)

    Returns:
        3D binary map of a filled ball
    """
    # define literals
    MAGNITUDE = 1

    # force integer values
    Nx = int(round(Nx))
    Ny = int(round(Ny))
    Nz = int(round(Nz))
    cx = int(round(cx))
    cy = int(round(cy))
    cz = int(round(cz))

    # check for zero values
    if cx == 0:
        cx = int(floor(Nx / 2)) + 1

    if cy == 0:
        cy = int(floor(Ny / 2)) + 1

    if cz == 0:
        cz = int(floor(Nz / 2)) + 1

    # create empty matrix
    ball = np.zeros((Nx, Ny, Nz)).astype(np.bool if binary else np.float32)

    # define pixel map
    r = makePixelMap(Nx, Ny, Nz, 'Shift', [0, 0, 0])

    # create ball
    ball[r <= radius] = MAGNITUDE

    # shift centre
    cx = cx - int(math.ceil(Nx / 2))
    cy = cy - int(math.ceil(Ny / 2))
    cz = cz - int(math.ceil(Nz / 2))
    ball = np.roll(ball, (cx, cy, cz), axis=(0, 1, 2))

    # plot results
    if plot_ball:
        raise NotImplementedError
        # voxelPlot(double(ball))
    return ball


def makeDisc(Nx, Ny, cx, cy, radius, plot_disc=False):
    """
        Create a binary map of a filled disc within a 2D grid.

        %     makeDisc creates a binary map of a filled disc within a
        %     two-dimensional grid (the disc position is denoted by 1's in the
        %     matrix with 0's elsewhere). A single grid point is taken as the disc
        %     centre thus the total diameter of the disc will always be an odd
        %     number of grid points. As the returned disc has a constant radius, if
        %     used within a k-Wave grid where dx ~= dy, the disc will appear oval
        %     shaped. If part of the disc overlaps the grid edge, the rest of the
        %     disc will wrap to the grid edge on the opposite side.
    Args:
        Nx:
        Ny:
        cx:
        cy:
        radius:
        plot_disc:

    Returns:

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

    # define pixel map
    r = makePixelMap(Nx, Ny, None, 'Shift', [0, 0])

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


def makeCircle(Nx, Ny, cx, cy, radius, arc_angle=None, plot_circle=False):
    """
        Create a binary map of a circle within a 2D grid.

        %     makeCircle creates a binary map of a circle or arc (using the
        %     midpoint circle algorithm) within a two-dimensional grid (the circle
        %     position is denoted by 1's in the matrix with 0's elsewhere). A
        %     single grid point is taken as the circle centre thus the total
        %     diameter will always be an odd number of grid points.
        %
        %     Note: The centre of the circle and the radius are not constrained by
        %     the grid dimensions, so it is possible to create sections of circles,
        %     or a blank image if none of the circle intersects the grid.
    Args:
        Nx:
        Ny:
        cx:
        cy:
        radius:
        plot_disc:

    Returns:

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
    circle = np.zeros((Nx, Ny))

    # initialise loop variables
    x = 0
    y = radius
    d = 1 - radius

    if (cx >= 1) and (cx <= Nx) and ((cy - y) >= 1) and ((cy - y) <= Ny):
        circle[cx, cy - y] = MAGNITUDE

    # draw the remaining cardinal points
    px = [cx, cx + y, cx - y]
    py = [cy + y, cy, cy]
    for point_index in range(len(px)):
        # check whether the point is within the arc made by arc_angle, and lies
        # within the grid
        if (np.arctan2(px[point_index] - cx, py[point_index] - cy) + np.pi) <= arc_angle:
            if ( px[point_index] >= 1 ) and ( px[point_index] <= Nx ) and ( py[point_index] >= 1 ) and ( py[point_index] <= Ny ):
                circle[px[point_index], py[point_index]] = MAGNITUDE

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
        px = [x+cx, y+cx,  y+cx,  x+cx, -x+cx, -y+cx, -y+cx, -x+cx]
        py = [y+cy, x+cy, -x+cy, -y+cy, -y+cy, -x+cy,  x+cy,  y+cy]

        # loop through each point
        for point_index in range(len(px)):

            # check whether the point is within the arc made by arc_angle, and
            # lies within the grid
            if (np.arctan2(px[point_index] - cx, py[point_index] - cy) + np.pi) <= arc_angle:
                if ( px[point_index] >= 1 ) and ( px[point_index] <= Nx ) and ( py[point_index] >= 1 ) and ( py[point_index] <= Ny ):
                    circle[px[point_index], py[point_index]] = MAGNITUDE

    if plot_circle:
        raise NotImplementedError

    return circle


def makeCartCircle(radius, num_points, center_pos=None, arc_angle=(2*np.pi), plot_circle=False):
    """
        Create a 2D Cartesian circle or arc.

        %     MakeCartCircle creates a 2 x num_points array of the Cartesian
        %     coordinates of points evenly distributed over a circle or arc (if
        %     arc_angle is given).
    Args:

    Returns:

    """
    full_circle = (arc_angle == 2 * np.pi)

    if center_pos is None:
        cx = cy = 0
    else:
        cx, cy = center_pos

    # ensure there is only a total of num_points including the endpoints when
    # arc_angle is not equal to 2*pi
    if not full_circle:
        num_points = num_points - 1

    # create angles
    angles = np.arange(0, num_points + 1) * arc_angle / num_points + np.pi/2

    # discard repeated final point if arc_angle is equal to 2*pi
    if full_circle:
        angles = angles[0:- 1]

    # create cartesian grid
    # circle = flipud([radius*cos(angles); radius*sin(-angles)]);        # B.0.3
    circle = np.vstack([radius*np.cos(angles), radius*np.sin(-angles)])  # B.0.4

    # offset if needed
    circle[0, :] = circle[0, :] + cx
    circle[1, :] = circle[1, :] + cy

    if plot_circle:
        raise NotImplementedError

    return circle


def makePixelMap(Nx, Ny, Nz=None, *args):
    """
    %MAKEPIXELMAP Create matrix of grid point distances from the centre point.
    %
    % DESCRIPTION:
    %     makePixelMap generates a matrix populated with values of how far each
    %     pixel in a grid is from the centre (given in pixel coordinates). Both
    %     single and double pixel centres can be used by setting the optional
    %     input parameter 'OriginSize'. For grids where the dimension size and
    %     centre pixel size are not both odd or even, the optional input
    %     parameter 'Shift' can be used to control the location of the
    %     centerpoint.
    %
    %     examples for a 2D pixel map:
    %
    %     Single pixel origin size for odd and even (with 'Shift' = [1 1] and
    %     [0 0], respectively) grid sizes:
    %
    %     x x x       x x x x         x x x x
    %     x 0 x       x x x x         x 0 x x
    %     x x x       x x 0 x         x x x x
    %                 x x x x         x x x x
    %
    %     Double pixel origin size for even and odd (with 'Shift' = [1 1] and
    %     [0 0], respectively) grid sizes:
    %
    %     x x x x      x x x x x        x x x x x
    %     x 0 0 x      x x x x x        x 0 0 x x
    %     x 0 0 x      x x 0 0 x        x 0 0 x x
    %     x x x x      x x 0 0 x        x x x x x
    %                  x x x x x        x x x x x
    %
    %     By default a single pixel centre is used which is shifted towards
    %     the final row and column.
    Args:
        *args:

    Returns:

    """
    # define defaults
    origin_size = 'single'
    shift_def = 1

    # detect whether the inputs are for two or three dimensions
    if Nz is None:
        map_dimension = 2
        shift = [shift_def, shift_def]
    else:
        map_dimension = 3
        shift = [shift_def, shift_def, shift_def]

    # replace with user defined values if provided
    if len(args) > 0:
        assert len(args) % 2 == 0, 'Optional inputs must be entered as param, value pairs.'
        for input_index in range(0, len(args), 2):
            if args[input_index] == 'Shift':
                shift = args[input_index + 1]
            elif args[input_index] == 'OriginSize':
                origin_size = args[input_index + 1]
            else:
                raise ValueError('Unknown optional input.')

    # catch input errors
    assert origin_size in ['single', 'double'], 'Unknown setting for optional input Center.'

    assert len(shift) == map_dimension, f'Optional input Shift must have {map_dimension} elements for {map_dimension} dimensional input parameters.'

    if map_dimension == 2:
        # create the maps for each dimension
        nx = createPixelDim(Nx, origin_size, shift[0])
        ny = createPixelDim(Ny, origin_size, shift[1])

        # create plaid grids
        r_x, r_y = np.meshgrid(nx, ny, indexing='ij')

        # extract the pixel radius
        r = np.sqrt(r_x**2 + r_y**2)
    if map_dimension == 3:
        # create the maps for each dimension
        nx = createPixelDim(Nx, origin_size, shift[0])
        ny = createPixelDim(Ny, origin_size, shift[1])
        nz = createPixelDim(Nz, origin_size, shift[2])

        # create plaid grids
        r_x, r_y, r_z = np.meshgrid(nx, ny, nz, indexing='ij')

        # extract the pixel radius
        r = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    return r


def createPixelDim(Nx, origin_size, shift):
    # Nested function to create the pixel radius variable

    # grid dimension has an even number of points
    if Nx % 2 == 0:

        # pixel numbering has a single centre point
        if origin_size == 'single':

            # centre point is shifted towards the final pixel
            if shift == 1:
                nx = np.arange(-Nx/2, Nx/2-1 + 1, 1)

            # centre point is shifted towards the first pixel
            else:
                nx = np.arange(-Nx/2+1, Nx/2 + 1, 1)

        # pixel numbering has a double centre point
        else:
            nx = np.hstack([np.arange(-Nx/2+1, 0 + 1, 1), np.arange(0, -Nx/2-1 + 1, 1)])

    # grid dimension has an odd number of points
    else:

        # pixel numbering has a single centre point
        if origin_size == 'single':
            nx = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1, 1)

        # pixel numbering has a double centre point
        else:

            # centre point is shifted towards the final pixel
            if shift == 1:
                nx = np.hstack([np.arange(-(Nx-1)/2, 0 + 1, 1), np.arange(0, (Nx-1)/2-1 + 1, 1)])

            # centre point is shifted towards the first pixel
            else:
                nx = np.hstack([np.arange(-(Nx-1)/2+1, 0 + 1, 1), np.arange(0, (Nx-1)/2 + 1, 1)])
    return nx