import math
from math import floor

import numpy as np


def water_absorption(f, temp):
    """
    WATERABSORPTION Calculate ultrasound absorption in distilled water.

    DESCRIPTION:
    waterAbsorption calculates the ultrasonic absorption in distilled
    water at a given temperature and frequency using a 7 th order
    polynomial fitted to the data given by Pinkerton(1949).

    USAGE:
    abs = waterAbsorption(f, T)

    INPUTS:
    f - f frequency value [MHz]
    T - water temperature value [degC]

    OUTPUTS:
    abs - absorption[dB / cm]

    ABOUT:
    author - Bradley Treeby
    date - 10 th November 2008
    last update - 4 th April 2019

    REFERENCES:
    [1] Pinkerton(1949) "The Absorption of Ultrasonic Waves in Liquids
    and its Relation to Molecular Constitution, " Proceedings of the
    Physical Society.Section B, 2, 129 - 141

    This function is part of the k - Wave Toolbox(http: // www.k - wave.org)
    Copyright(C) 2008 - 2019 Bradley Treeby

    See also waterDensity, waterNonlinearity, waterSoundSpeed

    This file is part of k - Wave.k - Wave is free software: you can
    redistribute it and / or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    k - Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.See the GNU Lesser General Public License for
    more details.

    You should have received a copy of the GNU Lesser General Public License
    along with k - Wave.If not, see < http:// www.gnu.org / licenses / >.
    """
    NEPER2DB = 8.686
    # check temperature is within range
    if not 0 <= temp <= 60:
        raise Warning("Temperature outside range of experimental data")

    # conversion factor between Nepers and dB NEPER2DB = 8.686;
    # coefficients for 7th order polynomial fit
    a = [56.723531840522710, -2.899633796917384, 0.099253401567561, -0.002067402501557, 2.189417428917596e-005,
         -6.210860973978427e-008, -6.402634551821596e-010, 3.869387679459408e-012]

    # TODO: this is not a vectorized version of this function. This is different than the MATLAB version
    # make sure vectors are in the correct orientation
    # T = reshape(T, 3, []);
    # f = reshape(f, [], 1);

    # compute absorption
    a_on_fsqr = (a[0] + a[1] * temp + a[2] * temp ** 2 + a[3] * temp ** 3 + a[4] * temp ** 4 + a[5] * temp ** 5 + a[
        6] * temp ** 6 + a[7] * temp ** 7) * 1e-17

    abs = NEPER2DB * 1e12 * f ** 2 * a_on_fsqr
    return abs


def hounsfield2soundspeed(ct_data):
    """

    Calclulates the soundspeed of a medium given a CT of the medium.
    For soft-tissue, the approximate sound speed can also be returned
    using the empirical relationship given by Mast.
    Args:
        ct_data: 

    Returns: sound_speed:       matrix of sound speed values of size of ct data

    """
    # calculate corresponding sound speed values if required using soft tissue  relationship
    # TODO confirm that this linear relationship is correct
    sound_speed = (hounsfield2density(ct_data) + 349) / 0.893

    return sound_speed


def hounsfield2density(ct_data, plot_fitting=False):
    """
    Convert Hounsfield units in CT data to density values [kg / m ^ 3]
    based on the experimental data given by Schneider et al.
    The conversion is made using a piece-wise linear fit to the data.

    Args:
        ct_data:
        plot_fitting:

    Returns:
        density_map:            density in kg / m ^ 3
    """
    # create empty density matrix
    density = np.zeros(ct_data.shape, like=ct_data)

    # apply conversion in several parts using linear fits to the data
    # Part 1: Less than 930 Hounsfield Units
    density[ct_data < 930] = np.polyval([1.025793065681423, -5.680404011488714], ct_data[ct_data < 930])

    # Part 2: Between 930 and 1098(soft tissue region)
    index_selection = np.logical_and(930 <= ct_data, ct_data <= 1098)
    density[index_selection] = np.polyval([0.9082709691264, 103.6151457847139],
                                          ct_data[index_selection])

    # Part 3: Between 1098 and 1260(between soft tissue and bone)
    index_selection = np.logical_and(1098 < ct_data, ct_data < 1260)
    density[index_selection] = np.polyval([0.5108369316599, 539.9977189228704], ct_data[index_selection])

    # Part 4: Greater than 1260(bone region)
    density[ct_data >= 1260] = np.polyval([0.6625370912451, 348.8555178455294], ct_data[ct_data >= 1260])

    if plot_fitting:
        raise NotImplementedError("Plotting function not implemented in Python")

    return density


def water_sound_speed(temp):
    """
    WATERSOUNDSPEED Calculate the sound speed in distilled water with temperature.

     DESCRIPTION:
         waterSoundSpeed calculates the sound speed in distilled water at a
         a given temperature using the 5th order polynomial given by Marczak
         (1997).

     USAGE:
         c = waterSoundSpeed(T)

     INPUTS:
         T             - water temperature in the range 0 to 95 [degC]

     OUTPUTS:
         c             - sound speed [m/s]

     ABOUT:
         author        - Bradley E. Treeby
         date          - 11th August 2008
         last update   - 4th April 2019

     REFERENCES:
         [1] Marczak (1997) "Water as a standard in the measurements of speed
         of sound in liquids," J. Acoust. Soc. Am., 102, 2776-2779.

     This function is part of the k-Wave Toolbox (http://www.k-wave.org)
     Copyright (C) 2008-2019 Bradley Treeby

     See also waterAbsorption, waterDensity, waterNonlinearity

     This file is part of k-Wave. k-Wave is free software: you can
     redistribute it and/or modify it under the terms of the GNU Lesser
     General Public License as published by the Free Software Foundation,
     either version 3 of the License, or (at your option) any later version.

     k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
     FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
     more details.

     You should have received a copy of the GNU Lesser General Public License
     along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
    """

    # check limits
    assert 95 >= temp >= 0, "temp must be between 0 and 95."

    # find value
    p = [2.787860e-9, -1.398845e-6, 3.287156e-4, -5.779136e-2, 5.038813, 1.402385e3]
    c = np.polyval(p, temp)
    return c


def water_density(temp):
    """
     WATERDENSITY Calculate density of air - saturated water with temperature.

     DESCRIPTION:
     waterDensity calculates the density of air - saturated water at a given % temperature using the 4 th order polynomial given by Jones(1992).

     USAGE:
     density = waterDensity(T)

     INPUTS:
     T - water temperature in the range 5 to 40[degC]

     OUTPUTS:
     density - density of water[kg / m ^ 3]

     ABOUT:
     author - Bradley E.Treeby
     date - 22 nd February 2018
     last update - 4 th April 2019

     REFERENCES:
     [1] F.E.Jones and G.L.Harris(1992) "ITS-90 Density of Water
     Formulation for Volumetric Standards Calibration, " J. Res. Natl.
     Inst.Stand.Technol., 97(3), 335 - 340.

     This function is part of the k - Wave Toolbox(http: // www.k - wave.org)
     Copyright(C) 2018 - 2019 Bradley Treeby

     See also waterAttenuation, waterNonlinearity, waterSoundSpeed

     This file is part of k - Wave.k - Wave is free software: you can
     redistribute it and / or modify it under the terms of the GNU Lesser
     General Public License as published by the Free Software Foundation,
     either version 3 of the License, or (at your option) any later version.

     k - Wave is distributed in the hope that it will be useful, but WITHOUT ANY
     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
     FOR A PARTICULAR PURPOSE.See the GNU Lesser General Public License for
     more details.

     You should have received a copy of the GNU Lesser General Public License
     along with k - Wave.If not, see < http:// www.gnu.org / licenses / >.
     """
    # check limits
    assert 5 <= temp <= 40, "T must be between 5 and 40."

    # calculate density of air - saturated water
    density = 999.84847 + 6.337563e-2 * temp - 8.523829e-3 * temp ** 2 + 6.943248e-5 * temp ** 3 - 3.821216e-7 * temp ** 4
    return density


def water_non_linearity(temp):
    """
    # WATERNONLINEARITY Calculate B/A of water with temperature.
    #
    # DESCRIPTION:
    #     waterNonlinearity calculates the parameter of nonlinearity B/A at a
    #     given temperature using a fourth-order polynomial fitted to the data
    #     given by Beyer (1960).
    #
    # USAGE:
    #     BonA = waterNonlinearity(T)
    #
    # INPUTS:
    #     T             - water temperature in the range 0 to 100 [degC]
    #
    # OUTPUTS:
    #     BonA          - parameter of nonlinearity
    #
    # ABOUT:
    #     author        - Bradley E. Treeby
    #     date          - 22nd February 2018
    #     last update   - 4th April 2019
    #
    # REFERENCES:
    #     [1] R. T Beyer (1960) "Parameter of nonlinearity in fluids," J.
    #     Acoust. Soc. Am., 32(6), 719-721.
    #
    # This function is part of the k-Wave Toolbox (http://www.k-wave.org)
    # Copyright (C) 2018-2019 Bradley Treeby
    #
    # See also waterAbsorption, waterDensity, waterSoundSpeed

    # This file is part of k-Wave. k-Wave is free software: you can
    # redistribute it and/or modify it under the terms of the GNU Lesser
    # General Public License as published by the Free Software Foundation,
    # either version 3 of the License, or (at your option) any later version.
    #
    # k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    # WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    # FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
    # more details.
    #
    # You should have received a copy of the GNU Lesser General Public License
    # along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
    """

    # check limits
    assert 0 <= temp <= 100, "Temp must be between 0 and 100."

    # find value
    p = [-4.587913769504693e-08, 1.047843302423604e-05, -9.355518377254833e-04, 5.380874771364909e-2, 4.186533937275504]
    BonA = np.polyval(p, temp);
    return BonA


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
            if (px[point_index] >= 1) and (px[point_index] <= Nx) and (py[point_index] >= 1) and (
                    py[point_index] <= Ny):
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
        px = [x + cx, y + cx, y + cx, x + cx, -x + cx, -y + cx, -y + cx, -x + cx]
        py = [y + cy, x + cy, -x + cy, -y + cy, -y + cy, -x + cy, x + cy, y + cy]

        # loop through each point
        for point_index in range(len(px)):

            # check whether the point is within the arc made by arc_angle, and
            # lies within the grid
            if (np.arctan2(px[point_index] - cx, py[point_index] - cy) + np.pi) <= arc_angle:
                if (px[point_index] >= 1) and (px[point_index] <= Nx) and (py[point_index] >= 1) and (
                        py[point_index] <= Ny):
                    circle[px[point_index], py[point_index]] = MAGNITUDE

    if plot_circle:
        raise NotImplementedError

    return circle


def makeCartCircle(radius, num_points, center_pos=None, arc_angle=(2 * np.pi), plot_circle=False):
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
    angles = np.arange(0, num_points + 1) * arc_angle / num_points + np.pi / 2

    # discard repeated final point if arc_angle is equal to 2*pi
    if full_circle:
        angles = angles[0:- 1]

    # create cartesian grid
    # circle = flipud([radius*cos(angles); radius*sin(-angles)]);        # B.0.3
    circle = np.vstack([radius * np.cos(angles), radius * np.sin(-angles)])  # B.0.4

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

    assert len(
        shift) == map_dimension, f'Optional input Shift must have {map_dimension} elements for {map_dimension} dimensional input parameters.'

    if map_dimension == 2:
        # create the maps for each dimension
        nx = createPixelDim(Nx, origin_size, shift[0])
        ny = createPixelDim(Ny, origin_size, shift[1])

        # create plaid grids
        r_x, r_y = np.meshgrid(nx, ny, indexing='ij')

        # extract the pixel radius
        r = np.sqrt(r_x ** 2 + r_y ** 2)
    if map_dimension == 3:
        # create the maps for each dimension
        nx = createPixelDim(Nx, origin_size, shift[0])
        ny = createPixelDim(Ny, origin_size, shift[1])
        nz = createPixelDim(Nz, origin_size, shift[2])

        # create plaid grids
        r_x, r_y, r_z = np.meshgrid(nx, ny, nz, indexing='ij')

        # extract the pixel radius
        r = np.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2)
    return r


def createPixelDim(Nx, origin_size, shift):
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
