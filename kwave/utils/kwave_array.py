import logging
import time
from dataclasses import dataclass
from math import ceil
from typing import Optional

import numpy as np
from numpy import arcsin, pi, cos, size, array
from numpy.linalg import linalg

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import tol_star
from kwave.utils.interp import get_delta_bli
from kwave.utils.mapgen import trim_cart_points, make_cart_rect, make_cart_arc, make_cart_bowl, make_cart_disc, make_cart_spherical_segment
from kwave.utils.math import sinc, get_affine_matrix
from kwave.utils.matlab import matlab_assign, matlab_mask, matlab_find


@dataclass
class Element:
    group_id: int
    type: str
    dim: int
    active: bool
    measure: float

    label: Optional[str] = None
    group_type: Optional[str] = None
    element_number: Optional[int] = None

    inner_diameter: Optional[float] = None
    outer_diameter: Optional[float] = None
    diameter: Optional[float] = None

    radius_of_curvature: Optional[float] = None
    position: Optional[np.ndarray] = None
    focus_position: Optional[np.ndarray] = None

    # custom element
    integration_points: Optional[np.ndarray] = None

    length: Optional[float] = None
    width: Optional[float] = None
    orientation: Optional[np.ndarray] = None

    start_point: Optional[np.ndarray] = None
    end_point: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Useful for consistency in tests where expected Matlab values can come in various types
        Returns:
            None
        """
        if self.position is not None:
            self.position = np.array(self.position, dtype=float)
        if self.focus_position is not None:
            self.focus_position = np.array(self.focus_position, dtype=float)
        self.active = bool(self.active)

        if self.inner_diameter:
            self.inner_diameter = float(self.inner_diameter)

        if self.outer_diameter:
            self.outer_diameter = float(self.outer_diameter)

        if self.diameter:
            self.diameter = float(self.diameter)

        if self.orientation is not None:
            self.orientation = np.array(self.orientation, dtype=float)

        if self.start_point is not None:
            if not (isinstance(self.start_point, list) or isinstance(self.start_point, np.ndarray)):
                self.start_point = [self.start_point]
            self.start_point = np.array(self.start_point, dtype=float)

        if self.end_point is not None:
            if not (isinstance(self.end_point, list) or isinstance(self.end_point, np.ndarray)):
                self.end_point = [self.end_point]
            self.end_point = np.array(self.end_point, dtype=float)

        self.measure = float(self.measure)


class kWaveArray(object):
    def __init__(
        self,
        axisymmetric: bool = False,
        bli_tolerance: float = 0.05,
        bli_type: str = "sinc",
        single_precision: bool = False,
        upsampling_rate: int = 10,
    ):
        assert 0 <= bli_tolerance <= 1
        assert bli_type in ["exact", "sinc"]

        self.axisymmetric = axisymmetric
        self.bli_tolerance = bli_tolerance
        self.bli_type = bli_type
        self.single_precision = single_precision
        self.upsampling_rate = upsampling_rate

        self.elements = []
        self.number_elements = 0
        self.number_groups = 0
        self.dim = 0
        self.array_transformation = []

        self.use_spiral_disc_points = 0
        self.num_arc_plot_points = 100
        self.element_plot_colour = np.array([0, 158, 194], dtype=float) / 255

    def add_annular_array(self, position, radius, diameters, focus_pos):
        assert isinstance(position, (list, tuple)), "'position' must be list or tuple"
        assert isinstance(radius, (int, float)), "'radius' must be an integer or float"
        assert isinstance(diameters, (list, tuple)), "'diameters' must be list or tuple"
        assert isinstance(focus_pos, (list, tuple)), "'focus_pos' must be list or tuple"
        assert len(position) == 3, "'position' must have 3 elements"
        assert len(focus_pos) == 3, "'focus_pos' must have 3 elements"
        assert all(len(i) == 2 for i in diameters), "'diameters' must be a list of lists, each with 2 elements"

        if self.number_elements == 0:
            self.dim = 3

        if self.dim != 3:
            raise ValueError(f"3D annular array cannot be added to an array with {self.dim}D elements.")

        annular_array_num_el = size(diameters, 0)

        self.number_groups += 1

        for el_ind in range(annular_array_num_el):
            self.number_elements += 1

            varphi_min = arcsin(diameters[el_ind][0] / (2 * radius))
            varphi_max = arcsin(diameters[el_ind][1] / (2 * radius))

            area = 2 * pi * radius**2 * (1 - cos(varphi_max)) - 2 * pi * radius**2 * (1 - cos(varphi_min))

            element = Element(
                group_id=self.number_groups,
                group_type="annular_array",
                element_number=el_ind + 1,
                type="annulus",
                dim=2,
                position=array(position),
                radius_of_curvature=radius,
                inner_diameter=diameters[el_ind][0],
                outer_diameter=diameters[el_ind][1],
                focus_position=array(focus_pos),
                active=True,
                measure=area,
            )
            self.elements.append(element)

    def add_annular_element(self, position, radius, diameters, focus_pos):
        assert isinstance(position, (list, tuple)), "'position' must be list or tuple"
        assert isinstance(radius, (int, float)), "'radius' must be an integer or float"
        assert isinstance(diameters, (list, tuple)), "'diameters' must be list or tuple"
        assert isinstance(focus_pos, (list, tuple)), "'focus_pos' must be list or tuple"
        assert len(position) == 3, "'position' must have 3 elements"
        assert len(focus_pos) == 3, "'focus_pos' must have 3 elements"
        assert len(diameters) == 2, "'diameters' must have 2 elements"

        if self.number_elements == 0:
            self.dim = 3

        if self.dim != 3:
            raise ValueError(f"3D annular array cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        varphi_min = arcsin(diameters[0] / (2 * radius))
        varphi_max = arcsin(diameters[1] / (2 * radius))

        area = 2 * pi * radius**2 * (1 - cos(varphi_max)) - 2 * pi * radius**2 * (1 - cos(varphi_min))

        self.elements.append(
            Element(
                group_id=0,
                type="annulus",
                dim=2,
                position=array(position),
                radius_of_curvature=radius,
                inner_diameter=diameters[0],
                outer_diameter=diameters[1],
                focus_position=array(focus_pos),
                active=True,
                measure=area,
            )
        )

    def add_bowl_element(self, position, radius, diameter, focus_pos):
        assert isinstance(position, (list, tuple)), "'position' must be list or tuple"
        assert isinstance(radius, (int, float)), "'radius' must be an integer or float"
        assert isinstance(diameter, (int, float)), "'diameter' must be an integer or float"
        assert isinstance(focus_pos, (list, tuple)), "'focus_pos' must be list or tuple"
        assert len(position) == 3, "'position' must have 3 elements"
        assert len(focus_pos) == 3, "'focus_pos' must have 3 elements"

        if self.number_elements == 0:
            self.dim = 3

        if self.dim != 3:
            raise ValueError(f"3D bowl element cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        varphi_max = arcsin(diameter / (2 * radius))

        area = 2 * pi * radius**2 * (1 - cos(varphi_max))

        self.elements.append(
            Element(
                group_id=0,
                type="bowl",
                dim=2,
                position=array(position),
                radius_of_curvature=radius,
                diameter=diameter,
                focus_position=array(focus_pos),
                active=True,
                measure=area,
            )
        )

    def add_custom_element(self, integration_points, measure, element_dim, label):
        assert isinstance(integration_points, (np.ndarray)), "'integration_points' must be a numpy array"
        assert isinstance(measure, (int, float)), "'measure' must be an integer or float"
        assert isinstance(element_dim, (int)) and element_dim in [1, 2, 3], "'element_dim' must be an integer and either 1, 2 or 3"
        assert isinstance(label, (str)), "'label' must be a string"

        # check the dimensionality of the integration points
        input_dim = integration_points.shape[0]
        if (input_dim < 1) or (input_dim > 3):
            raise ValueError("Input integration_points must be a 1 x N (in 1D), 2 x N (in 2D), or 3 x N (in 3D) array.")

        # check if this is the first element, and set the dimension
        if self.number_elements == 0:
            self.dim = input_dim

        # check that the element is being added to an array with the
        # correct dimensions
        if self.dim != input_dim:
            raise ValueError(f"{input_dim}D custom element cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        self.elements.append(
            Element(
                group_id=0, type="custom", dim=element_dim, label=label, integration_points=integration_points, active=True, measure=measure
            )
        )

    def add_rect_element(self, position, Lx, Ly, theta):
        assert isinstance(position, (list, tuple)), "'position' must be a list or tuple"
        assert isinstance(Lx, (int, float)), "'Lx' must be an integer or float"
        assert isinstance(Ly, (int, float)), "'Ly' must be an integer or float"

        coord_dim = len(position)

        if coord_dim not in [2, 3]:
            raise ValueError("Input position for rectangular element must be specified as a 2 (2D) or 3 (3D) element array.")

        if coord_dim == 3:
            assert isinstance(theta, (Vector, list, tuple)) and len(theta) == 3, "'theta' must be a list or tuple of length 3"
        else:
            assert isinstance(theta, (int, float)), "'theta' must be an integer or float"

        if self.number_elements == 0:
            self.dim = coord_dim

        if self.dim != coord_dim:
            raise ValueError(f"{coord_dim}D rectangular element cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        area = Lx * Ly

        self.elements.append(
            Element(
                group_id=0,
                type="rect",
                dim=2,
                position=array(position),
                length=Lx,
                width=Ly,
                orientation=array(theta) if coord_dim == 3 else theta,
                active=True,
                measure=area,
            )
        )

    def add_arc_element(self, position, radius, diameter, focus_pos):
        assert isinstance(position, (list, tuple, Vector)), "'position' must be list, tuple or Vector"
        assert isinstance(radius, (int, float)), "'radius' must be an integer or float"
        assert isinstance(diameter, (int, float)), "'diameter' must be an integer or float"
        assert isinstance(focus_pos, (list, tuple, Vector)), "'focus_pos' must be list, tuple or Vector"
        assert len(position) == 2, "'position' must have 2 elements"
        assert len(focus_pos) == 2, "'focus_pos' must have 2 elements"

        if self.number_elements == 0:
            self.dim = 2

        if self.dim != 2:
            raise ValueError(f"2D arc element cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        varphi_max = arcsin(diameter / (2 * radius))

        length = 2 * radius * varphi_max

        self.elements.append(
            Element(
                group_id=0,
                type="arc",
                dim=1,
                position=array(position),
                radius_of_curvature=radius,
                diameter=diameter,
                focus_position=array(focus_pos),
                active=True,
                measure=length,
            )
        )

    def add_disc_element(self, position, diameter, focus_pos=None):
        assert isinstance(position, (list, tuple)), "'position' must be a list or tuple"
        assert isinstance(diameter, (int, float)), "'diameter' must be an integer or float"

        coord_dim = len(position)

        if coord_dim not in [2, 3]:
            raise ValueError("Input position for disc element must be specified as a 2 (2D) or 3 (3D) element array.")

        if coord_dim == 3:
            assert isinstance(focus_pos, (list, tuple)) and len(focus_pos) == 3, "'focus_pos' must be a list or tuple of length 3"
        else:
            focus_pos = []

        if self.number_elements == 0:
            self.dim = coord_dim

        if self.dim != coord_dim:
            raise ValueError(f"{coord_dim}D disc element cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        area = pi * (diameter / 2) ** 2

        self.elements.append(
            Element(
                group_id=0,
                type="disc",
                dim=2,
                position=array(position),
                diameter=diameter,
                focus_position=array(focus_pos),
                active=True,
                measure=area,
            )
        )

    def remove_element(self, element_num):
        if element_num > self.number_elements:
            raise ValueError(f"Cannot remove element {element_num} from array with {self.number_elements} elements.")

        self.elements.pop(element_num)

        self.number_elements -= 1

    def add_line_element(self, start_point, end_point):
        assert isinstance(start_point, (list, tuple)), "'start_point' must be a list or tuple"
        assert isinstance(end_point, (list, tuple)), "'end_point' must be a list or tuple"

        input_dim = len(start_point)

        if input_dim not in [1, 2, 3]:
            raise ValueError("Input start_point must have 1 (in 1D), 2 (in 2D), or 3 (in 3D) elements.")

        if input_dim != len(end_point):
            raise ValueError("Inputs start_point and end_point must have the same number of elements.")

        if self.number_elements == 0:
            self.dim = input_dim

        if self.dim != input_dim:
            raise ValueError(f"{input_dim}D line element cannot be added to an array with {self.dim}D elements.")

        self.number_elements += 1

        line_length = linalg.norm(array(end_point) - array(start_point))

        self.elements.append(
            Element(
                group_id=0, type="line", dim=1, start_point=array(start_point), end_point=array(end_point), active=True, measure=line_length
            )
        )

    def get_element_grid_weights(self, kgrid, element_num):
        return self.get_off_grid_points(kgrid, element_num, False)

    def get_element_binary_mask(self, kgrid, element_num):
        return self.get_off_grid_points(kgrid, element_num, True)

    def get_array_grid_weights(self, kgrid):
        self.check_for_elements()

        if kgrid.Nz >= 1:
            grid_weights = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz))
        else:
            grid_weights = np.zeros((kgrid.Nx, kgrid.Ny))

        assert len(grid_weights.shape) == self.dim, "Grid weights shape must match kArray dimensions."

        for ind in range(self.number_elements):
            grid_weights += self.get_off_grid_points(kgrid, ind, False)

        assert len(grid_weights.shape) == self.dim, "Grid weights shape must match kArray dimensions."

        return grid_weights

    def get_array_binary_mask(self, kgrid):
        self.check_for_elements()

        mask = np.squeeze(np.zeros((kgrid.Nx, kgrid.Ny, max(kgrid.Nz, 1)), dtype=bool))

        for ind in range(self.number_elements):
            grid_weights = self.get_off_grid_points(kgrid, ind, True)
            mask = np.bitwise_or(np.squeeze(mask), grid_weights)

        return mask

    def check_for_elements(self):
        if self.number_elements == 0:
            raise ValueError("Cannot call method on an array with zero elements.")

    def affine(self, vec):
        if len(self.array_transformation) == 0:
            return vec

        # check vector is the correct length
        if self.dim != len(vec):
            raise ValueError("Input vector length must match the number of dimensions")

        # apply transformation
        vec = np.append(vec, [1])
        vec = np.matmul(self.array_transformation, vec)
        return vec[:-1]

    def get_off_grid_points(self, kgrid, element_num, mask_only):
        # check the array has elements
        self.check_for_elements()

        # check inputs
        assert isinstance(kgrid, kWaveGrid)
        assert 0 <= element_num <= self.number_elements - 1

        # compute measure (length/area/volume) in grid squares (assuming dx = dy = dz)
        m_grid = self.elements[element_num].measure / (kgrid.dx) ** (self.elements[element_num].dim)

        # get number of integration points
        if self.elements[element_num].type == "custom":
            # assign number of integration points directly
            m_integration = self.elements[element_num].integration_points.shape[1]
        else:
            # compute the number of integration points using the upsampling rate
            m_integration = ceil(m_grid * self.upsampling_rate)

        # compute integration points covering element
        if self.elements[element_num].type == "annulus":
            # compute points using make_cart_spherical_segment
            integration_points = make_cart_spherical_segment(
                self.affine(self.elements[element_num].position),
                self.elements[element_num].radius_of_curvature,
                self.elements[element_num].inner_diameter,
                self.elements[element_num].outer_diameter,
                self.affine(self.elements[element_num].focus_position),
                m_integration,
            )

        elif self.elements[element_num].type == "arc":
            # compute points using make_cart_arc
            integration_points = make_cart_arc(
                self.affine(self.elements[element_num].position),
                self.elements[element_num].radius_of_curvature,
                self.elements[element_num].diameter,
                self.affine(self.elements[element_num].focus_position),
                m_integration,
            )

        elif self.elements[element_num].type == "bowl":
            # compute points using make_cart_bowl
            integration_points = make_cart_bowl(
                self.affine(self.elements[element_num].position),
                self.elements[element_num].radius_of_curvature,
                self.elements[element_num].diameter,
                self.affine(self.elements[element_num].focus_position),
                m_integration,
            )

        elif self.elements[element_num].type == "custom":
            # directly assign integration points
            integration_points = self.elements[element_num].integration_points

        elif self.elements[element_num].type == "disc":
            # compute points using make_cart_disc
            integration_points = make_cart_disc(
                self.affine(self.elements[element_num].position),
                self.elements[element_num].diameter / 2,
                self.affine(self.elements[element_num].focus_position),
                m_integration,
                False,
                self.use_spiral_disc_points,
            )

        elif self.elements[element_num].type == "rect":
            # compute points using make_cart_rect
            integration_points = make_cart_rect(
                self.affine(self.elements[element_num].position),
                self.elements[element_num].length,
                self.elements[element_num].width,
                self.elements[element_num].orientation,
                m_integration,
            )

        elif self.elements[element_num].type == "line":
            # get distance between points in each dimension
            d = (self.elements[element_num].end_point - self.elements[element_num].start_point) / m_integration

            # compute a set of uniformly spaced Cartesian points
            # covering the line using linspace, where the end
            # points are offset by half the point spacing
            if self.dim == 1:
                integration_points = np.linspace(
                    self.elements[element_num].start_point + d[0] / 2, self.elements[element_num].end_point - d[0] / 2, m_integration
                )
            elif self.dim == 2:
                px = np.linspace(
                    self.elements[element_num].start_point[0] + d[0] / 2, self.elements[element_num].end_point[0] - d[0] / 2, m_integration
                )
                py = np.linspace(
                    self.elements[element_num].start_point[1] + d[1] / 2, self.elements[element_num].end_point[1] - d[1] / 2, m_integration
                )
                integration_points = np.array([px, py])
            elif self.dim == 3:
                px = np.linspace(
                    self.elements[element_num].start_point[0] + d[0] / 2, self.elements[element_num].end_point[0] - d[0] / 2, m_integration
                )
                py = np.linspace(
                    self.elements[element_num].start_point[1] + d[1] / 2, self.elements[element_num].end_point[1] - d[1] / 2, m_integration
                )
                pz = np.linspace(
                    self.elements[element_num].start_point[2] + d[2] / 2, self.elements[element_num].end_point[2] - d[2] / 2, m_integration
                )
                integration_points = np.array([px, py, pz])

        else:
            raise ValueError(f"{self.elements[element_num].type} is not a valid array element type.")

        # recompute actual number of points
        m_integration = integration_points.shape[1]

        # compute scaling factor
        scale = m_grid / m_integration

        if self.axisymmetric:
            # create new expanded grid
            kgrid_expanded = kWaveGrid(Vector([kgrid.Nx, 2 * kgrid.Ny]), Vector([kgrid.dx, kgrid.dy]))

            # remove integration points which are outside grid
            integration_points = trim_cart_points(kgrid_expanded, integration_points)

            # calculate grid weights from BLIs centered on the integration points
            grid_weights = off_grid_points(
                kgrid_expanded,
                integration_points,
                scale,
                bli_tolerance=self.bli_tolerance,
                bli_type=self.bli_type,
                mask_only=mask_only,
                single_precision=self.single_precision,
            )

            # keep points in the positive y domain
            grid_weights = grid_weights[:, kgrid.Ny :]

        else:
            # remove integration points which are outside grid
            integration_points = trim_cart_points(kgrid, integration_points)

            # calculate grid weights from BLIs centered on the integration points
            grid_weights = off_grid_points(
                kgrid,
                integration_points,
                scale,
                bli_tolerance=self.bli_tolerance,
                bli_type=self.bli_type,
                mask_only=mask_only,
                single_precision=self.single_precision,
            )

        return grid_weights

    def get_distributed_source_signal(self, kgrid, source_signal):
        start_time = time.time()

        self.check_for_elements()

        mask = self.get_array_binary_mask(kgrid)
        mask_ind = matlab_find(mask).squeeze(axis=-1)
        num_source_points = np.sum(mask)

        Nt = np.shape(source_signal)[1]

        if self.single_precision:
            data_type = "float32"
            sz_bytes = num_source_points * Nt * 4
        else:
            data_type = "float64"
            sz_bytes = num_source_points * Nt * 8

        sz_ind = 1
        while sz_bytes > 1024:
            sz_bytes = sz_bytes / 1024
            sz_ind += 1

        prefixes = ["", "K", "M", "G", "T"]
        sz_bytes = np.round(sz_bytes, 2)  # TODO: should round to significant to map matlab functionality
        logging.log(logging.INFO, f"approximate size of source matrix: {str(sz_bytes)} {prefixes[sz_ind]} B ( {data_type} precision)")

        source_signal = source_signal.astype(data_type)

        distributed_source_signal = np.zeros((num_source_points, Nt), dtype=data_type)

        for ind in range(self.number_elements):
            source_weights = self.get_element_grid_weights(kgrid, ind)

            element_mask_ind = matlab_find(np.array(source_weights), val=0, mode="neq").squeeze(axis=-1)

            local_ind = np.isin(mask_ind, element_mask_ind)

            distributed_source_signal[local_ind] += matlab_mask(source_weights, element_mask_ind - 1) * source_signal[ind, :][None, :]

        end_time = time.time()
        logging.log(logging.INFO, f"total computation time : {end_time - start_time:.2f} s")

        return distributed_source_signal

    def combine_sensor_data(self, kgrid, sensor_data):
        self.check_for_elements()

        mask = self.get_array_binary_mask(kgrid)
        mask_ind = matlab_find(mask).squeeze(axis=-1)

        Nt = np.shape(sensor_data)[1]
        # TODO (Walter): this assertion does not work when "auto" is set
        # assert kgrid.Nt == Nt, 'sensor_data must have the same number of time steps as kgrid'

        combined_sensor_data = np.zeros((self.number_elements, Nt))

        for element_num in range(self.number_elements):
            source_weights = self.get_element_grid_weights(kgrid, element_num)

            element_mask_ind = matlab_find(np.array(source_weights), val=0, mode="neq").squeeze(axis=-1)

            local_ind = np.isin(mask_ind, element_mask_ind)

            combined_sensor_data[element_num, :] = np.sum(
                sensor_data[local_ind] * matlab_mask(source_weights, element_mask_ind - 1), axis=0
            )

            m_grid = self.elements[element_num].measure / (kgrid.dx) ** (self.elements[element_num].dim)

            combined_sensor_data[element_num, :] = combined_sensor_data[element_num, :] / m_grid

        return combined_sensor_data

    def set_array_position(self, translation, rotation):
        # get affine matrix and store
        self.array_transformation = get_affine_matrix(translation, rotation)

    def set_affine_transform(self, affine_transform):
        # check array has elements
        if self.number_elements == 0:
            raise ValueError("Array must have at least one element before a transformation can be defined.")

        # check the input is the correct size
        sx, sy = affine_transform.shape
        if (self.dim == 2 and (sx != 3 or sy != 3)) or (self.dim == 3 and (sx != 4 or sy != 4)):
            raise ValueError("Affine transform must be a 3x3 matrix for arrays in 2D, and 4x4 for arrays in 3D.")

            # assign the transform
        self.array_transformation = affine_transform

    def get_element_positions(self):
        # initialise output
        element_pos = np.zeros((self.dim, self.number_elements))

        # loop through the elements and assign transformed position
        for i, element in enumerate(self.elements):
            element_pos[:, i] = self.affine(element.position)

        return element_pos


def off_grid_points(
    kgrid, points, scale=1, bli_tolerance=0.1, bli_type="sinc", mask_only=False, single_precision=False, debug=False, display_wait_bar=False
):
    wait_bar_update_freq = 100

    # check dimensions of points input
    if points.shape[0] != kgrid.dim:
        raise ValueError("Input points must be given as matrix with dimensions num_dims x num_points.")

    # get the number of off-grid points
    num_points = points.shape[1]

    # expand scale value if scalar
    if np.isscalar(scale):
        scale = scale * np.ones(num_points)
    elif np.size(scale) != num_points:
        raise ValueError("Input scale must be scalar or the same length as points.")
    if not isinstance(debug, bool):
        raise ValueError("Optional input 'Debug' must be Boolean.")

    if not (isinstance(bli_tolerance, (int, float)) and 0 < bli_tolerance < 1):
        raise ValueError("bli_tolerance should be a real scalar value between 0 and 1.")
    if bli_type not in ["sinc", "exact"]:
        raise ValueError("Optional input 'bli_type' must be either 'sinc' or 'exact'.")
    if not isinstance(mask_only, bool):
        raise ValueError("Optional input 'mask_only' must be Boolean.")
    if not isinstance(single_precision, bool):
        raise ValueError("Optional input 'single_precision' must be Boolean.")
    if not isinstance(display_wait_bar, bool):
        raise ValueError("Optional input 'display_wait_bar' must be Boolean.")

    # reassign bli_tolerance if bli_type is 'exact'
    if bli_type == "exact":
        bli_tolerance = 0

    # extract grid vectors (to avoid recomputing them below)
    x_vec = kgrid.x_vec
    y_vec = kgrid.y_vec
    z_vec = kgrid.z_vec

    # preallocate some variables for speed
    if bli_tolerance == 0:
        pi_on_dx = np.pi / kgrid.dx
        pi_on_dy = np.pi / kgrid.dy
        pi_on_dz = np.pi / kgrid.dz
    else:
        scalar_dxyz = True
        pi_on_dxyz = np.pi / kgrid.dx
        if kgrid.dim == 2:
            if kgrid.dx != kgrid.dy:
                scalar_dxyz = False
                pi_on_dxyz = np.pi / np.array([kgrid.dx, kgrid.dy])
        elif kgrid.dim == 3:
            if not (kgrid.dx == kgrid.dy and kgrid.dx == kgrid.dz):
                scalar_dxyz = False
                pi_on_dxyz = np.pi / np.array([kgrid.dx, kgrid.dy, kgrid.dz])

    # initialise the source mask
    if mask_only:
        mask_type = bool
    elif single_precision:
        mask_type = np.float32
    else:
        mask_type = np.float64

    if kgrid.dim == 1:
        mask = np.zeros((kgrid.Nx, 1), dtype=mask_type)
    elif kgrid.dim == 2:
        mask = np.zeros((kgrid.Nx, kgrid.Ny), dtype=mask_type)
    elif kgrid.dim == 3:
        mask = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz), dtype=mask_type)

    # display wait bar
    if display_wait_bar:
        import tqdm

        tqdm.tqdm(total=100, desc="Computing off-grid source mask...")

    # add to the overall mask using contributions from each source point
    for point_ind in range(num_points):
        # extract a single point
        point = points[:, point_ind]

        # convert to the computational coordinate if the physical coordinate is
        # sampled nonuniformly
        if kgrid.nonuniform:
            point, BLIscale = mapPoint(kgrid, point)  # noqa: F821

        if bli_tolerance == 0:
            if mask_only:
                mask = np.ones(mask.shape, dtype=bool)
                return mask
            else:
                if kgrid.dim == 1:
                    if bli_type == "sinc":
                        mask = mask + scale[point_ind] * sinc(pi_on_dx * (x_vec - point[0]))
                    elif bli_type == "exact":
                        mask = mask + scale[point_ind] * get_delta_bli(kgrid.Nx, kgrid.dx, x_vec, point[0])

                elif kgrid.dim == 2:
                    if bli_type == "sinc":
                        mask_t_x = sinc(pi_on_dx * (x_vec - point[0]))
                        mask_t_y = sinc(pi_on_dy * (y_vec - point[1]))
                    elif bli_type == "exact":
                        mask_t_x = get_delta_bli(kgrid.Nx, kgrid.dx, x_vec, point[0])
                        mask_t_y = get_delta_bli(kgrid.Ny, kgrid.dy, y_vec, point[1])

                    mask = mask + scale[point_ind] * (mask_t_x @ mask_t_y.T)

                elif kgrid.dim == 3:
                    if bli_type == "sinc":
                        mask_t_x = sinc(pi_on_dx * (x_vec - point[0]))
                        mask_t_y = sinc(pi_on_dy * (y_vec - point[1]))
                        mask_t_z = sinc(pi_on_dz * (z_vec - point[2]))
                    elif bli_type == "exact":
                        mask_t_x = get_delta_bli(kgrid.Nx, kgrid.dx, x_vec, point[0])
                        mask_t_y = get_delta_bli(kgrid.Ny, kgrid.dy, y_vec, point[1])
                        mask_t_z = get_delta_bli(kgrid.Nz, kgrid.dz, z_vec, point[2])

                    mask = mask + scale[point_ind] * np.reshape(np.kron(mask_t_y @ mask_t_z.T, mask_t_x), [kgrid.Nx, kgrid.Ny, kgrid.Nz])

        else:
            # create an array of neighbouring grid points for BLI evaluation
            if kgrid.dim == 1:
                ind, is_, _, _ = tol_star(bli_tolerance, kgrid, point, debug)
                xs = x_vec[is_]
                xyz = xs
            elif kgrid.dim == 2:
                ind, is_, js, _ = tol_star(bli_tolerance, kgrid, point, debug)
                xs = x_vec[is_]
                ys = y_vec[js]
                xyz = np.array([xs, ys]).squeeze().T
            elif kgrid.dim == 3:
                ind, is_, js, ks = tol_star(bli_tolerance, kgrid, point, debug)
                xs = x_vec[is_.astype(int)].squeeze(axis=-1)
                ys = y_vec[js.astype(int)].squeeze(axis=-1)
                zs = z_vec[ks.astype(int)].squeeze(axis=-1)
                xyz = np.array([xs, ys, zs]).T

            ind = ind.astype(int)

            if mask_only:
                # add current points to the mask
                mask = matlab_assign(mask, ind - 1, True)
            else:
                # evaluate a BLI centered on point at grid nodes XYZ
                if scalar_dxyz:
                    if single_precision:
                        mask_t = sinc(pi_on_dxyz * (xyz - point.T))
                    else:
                        mask_t = sinc(pi_on_dxyz * (xyz - point.T))
                else:
                    if single_precision:
                        mask_t = sinc(pi_on_dxyz * (xyz - point.T))
                    else:
                        mask_t = sinc(pi_on_dxyz * (xyz - point.T))
                current_mask_t = np.prod(np.atleast_2d(mask_t), axis=1)

                # apply scaling for non-uniform grid
                if kgrid.nonuniform:
                    current_mask_t = mask_t * BLIscale

                updated_mask_value = matlab_mask(mask, ind - 1).squeeze(axis=-1) + scale[point_ind] * current_mask_t
                # add this contribution to the overall source mask
                mask = matlab_assign(mask, ind - 1, updated_mask_value)

        # update the waitbar
        if display_wait_bar and (point_ind % wait_bar_update_freq == 0):
            tqdm.update(wait_bar_update_freq)
    return mask
