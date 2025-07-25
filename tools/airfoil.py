import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from typing import cast, Union, overload
from collections.abc import Sequence, Callable
from functools import singledispatchmethod

AirfoilFunction = Union[
    Callable[[float], float],
    Callable[[float], tuple[float, float]],
    Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
]

class Airfoil:
    """
    A class to represent a airfoil shape defined by a set of points in 2D space.

    Points Array (`points`):
    - Represents the airfoil contour as a closed shape.
    - Order: Counter-clockwise (CCW) traversal.
    - Leading edge: points[0] (minimum x-coordinate).
    - Trailing edge: Stored at index `tail_index` (farthest point from leading edge).
    - Closure: points[0] and points[-1] do NOT coincide (implicit closure between last and first point).

    Key Properties:
    1. `panel`: Array of panel vectors
    - panel[i] = points[(i+1) % n] - points[i] (vector from point i to next)
    - Length: n panels (same as number of points)

    2. `norm`: Outward unit normal vectors for each panel
    - Derived from panel vectors: (dy, -dx) normalized

    3. `tangent`: Unit tangent vectors for panels
    - Direction: points[i] → points[(i+1) % n]

    4. `length`: Length (magnitude) of each panel

    5. `midpoint`: Midpoint coordinates of each panel

    6. `tangent_length`: Cumulative arc length from leading edge
    - tangent_length[i] = sum(length[0] to length[i])
    - Total contour length = tangent_length[-1]

    7. `chord`: Chord length (distance between leading edge and trailing edge)

    8. `tail_index`: Index of trailing edge point

    9. `center`: Center point coordinates of the airfoil
    - Calculated as weighted average of panel midpoints using panel lengths as weights
    - Returns: (x, y) coordinates as tuple[float, float]

    Critical Methods:
    1. position_along_edge(x, upper):
    - Maps distance `x` from leading edge to point on contour
    - upper=True: Upper surface (distance measured from LE along upper side)
    - upper=False: Lower surface (distance measured from LE along lower side)
    - Returns: (x, y) coordinates and panel index containing point
    - Interpolation: Point lies on panel between points[i] and points[(i+1) % n]

    2. expand(distance):
    - Offsets airfoil along outward normals
    - distance types:
        • float: Uniform offset for all panels
        • (upper_func, lower_func): 
        - upper_func(s): Displacement for upper surface at arc length s from LE
        - lower_func(s): Displacement for lower surface at arc length s from LE

    3. Rotation:
    - inplace_rotate(angle): Rotates CCW by angle (radians) in-place
    - rotate(angle): Returns new rotated Airfoil instance

    Surface Order & Indexing:
    - Lower surface: Points [0 → tail_index] (LE to TE)
    - Upper surface: Points [tail_index → -1] (TE to LE)
    - Panel i connects points[i] → points[i+1] (last panel: points[-1] → points[0])

    Constructor Notes:
    Input points are processed to:
    1. Ensure CCW order (reverse if clockwise)
    2. Set points[0] as leading edge (min x)
    3. Resample edges:
    - Split edges > max_edge_length
    - Remove points < min_edge_length apart
    4. Rotate to align chord horizontally (unless auto_horizontal=False)

    Key Invariants:
    - points[0] ≠ points[-1]
    - panel[i] = points[(i+1) % n] - points[i]
    - norm[i] perpendicular to panel[i], pointing OUTWARD
    - tail_index = index of max(points[i].distance(points[0]))
    """
    _norm_buffer: npt.NDArray[np.float64] | None = None
    _panel_buffer: npt.NDArray[np.float64] | None = None
    _tangent_buffer: npt.NDArray[np.float64] | None = None
    _length_buffer: npt.NDArray[np.float64] | None = None
    _mid_point_buffer: npt.NDArray[np.float64] | None = None
    _tangent_length_buffer: npt.NDArray[np.float64] | None = None
    _center_buffer: tuple[float, float] | None = None
    _tail_index: int
    _chord_length: float
    points: npt.NDArray[np.float64]

    @classmethod
    def from_function(
        cls,
        func: Sequence[AirfoilFunction],
        x_range: tuple[float, float] = (0.0, 1.0),
        n_points: int = 100,
        max_edge_length: float = 1.0,
        min_edge_length: float = 0.01,
        reorder: bool = True,
        auto_horizontal: bool = True
    ) -> "Airfoil":
        """
        Generates an airfoil shape from a sequence of functions with equal arc length distribution.
        Functions can be in the following formats:
        - A function that takes a float (as x-coordinate) and returns a float (y-coordinate).
        - A function that takes a float (as parameter t) and returns a tuple (x, y).
        - A function that takes a numpy array of x-coordinates and returns a numpy array of y-coordinates.
        - A function that takes a numpy array of parameters t and returns a 2d numpy array of points.

        @param func: A sequence of functions defining the airfoil shape.
        @param x_range: The range of x values to sample the functions over.
        @param n_points: The number of points to sample per function.
        @param max_edge_length: The maximum length of an edge between two points.
        @param min_edge_length: The minimum length of an edge between two points.
        @param reorder: Whether to reorder the points to optimize the shape.
        @param auto_horizontal: Whether to automatically rotate the airfoil to make the chord horizontal.
        """
        def _calculate_arc_length_integrand(t: float, func_callable: Callable[[float], tuple[float, float]]) -> float:
            """Calculate the derivative magnitude for arc length integration."""
            dt = 1e-8
            p1 = func_callable(t)
            p2 = func_callable(t + dt)
            dx_dt = (p2[0] - p1[0]) / dt
            dy_dt = (p2[1] - p1[1]) / dt
            return np.sqrt(dx_dt**2 + dy_dt**2)
        
        def _sample_equal_arc_length(func_callable: Callable[[float], tuple[float, float]], 
                                   param_range: tuple[float, float], 
                                   n_points: int) -> npt.NDArray[np.float64]:
            """Sample points with equal arc length distribution."""
            total_length, _ = quad(_calculate_arc_length_integrand, param_range[0], param_range[1], 
                                 args=(func_callable,), limit=1000)
            target_segment_length = total_length / (n_points - 1)
            
            points = []
            current_param = param_range[0]
            points.append(func_callable(current_param))
            
            for _ in range(1, n_points - 1):
                def arc_length_from_current(t):
                    if t <= current_param:
                        return 0.0
                    length, _ = quad(_calculate_arc_length_integrand, current_param, t, 
                                   args=(func_callable,), limit=500)
                    return length - target_segment_length
                t_low = current_param
                t_high = param_range[1]
                t_mid = current_param
                for _ in range(50):
                    t_mid = (t_low + t_high) / 2
                    arc_diff = arc_length_from_current(t_mid)
                    if abs(arc_diff) < 1e-10:
                        break
                    elif arc_diff < 0:
                        t_low = t_mid
                    else:
                        t_high = t_mid
                current_param = t_mid
                points.append(func_callable(current_param))
            points.append(func_callable(param_range[1]))
            return np.array(points, dtype=np.float64)

        mid = (x_range[0] + x_range[1]) / 2
        points: npt.NDArray[np.float64] = np.zeros((0, 2), dtype=np.float64)
        
        for f in func:
            try: # float -> float | float -> tuple[float, float]
                f = cast(Callable[[float], Union[float, tuple[float, float]]], f)
                res = f(mid)
                if not isinstance(res, (float, tuple)):
                    raise TypeError("Checkout to ndarray")
                if isinstance(res, float):
                    def inner_func(x: float, f=f) -> tuple[float, float]:
                        return x, f(x)
                    f = inner_func
                f = cast(Callable[[float], tuple[float, float]], f)
                new_points = _sample_equal_arc_length(f, x_range, n_points)
                points = np.concatenate((points, new_points), axis=0)
                
            except Exception:
                f = cast(Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], f)
                res = f(np.array([mid], dtype=np.float64))
                if not isinstance(res, np.ndarray):
                    raise TypeError("Unable to process function output, expected float, tuple or ndarray.")
                x_dense = np.linspace(x_range[0], x_range[1], n_points * 5, dtype=np.float64)
                y_dense = f(x_dense)
                if y_dense.ndim == 1:
                    dense_points = np.stack((x_dense, y_dense), axis=-1)
                else:
                    dense_points = y_dense
                diffs = np.diff(dense_points, axis=0)
                segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
                total_length = cumulative_lengths[-1]
                target_lengths = np.linspace(0, total_length, n_points)
                sampled_points = []
                
                for target_length in target_lengths:
                    idx = np.searchsorted(cumulative_lengths, target_length)
                    if idx == 0:
                        sampled_points.append(dense_points[0])
                    elif idx >= len(dense_points):
                        sampled_points.append(dense_points[-1])
                    else:
                        # Interpolate
                        t = (target_length - cumulative_lengths[idx-1]) / (cumulative_lengths[idx] - cumulative_lengths[idx-1])
                        interpolated_point = dense_points[idx-1] + t * (dense_points[idx] - dense_points[idx-1])
                        sampled_points.append(interpolated_point)
                new_points = np.array(sampled_points, dtype=np.float64)
                points = np.concatenate((points, new_points), axis=0)
                
        points = cast(npt.NDArray[np.float64], points)
        return cls(points, max_edge_length=max_edge_length, min_edge_length=min_edge_length, reorder=reorder, auto_horizontal=auto_horizontal)

    def __init__(
        self,
        points: npt.NDArray[np.float64],
        max_edge_length: float = 1.0,
        min_edge_length: float = 0.01,
        reorder: bool = False,
        auto_horizontal: bool = True
    ):
        """
        Initializes the airfoil with a set of closed points.

        @param points: A 2D numpy array of shape (n, 2) where n is the number of points, first point must be the leading edge.
        @param max_edge_length: The maximum length of an edge between two points. If an edge exceeds this length, it will be subdivided.
        @param reorder: Whether to reorder the points to optimize the shape.
        """
        self.points = np.asarray(points)
        if reorder:
            self.points = self._sort_curve_points_optimized(self.points)
        if np.array_equal(self.points[0], self.points[-1]):
            self.points = self.points[:-1]
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Points must be a 2D array with shape (n, 2).")
        if len(self.points) < 3:
            raise ValueError("At least 3 points are required to form a shape.")
        self.max_edge_length = max_edge_length
        self.min_edge_length = min_edge_length
        self._split_points()
        self._check_counter_clockwise()
        self._ensure_leading_edge_first()  # Ensure leading edge is first after all processing
        self._check_chord_horizontal(auto_horizontal)

    def _split_points(self):
        """
        Splits the points into segments based on the maximum edge length.
        Also removes points that are too close to each other based on minimum edge length.
        """
        new_points = []
        original_points = self.points.copy()
        for i in range(len(original_points)):
            current_point = original_points[i]
            next_point = original_points[(i + 1) % len(original_points)]
            new_points.append(current_point)
            edge_vector = next_point - current_point
            edge_length = np.linalg.norm(edge_vector)
            if edge_length > self.max_edge_length:
                n_subdivisions = int(np.ceil(edge_length / self.max_edge_length))
                for j in range(1, n_subdivisions):
                    ratio = j / n_subdivisions
                    intermediate_point = current_point + ratio * edge_vector
                    new_points.append(intermediate_point)
        new_points_array = np.array(new_points)
        tolerance = 1e-10
        unique_points = [new_points_array[0]]
        for i in range(1, len(new_points_array)):
            if np.linalg.norm(new_points_array[i] - unique_points[-1]) > tolerance:
                unique_points.append(new_points_array[i])
        if len(unique_points) > 1 and np.linalg.norm(unique_points[-1] - unique_points[0]) < tolerance:
            unique_points.pop()
        filtered_points = [unique_points[0]]
        for i in range(1, len(unique_points)):
            current_point = unique_points[i]
            last_kept_point = filtered_points[-1]
            distance = np.linalg.norm(current_point - last_kept_point)
            if distance >= self.min_edge_length:
                filtered_points.append(current_point)
        if len(filtered_points) > 2:
            distance_to_first = np.linalg.norm(filtered_points[-1] - filtered_points[0])
            if distance_to_first < self.min_edge_length:
                filtered_points.pop()
        self.points = np.array(filtered_points)

    def _check_counter_clockwise(self):
        """
        Checks if the points are in counter-clockwise order and reverses them if not.
        """
        signed_area = 0.0
        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            signed_area += (self.points[j][0] - self.points[i][0]) * (self.points[j][1] + self.points[i][1])
        if signed_area > 0:
            self.points = self.points[::-1]

    def _ensure_leading_edge_first(self):
        """
        Ensures that the first point is the leading edge (minimum x-coordinate).
        If not, reorders the points so that the leading edge is first.
        """
        min_x_index = np.argmin(self.points[:, 0])
        if min_x_index != 0:
            # Roll the points so that the leading edge becomes the first point
            self.points = np.roll(self.points, -min_x_index, axis=0)

    def _check_chord_horizontal(self, auto_horizontal: bool = True):
        """
        Checks if the chord (line between the first and farthest points) is horizontal.
        If not, it rotates the points to make it horizontal.
        """
        # First ensure the leading edge is at index 0
        self._ensure_leading_edge_first()
        
        lengths = np.linalg.norm(self.points - self.points[0], axis=1)
        max_length_index = np.argmax(lengths)
        self._tail_index = max_length_index.item()
        farthest_point = self.points[max_length_index]
        chord_vector = farthest_point - self.points[0]
        if not np.isclose(chord_vector[1], 0) and auto_horizontal:
            angle = np.arctan2(chord_vector[1], chord_vector[0])
            rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]
            ])
            self.points = self.points @ rotation_matrix.T
        self._chord_length = np.linalg.norm(self.points[self._tail_index] - self.points[0]).item()
    
    def _sort_curve_points_optimized(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Reorders the points on a closed curve to ensure proper airfoil shape.
        Uses angular sorting from leading edge to maintain convex shape.

        @param points: A numpy array of shape (n, 2) representing points on a closed curve.
        @return: The reordered array of points.
        """
        if points.shape[0] < 3:
            return points

        # Find leading edge (leftmost point, if tie then lowest y)
        leftmost_idx = np.lexsort((points[:, 1], points[:, 0]))[0]
        leading_edge = points[leftmost_idx]

        # Calculate distances from leading edge to all other points
        distances = np.linalg.norm(points - leading_edge, axis=1)

        # Find trailing edge (farthest point from leading edge)
        trailing_edge_idx = np.argmax(distances)
        trailing_edge = points[trailing_edge_idx]

        # Calculate angles from leading edge to all other points
        vectors = points - leading_edge
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Normalize angles to [0, 2π]
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        # Separate upper and lower surface points based on y-coordinate relative to leading and trailing edges
        upper_surface_mask = points[:, 1] >= 0
        lower_surface_mask = ~upper_surface_mask

        # Sort upper surface points by increasing x-coordinate
        upper_indices = np.where(upper_surface_mask)[0]
        upper_sorted = upper_indices[np.argsort(points[upper_indices, 0])]

        # Sort lower surface points by decreasing x-coordinate
        lower_indices = np.where(lower_surface_mask)[0]
        lower_sorted = lower_indices[np.argsort(-points[lower_indices, 0])]

        # Combine: leading edge -> upper surface -> trailing edge -> lower surface
        sorted_indices = np.concatenate(([leftmost_idx], upper_sorted, lower_sorted))

        return points[sorted_indices]

    def inplace_rotate(self, angle: float) -> None:
        """
        Rotates the shape by a given angle in radians counter-clockwise.

        @param angle: The angle in radians to rotate the shape.
        """
        self._norm_buffer = None
        self._panel_buffer = None
        self._tangent_buffer = None
        self._length_buffer = None
        self._mid_point_buffer = None
        self._tangent_length_buffer = None
        self._center_buffer = None
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.points = self.points @ rotation_matrix.T
    
    def rotate(self, angle: float) -> 'Airfoil':
        """
        Returns a new Airfoil instance rotated by a given angle in radians counter-clockwise.

        @param angle: The angle in radians to rotate the shape.
        @return: A new Airfoil instance with the rotated points.
        """
        new_airfoil = Airfoil(self.points.copy(), self.max_edge_length)
        new_airfoil.inplace_rotate(angle)
        return new_airfoil
    
    def expand(
        self,
        distance: Union[
            float, 
            tuple[
                Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
                Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            ],
            tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64]
            ]
        ]
    ) -> 'Airfoil':
        """
        Returns a new Airfoil instance expanded by 
        a given distance or the displacement distance function (mapping the position along the edge to the distance)
        along its normal vectors.

        @param distance: The distance by which to expand the shape. Can be:
                        - float: Uniform distance for all panels
                        - tuple of functions (upper_func, lower_func): Functions that take distances from 
                          leading edge and return displacement distances for upper and lower surfaces
                        - tuple of arrays (upper_distances, lower_distances): Arrays specifying displacement
                          distances for each panel. upper_distances shape must be (n_upper_panels,) and 
                          lower_distances shape must be (n_lower_panels,) where:
                          * n_lower_panels: number of panels from leading edge to trailing edge on lower surface
                          * n_upper_panels: number of panels from trailing edge to leading edge on upper surface
        @return: A new Airfoil instance with the expanded points.
        """
        if isinstance(distance, (int, float)):
            panel_normals = self.norm
            new_points = np.zeros((len(self.points) * 2, 2), dtype=np.float64)
            start_points = self.points
            end_points = np.roll(self.points, -1, axis=0)
            new_points[::2] = start_points + panel_normals * distance
            new_points[1::2] = end_points + panel_normals * distance
            return Airfoil(
                new_points,
                self.max_edge_length,
                self.min_edge_length,
                auto_horizontal=False
            )
        
        if isinstance(distance, tuple) and isinstance(distance[0], np.ndarray):
            # distance is a tuple of arrays for upper and lower surfaces
            upper_distances, lower_distances = distance
            upper_distances = cast(npt.NDArray[np.float64], upper_distances)
            lower_distances = cast(npt.NDArray[np.float64], lower_distances)
            
            # Get panel masks for upper and lower surfaces
            panel_upper_mask = np.arange(len(self.points)) >= self.tail_index
            panel_lower_mask = ~panel_upper_mask
            
            # Count panels for each surface
            n_lower_panels = np.sum(panel_lower_mask)
            n_upper_panels = np.sum(panel_upper_mask)
            
            # Validate array shapes
            if len(lower_distances) != n_lower_panels:
                raise ValueError(f"Lower surface distance array length ({len(lower_distances)}) "
                               f"does not match number of lower surface panels ({n_lower_panels})")
            if len(upper_distances) != n_upper_panels:
                raise ValueError(f"Upper surface distance array length ({len(upper_distances)}) "
                               f"does not match number of upper surface panels ({n_upper_panels})")
            
            # Create panel displacement distances array
            panel_displacement_distances = np.zeros(len(self.points))
            if np.any(panel_lower_mask):
                panel_displacement_distances[panel_lower_mask] = lower_distances
            if np.any(panel_upper_mask):
                panel_displacement_distances[panel_upper_mask] = upper_distances
                
            # Apply displacement along normals
            panel_normals = self.norm
            new_points = np.zeros((len(self.points) * 2, 2), dtype=np.float64)
            start_points = self.points
            end_points = np.roll(self.points, -1, axis=0)
            
            new_points[::2] = start_points + panel_normals * panel_displacement_distances.reshape(-1, 1)
            new_points[1::2] = end_points + panel_normals * panel_displacement_distances.reshape(-1, 1)
            return Airfoil(
                new_points,
                self.max_edge_length,
                self.min_edge_length,
                auto_horizontal=False
            )
        
        # If we reach here, distance is a tuple of functions
        upper_func, lower_func = distance
        upper_func = cast(Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], upper_func)
        lower_func = cast(Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], lower_func)
        tangent_lengths = self.tangent_length

        mid_pos = np.insert(
            (tangent_lengths[1:] + tangent_lengths[:-1]) / 2,
            0,
            tangent_lengths[0] / 2
        )

        panel_upper_mask = np.arange(len(self.points)) >= self.tail_index
        panel_lower_mask = ~panel_upper_mask
        panel_displacement_distances = np.zeros(len(self.points))
        if np.any(panel_lower_mask):
            panel_displacement_distances[panel_lower_mask] = lower_func(mid_pos[panel_lower_mask])
        if np.any(panel_upper_mask):
            upper_mid_pos = tangent_lengths[-1] - mid_pos[panel_upper_mask]
            panel_displacement_distances[panel_upper_mask] = upper_func(upper_mid_pos)
        panel_normals = self.norm
        new_points = np.zeros((len(self.points) * 2, 2), dtype=np.float64)
        start_points = self.points
        end_points = np.roll(self.points, -1, axis=0)

        new_points[::2] = start_points + panel_normals * panel_displacement_distances.reshape(-1, 1)
        new_points[1::2] = end_points + panel_normals * panel_displacement_distances.reshape(-1, 1)
        return Airfoil(
            new_points,
            self.max_edge_length,
            self.min_edge_length,
            auto_horizontal=False
        )


    @property
    def norm(self) -> npt.NDArray[np.float64]:
        """
        Returns the normal vectors of the shape's edges, pointed outward.
        """
        if self._norm_buffer is None:
            self._norm_buffer = np.zeros_like(self.points)
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                edge = p2 - p1
                normal = np.array([edge[1], -edge[0]])
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal /= normal_length
                self._norm_buffer[i] = normal
        return self._norm_buffer
    
    @property
    def panel(self) -> npt.NDArray[np.float64]:
        """
        Returns the panel vectors of the shape, returning a n * 2 array
        """
        if self._panel_buffer is None:
            self._panel_buffer = np.roll(self.points, -1, axis=0) - self.points
        return self._panel_buffer

    @property
    def tangent(self) -> npt.NDArray[np.float64]:
        """
        Returns the tangent vectors of the shape's edges.
        """
        if self._tangent_buffer is None:
            panels = self.panel
            self._tangent_buffer = np.zeros_like(panels)
            for i in range(len(panels)):
                panel_length = np.linalg.norm(panels[i])
                if panel_length > 0:
                    self._tangent_buffer[i] = panels[i] / panel_length
        return self._tangent_buffer

    @property
    def length(self) -> npt.NDArray[np.float64]:
        """
        Returns the lengths of the segments of the shape.
        """
        if self._length_buffer is None:
            self._length_buffer = np.zeros(self.points.shape[0], dtype=np.float64)
            p1 = self.points
            p2 = np.roll(self.points, -1, axis=0)
            self._length_buffer = cast(npt.NDArray[np.float64], np.linalg.norm(p2 - p1, axis=1))
        return self._length_buffer

    @property
    def chord(self) -> float:
        """
        Returns the chord length of the airfoil.
        """
        return self._chord_length

    @property
    def midpoint(self) -> npt.NDArray[np.float64]:
        """
        Returns the mid-point of each segment of the shape.
        """
        if self._mid_point_buffer is None:
            p1 = self.points
            p2 = np.roll(self.points, -1, axis=0)
            self._mid_point_buffer = (p1 + p2) / 2
        return self._mid_point_buffer
    
    @property
    def tail_index(self) -> int:
        """
        Returns the index of the tail point (the farthest point from the leading edge).
        """
        return self._tail_index

    @property
    def tangent_length(self) -> npt.NDArray[np.float64]:
        """
        Returns the cumulative tangent lengths along the edges of the shape.
        """
        if self._tangent_length_buffer is None:
            panel_lengths = np.linalg.norm(self.panel, axis=1)
            self._tangent_length_buffer = np.cumsum(panel_lengths)
        return self._tangent_length_buffer

    @property
    def center(self) -> tuple[float, float]:
        """
        Returns the center point coordinates of the airfoil.
        
        The center is calculated as the weighted average of panel midpoints,
        where each panel midpoint is weighted by its corresponding panel length.
        This provides a geometrically meaningful center that accounts for the
        distribution of the airfoil contour.
        
        Formula:
        center = Σ(midpoint[i] * length[i]) / Σ(length[i])
        
        @return: Tuple of (x, y) coordinates representing the airfoil center.
        """
        if self._center_buffer is None:
            midpoints = self.midpoint  # Shape: (n_panels, 2)
            lengths = self.length      # Shape: (n_panels,)
            
            # Calculate weighted average: sum(midpoint * weight) / sum(weight)
            weighted_sum = np.sum(midpoints * lengths[:, np.newaxis], axis=0)
            total_length = np.sum(lengths)
            
            center_coords = weighted_sum / total_length
            self._center_buffer = (float(center_coords[0]), float(center_coords[1]))
        
        return self._center_buffer

    @singledispatchmethod
    def _position_along_edge_impl(
        self,
        x: float | npt.NDArray[np.float64],
        upper: bool | npt.NDArray[np.bool_],
    ) -> tuple[tuple[float, float], int] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        raise NotImplementedError("This method should be implemented for specific types.")
        
    @_position_along_edge_impl.register(float)
    def _(
        self,
        x: float,
        upper: bool,
    ) -> tuple[tuple[float, float], int]:
        """
        Converting the absolute position of some point along the edge of the airfoil and the index of the panel
        """
        tangent_lengths = self.tangent_length
        
        if upper:
            # Last - index
            value = tangent_lengths[-1] - x
            ind: int = np.searchsorted(tangent_lengths, value).item()
            ratio = (tangent_lengths[ind] - value) / self.length[ind]
            res: list[float] = (self.points[ind] * ratio + self.points[(ind + 1) % len(self.points)] * (1 - ratio)).tolist()
            return (res[0], res[1]), ind
        else:
            ind: int = np.searchsorted(tangent_lengths, x).item()
            ratio = (tangent_lengths[ind] - x) / self.length[ind]
            res: list[float] = (self.points[ind] * ratio + self.points[(ind + 1) % len(self.points)] * (1 - ratio)).tolist()
            return (res[0], res[1]), ind

    @_position_along_edge_impl.register(np.ndarray)
    def _(
        self,
        x: npt.NDArray[np.float64],
        upper: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Vectorized version of position_along_edge for computing positions of multiple points along the airfoil edge.
        
        @param x: Array of distances along the edge, shape=(n,)
        @param upper: Array of booleans indicating whether to use upper surface, shape=(n,)
        @return: Tuple of (positions_array, panel_indices_array)
                positions_array: shape=(n, 2)
                panel_indices_array: shape=(n,)
        """
        tangent_lengths = self.tangent_length
        x_array = np.asarray(x, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.bool_)
        x_array, upper_array = np.broadcast_arrays(x_array, upper_array)
        n_points = x_array.size
        positions = np.zeros((n_points, 2), dtype=np.float64)
        indices = np.zeros(n_points, dtype=np.int64)
        upper_mask = upper_array.flatten()
        lower_mask = ~upper_mask
        
        if np.any(upper_mask):
            x_upper = x_array.flatten()[upper_mask]
            values_upper = tangent_lengths[-1] - x_upper
            inds_upper = np.searchsorted(tangent_lengths, values_upper)
            inds_upper = np.clip(inds_upper, 0, len(tangent_lengths) - 1)
            ratios_upper = (tangent_lengths[inds_upper] - values_upper) / self.length[inds_upper]
            next_inds_upper = (inds_upper + 1) % len(self.points)
            positions_upper = (self.points[inds_upper] * ratios_upper[:, np.newaxis] + 
                             self.points[next_inds_upper] * (1 - ratios_upper)[:, np.newaxis])
            positions[upper_mask] = positions_upper
            indices[upper_mask] = inds_upper
        
        if np.any(lower_mask):
            x_lower = x_array.flatten()[lower_mask]
            inds_lower = np.searchsorted(tangent_lengths, x_lower)
            inds_lower = np.clip(inds_lower, 0, len(tangent_lengths) - 1)
            ratios_lower = (tangent_lengths[inds_lower] - x_lower) / self.length[inds_lower]
            next_inds_lower = (inds_lower + 1) % len(self.points)
            positions_lower = (self.points[inds_lower] * ratios_lower[:, np.newaxis] + 
                             self.points[next_inds_lower] * (1 - ratios_lower)[:, np.newaxis])
            positions[lower_mask] = positions_lower
            indices[lower_mask] = inds_lower
        
        positions = positions.reshape(x_array.shape + (2,))
        indices = indices.reshape(x_array.shape)
        return positions, indices
        
    @overload
    def position_along_edge(
        self,
        x: float,
        upper: bool,
    ) -> tuple[tuple[float, float], int]: ...

    @overload
    def position_along_edge(
        self,
        x: npt.NDArray[np.float64],
        upper: npt.NDArray[np.bool_],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...

    def position_along_edge(
        self,
        x: float | npt.NDArray[np.float64],
        upper: bool | npt.NDArray[np.bool_],
    ) -> tuple[tuple[float, float], int] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Converts the absolute position of some point along the edge of the airfoil and the index of the panel.
        If x is a single float, returns a single position and index.
        If x is an array, returns an array of positions and indices.
        
        @param x: The distance along the edge (float or array).
        @param upper: Whether to use the upper surface (bool or array).
        @return: A tuple containing either a single position and index or arrays of positions and indices.
        """
        return self._position_along_edge_impl(x, upper)
