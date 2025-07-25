import numpy as np
import numpy.typing as npt

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QImage, QColor, QPen, QFont, QPaintEvent
from collections.abc import Callable
from tools.airfoil import Airfoil


class View(QWidget):
    _field_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    _x_min: float
    _x_max: float
    _y_min: float
    _y_max: float
    _calc_scale: float
    _airfoil: Airfoil | None
    _calc_width: int
    _calc_height: int
    _field_image: QImage | None
    _vmin: float | np.floating
    _vmax: float | np.floating
    _is_vector_field: bool
    _vector_field: npt.NDArray[np.float64] | None
    _vector_points: npt.NDArray[np.float64] | None

    def __init__(
        self,
        field_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        width: int = 800, 
        height: int = 600,
        x_range: tuple[float, float] = (-10, 10),
        y_range: tuple[float, float] = (-10, 10),
        calc_scale: float = 0.5,
        airfoil: Airfoil | None = None,
        parent: QWidget | None = None
    ) -> None:
        """
        A View widget for visualizing 2D vector or scaler fields.

        @param field_func: Function that takes an array of points (N, 2) and returns field intensities (N,).
        @param width: Width of the display in pixels.
        @param height: Height of the display in pixels.
        @param x_range: Range of x coordinates as (min, max).
        @param y_range: Range of y coordinates as (min, max).
        @param calc_scale: Scale factor for calculation resolution (0-1).
        @param airfoil: Airfoil object with points property for masking interior.
        @param refine_factor: Densify the grid around the airfoil.expand(refine_factor * airfoil.chord)
        """
        super().__init__(parent)
        self._field_func = field_func
        self._x_min, self._x_max = x_range
        self._y_min, self._y_max = y_range
        self._calc_scale = calc_scale
        self._airfoil = airfoil
        
        self._calc_width = max(1, int(width * calc_scale))
        self._calc_height = max(1, int(height * calc_scale))
        
        self.setFixedSize(width, height)
        self._refine_range()
        
        self._field_image = None
        self._vmin = 0
        self._vmax = 1
        self._is_vector_field = False
        self._vector_field = None
        self._vector_points = None
        self._update_field()

    def _refine_range(self) -> None:
        """Ensure ranges won't affect aspect ratio"""
        if (self._x_max - self._x_min) / (self._y_max - self._y_min) > self.width() / self.height():
            center_y = (self._y_max + self._y_min) / 2
            height = (self._x_max - self._x_min) * self.height() / self.width()
            self._y_min = center_y - height / 2
            self._y_max = center_y + height / 2
        else:
            center_x = (self._x_max + self._x_min) / 2
            width = (self._y_max - self._y_min) * self.width() / self.height()
            self._x_min = center_x - width / 2
            self._x_max = center_x + width / 2
        
    def _world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        sx = (x - self._x_min) / (self._x_max - self._x_min) * self.width()
        sy = (1 - (y - self._y_min) / (self._y_max - self._y_min)) * self.height()
        return sx, sy

    def _screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        x = sx / self.width() * (self._x_max - self._x_min) + self._x_min
        y = (1 - sy / self.height()) * (self._y_max - self._y_min) + self._y_min
        return x, y
    
    def _calculate_tick_spacing(self, range_min: float, range_max: float, target_ticks: int = 6) -> tuple[float, float]:
        """
        Calculate nice tick spacing for axis labels
        
        @param range_min: Minimum value of the range
        @param range_max: Maximum value of the range  
        @param target_ticks: Target number of ticks
        @return: (tick_spacing, start_tick) where start_tick is the first tick value
        """
        range_span = range_max - range_min
        rough_spacing = range_span / target_ticks
        magnitude = 10 ** np.floor(np.log10(rough_spacing))
        normalized = rough_spacing / magnitude
        if normalized <= 1:
            nice_spacing = 1 * magnitude
        elif normalized <= 2:
            nice_spacing = 2 * magnitude
        elif normalized <= 5:
            nice_spacing = 5 * magnitude
        else:
            nice_spacing = 10 * magnitude
        start_tick = np.ceil(range_min / nice_spacing) * nice_spacing
        return nice_spacing, start_tick
    
    def _point_in_polygon(self, points: npt.NDArray[np.float64], polygon: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Check if points are inside polygon using improved ray casting algorithm
        
        @param points: Array of points to check (N, 2)
        @param polygon: Array of polygon vertices (M, 2)
        @return: Boolean array indicating which points are inside (N,)
        """
        x = points[:, 0]
        y = points[:, 1]
        n = len(polygon)
        inside = np.zeros(len(points), dtype=bool)
        
        # Close the polygon if not already closed
        poly_x = np.append(polygon[:, 0], polygon[0, 0])
        poly_y = np.append(polygon[:, 1], polygon[0, 1])
        
        for i in range(n):
            p1x = poly_x[i]
            p1y = poly_y[i]
            p2x = poly_x[i + 1]
            p2y = poly_y[i + 1]
            if p1y == p2y:
                continue
            if p1y > p2y:
                p1x, p1y, p2x, p2y = p2x, p2y, p1x, p1y
            mask = (y > p1y) & (y <= p2y)
            if np.any(mask):
                dy = p2y - p1y
                dx = p2x - p1x
                t = (y[mask] - p1y) / dy
                x_intersect = p1x + t * dx
                inside[mask] = inside[mask] ^ (x[mask] < x_intersect)
        return inside
    
    def _value_to_color(self, value: float) -> QColor:
        """Map normalized field intensity value to color (blue-cyan-yellow-red)"""
        if np.isnan(value) or np.isinf(value):
            return QColor(0, 0, 0)
            
        value = np.clip(value, 0.0, 1.0)
        
        if value < 0.25:
            # Blue to cyan
            r = 0
            g = value * 4 * 255
            b = 255
        elif value < 0.5:
            # Cyan to green
            r = 0
            g = 255
            b = (1 - (value - 0.25) * 4) * 255
        elif value < 0.75:
            # Green to yellow
            r = (value - 0.5) * 4 * 255
            g = 255
            b = 0
        else:
            # Yellow to red
            r = 255
            g = (1 - (value - 0.75) * 4) * 255
            b = 0
            
        r_int = int(np.clip(r, 0, 255))
        g_int = int(np.clip(g, 0, 255))
        b_int = int(np.clip(b, 0, 255))
        return QColor(r_int, g_int, b_int)
    
    def _update_field(self) -> None:
        """Update flow field image (using vectorized calculation)"""
        x = np.linspace(self._x_min, self._x_max, self._calc_width)
        y = np.linspace(self._y_min, self._y_max, self._calc_height)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        field_data = self._field_func(points)
        
        mask_inside = None
        if self._airfoil is not None:
            mask_inside = self._point_in_polygon(points, self._airfoil.points)
        
        if field_data.ndim == 1:
            # Scalar field
            self._is_vector_field = False
            intensities = field_data
            valid_mask = np.isfinite(intensities)
            if np.any(valid_mask):
                self._vmin = np.percentile(intensities[valid_mask], 1)
                self._vmax = np.percentile(intensities[valid_mask], 99)
                if np.isclose(self._vmax - self._vmin, 0):
                    self._vmin = np.min(intensities[valid_mask])
                    self._vmax = np.max(intensities[valid_mask]) + 1e-9
                normalized = (intensities - self._vmin) / (self._vmax - self._vmin)
                normalized = np.where(valid_mask, normalized, 0.0)
            else:
                self._vmin, self._vmax = 0.0, 1.0
                normalized = np.zeros_like(intensities)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            # Vector field
            self._is_vector_field = True
            self._vector_field = field_data.reshape(self._calc_height, self._calc_width, 2)
            self._vector_points = points.reshape(self._calc_height, self._calc_width, 2)
            magnitudes = np.linalg.norm(field_data, axis=1)
            valid_mask = np.isfinite(magnitudes)
            if np.any(valid_mask):
                self._vmin = np.percentile(magnitudes[valid_mask], 1)
                self._vmax = np.percentile(magnitudes[valid_mask], 99)
                if np.isclose(self._vmax - self._vmin, 0):
                    self._vmin = np.min(magnitudes[valid_mask])
                    self._vmax = np.max(magnitudes[valid_mask]) + 1e-9
                normalized = (magnitudes - self._vmin) / (self._vmax - self._vmin)
                normalized = np.where(valid_mask, normalized, 0.0)
            else:
                self._vmin, self._vmax = 0.0, 1.0
                normalized = np.zeros_like(magnitudes)
            normalized = np.clip(normalized, 0.0, 1.0)
        
        self._field_image = QImage(self._calc_width, self._calc_height, QImage.Format.Format_RGB32)
        h0 = self._calc_height - 1
        for i in range(self._calc_height):
            for j in range(self._calc_width):
                idx = i * self._calc_width + j
                if mask_inside is not None and mask_inside[idx]:
                    color = QColor(255, 255, 255)
                else:
                    norm_value = normalized[idx]
                    color = self._value_to_color(norm_value)
                self._field_image.setPixelColor(j, h0 - i, color)
        
        self.update()
    
    def set_ranges(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        """Set coordinate ranges"""
        self._x_min, self._x_max = x_range
        self._y_min, self._y_max = y_range
        self._update_field()
    
    def set_field_function(self, func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:
        self._field_func = func
        self._update_field()
    
    def set_calculation_scale(self, scale: float) -> None:
        self._calc_scale = max(0.1, min(1.0, scale))
        self._calc_width = max(1, int(self.width() * self._calc_scale))
        self._calc_height = max(1, int(self.height() * self._calc_scale))
        self._update_field()
    
    def paintEvent(self, a0: QPaintEvent | None) -> None:
        if self._field_image is None:
            return
            
        painter = QPainter(self)
        scaled_img = self._field_image.scaled(
            self.width(), 
            self.height()
        )
        painter.drawImage(0, 0, scaled_img)
        if self._is_vector_field and self._vector_field is not None:
            self._draw_vectors(painter)
        self._draw_axes(painter)
        if self._airfoil is not None:
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            airfoil_points = self._airfoil.points
            for i in range(len(airfoil_points)):
                p1 = airfoil_points[i]
                p2 = airfoil_points[(i + 1) % len(airfoil_points)]
                sx1, sy1 = self._world_to_screen(p1[0], p1[1])
                sx2, sy2 = self._world_to_screen(p2[0], p2[1])
                painter.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
        self._draw_color_legend(painter)
    
    def _draw_axes(self, painter: QPainter) -> None:
        """Draw coordinate axes with tick marks and labels"""
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        sx0, sy0 = self._world_to_screen(self._x_min, 0)
        sx1, sy1 = self._world_to_screen(self._x_max, 0)
        
        if self._y_min <= 0 <= self._y_max:
            painter.drawLine(int(sx0), int(sy0), int(sx1), int(sy1))
            x_spacing, x_start = self._calculate_tick_spacing(self._x_min, self._x_max)
            x_tick = x_start
            while x_tick <= self._x_max:
                if self._x_min <= x_tick <= self._x_max:
                    tick_sx, tick_sy = self._world_to_screen(x_tick, 0)
                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    tick_length = 5
                    painter.drawLine(int(tick_sx), int(tick_sy) - tick_length, 
                                   int(tick_sx), int(tick_sy) + tick_length)
                    if abs(x_tick) < 1e-10:
                        label = "0"
                    else:
                        label = f"{x_tick:.3g}"
                    text_width = painter.fontMetrics().horizontalAdvance(label)
                    text_x = int(tick_sx) - text_width // 2
                    text_y = int(tick_sy) + tick_length + 15
                    painter.setPen(QPen(QColor(255, 255, 255), 3))
                    painter.drawText(text_x, text_y, label)
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.drawText(text_x, text_y, label)
                x_tick += x_spacing
        
        sx0, sy0 = self._world_to_screen(0, self._y_min)
        sx1, sy1 = self._world_to_screen(0, self._y_max)

        if self._x_min <= 0 <= self._x_max:
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawLine(int(sx0), int(sy0), int(sx1), int(sy1))
            y_spacing, y_start = self._calculate_tick_spacing(self._y_min, self._y_max)
            y_tick = y_start
            while y_tick <= self._y_max:
                if self._y_min <= y_tick <= self._y_max:
                    tick_sx, tick_sy = self._world_to_screen(0, y_tick)
                    painter.setPen(QPen(QColor(0, 0, 0), 2))
                    tick_length = 5
                    painter.drawLine(int(tick_sx) - tick_length, int(tick_sy), 
                                   int(tick_sx) + tick_length, int(tick_sy))
                    if abs(y_tick) > 1e-10:
                        label = f"{y_tick:.3g}"
                        text_width = painter.fontMetrics().horizontalAdvance(label)
                        text_x = int(tick_sx) - tick_length - text_width - 5
                        text_y = int(tick_sy) + 4
                        painter.setPen(QPen(QColor(255, 255, 255), 3))
                        painter.drawText(text_x, text_y, label)
                        painter.setPen(QPen(QColor(0, 0, 0), 1))
                        painter.drawText(text_x, text_y, label)
                
                y_tick += y_spacing
    
    def _draw_vectors(self, painter: QPainter) -> None:
        """Draw vector arrows on the field"""
        if self._vector_field is None or self._vector_points is None:
            return
            
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        step_i = max(1, self._calc_height // 20)
        step_j = max(1, self._calc_width // 20)
        world_width = self._x_max - self._x_min
        world_height = self._y_max - self._y_min
        scale_factor = min(world_width, world_height) / 20.0
        for i in range(0, self._calc_height, step_i):
            for j in range(0, self._calc_width, step_j):
                vector = self._vector_field[i, j]
                point = self._vector_points[i, j]
                if self._airfoil is not None:
                    inside = self._point_in_polygon(point.reshape(1, -1), self._airfoil.points)[0]
                    if inside:
                        continue
                sx, sy = self._world_to_screen(point[0], point[1])
                magnitude = np.linalg.norm(vector)
                if magnitude < 1e-10:
                    continue
                normalized_vector = vector / magnitude
                scaled_vector = normalized_vector * scale_factor * min(magnitude / self._vmax, 1.0)
                end_world_x = point[0] + scaled_vector[0]
                end_world_y = point[1] + scaled_vector[1]
                end_sx, end_sy = self._world_to_screen(end_world_x, end_world_y)
                painter.drawLine(int(sx), int(sy), int(end_sx), int(end_sy))
                self._draw_arrowhead(painter, sx, sy, end_sx, end_sy)
    
    def _draw_arrowhead(self, painter: QPainter, start_x: float, start_y: float, end_x: float, end_y: float) -> None:
        """Draw an arrowhead at the end of a vector"""
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-10:
            return
        dx /= length
        dy /= length
        head_length = min(8, length * 0.3)
        head_angle = 0.5
        cos_angle = np.cos(head_angle)
        sin_angle = np.sin(head_angle)
        left_x = end_x - head_length * (dx * cos_angle - dy * sin_angle)
        left_y = end_y - head_length * (dy * cos_angle + dx * sin_angle)
        right_x = end_x - head_length * (dx * cos_angle + dy * sin_angle)
        right_y = end_y - head_length * (dy * cos_angle - dx * sin_angle)
        painter.drawLine(int(end_x), int(end_y), int(left_x), int(left_y))
        painter.drawLine(int(end_x), int(end_y), int(right_x), int(right_y))
    
    def _draw_color_legend(self, painter: QPainter) -> None:
        """Draw a color legend on the right side of the view"""
        legend_width = min(30, self.width() // 10)
        legend_height = min(200, self.height() // 3)
        legend_margin = min(40, self.width() // 20)
        legend_x = self.width() - legend_width - legend_margin
        legend_y = legend_margin
        num_segments = 100
        segment_height = legend_height / num_segments
        
        for i in range(num_segments):
            norm_value = 1.0 - (i / (num_segments - 1))
            color = self._value_to_color(norm_value)
            y_pos = legend_y + i * segment_height
            painter.fillRect(legend_x, int(y_pos), 
                           legend_width, int(segment_height) + 1, color)
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawRect(legend_x, legend_y, legend_width, legend_height)
        painter.setPen(QColor(0, 0, 0))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        max_text = f"{self._vmax:.2f}"
        painter.drawText(legend_x - 5 - painter.fontMetrics().horizontalAdvance(max_text), 
                        legend_y + 5, max_text)
        min_text = f"{self._vmin:.2f}"
        painter.drawText(legend_x - 5 - painter.fontMetrics().horizontalAdvance(min_text), 
                        legend_y + legend_height + 5, min_text)
        mid_value = (self._vmax + self._vmin) / 2
        mid_text = f"{mid_value:.2f}"
        painter.drawText(legend_x - 5 - painter.fontMetrics().horizontalAdvance(mid_text), 
                        legend_y + legend_height // 2 + 5, mid_text)