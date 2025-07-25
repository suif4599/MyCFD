import numpy as np
import numpy.typing as npt

from .potential import Potential
from functools import singledispatch
from ..reshape import reshape2
from typing import overload


class SourcePotential(Potential):
    """
    Class for source potential functions.
    Implements both ideal (point) and finite-length source panels.
    Far field velocity function (ideal point source): $v_r = \\frac{Q}{2\\pi r},\\quad v_\\theta = 0$.
    Velocity function: $\\vec{v} = \\frac{Q}{2\\pi L} \\left( -y, x \\right)$ **in local coordinates** (it returns in global coordinates).
    """

    @singledispatch
    @staticmethod
    def _far_velocity_impl(*args, **kwargs):
        raise NotImplementedError("Unsupported type for source far field velocity calculation.")
    
    @_far_velocity_impl.register(np.ndarray)
    @reshape2
    @staticmethod
    def _(
        points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r_squared = points[:, 0]**2 + points[:, 1]**2
        r = np.sqrt(r_squared)
        min_distance = 1e-10
        mask = r > min_distance
        vx = np.zeros_like(r)
        vy = np.zeros_like(r)
        valid_points = mask
        if np.any(valid_points):
            factor = 1.0 / (2 * np.pi * r_squared[valid_points])
            vx[valid_points] = factor * points[valid_points, 0]
            vy[valid_points] = factor * points[valid_points, 1]
        near_singular = ~mask
        if np.any(near_singular):
            smoothed_r_squared = r_squared[near_singular] + min_distance**2
            factor = 1.0 / (2 * np.pi * smoothed_r_squared)
            vx[near_singular] = factor * points[near_singular, 0]
            vy[near_singular] = factor * points[near_singular, 1]
        return np.column_stack((vx, vy))
    
    @_far_velocity_impl.register(tuple)
    @staticmethod
    def _(point: tuple[float, float]) -> tuple[float, float]:
        point_array = np.array([point], dtype=np.float64)
        result = SourcePotential._far_velocity_impl(point_array)
        return float(result[0, 0]), float(result[0, 1])
    
    @_far_velocity_impl.register(list)
    @staticmethod
    def _(point: list[float]) -> list[float]:
        point_array = np.array([point], dtype=np.float64)
        result = SourcePotential._far_velocity_impl(point_array)
        return [float(result[0, 0]), float(result[0, 1])]

    @_far_velocity_impl.register(float)
    @staticmethod
    def _(x: float, y: float) -> tuple[float, float]:
        point_array = np.array([[x, y]], dtype=np.float64)
        result = SourcePotential._far_velocity_impl(point_array)
        return float(result[0, 0]), float(result[0, 1])

    @overload
    @staticmethod
    def far_velocity(
        points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    @overload
    @staticmethod
    def far_velocity(
        point: tuple[float, float]
    ) -> tuple[float, float]: ...

    @overload
    @staticmethod
    def far_velocity(
        point: list[float]
    ) -> list[float]: ...

    @overload
    @staticmethod
    def far_velocity(
        x: float, y: float
    ) -> tuple[float, float]: ...

    @staticmethod
    def far_velocity(*args, **kwargs) -> npt.NDArray[np.float64] | tuple[float, float] | list[float]:
        """
        Calculate the far field velocity at given points (less accurate near surfaces).
        """
        return SourcePotential._far_velocity_impl(*args, **kwargs)
    
    @singledispatch
    @staticmethod
    def _velocity_impl(*args, **kwargs):
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(np.ndarray)
    @staticmethod
    def _(
        r: npt.NDArray[np.float64],
        panal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        l: npt.NDArray[np.float64] = np.linalg.norm(panal, axis=-1)
        tangent = panal / l.reshape(-1, 1)
        normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
        r_squared: npt.NDArray[np.float64] = np.sum(r * r, axis=-1)
        r2 = r - panal
        r2_squared: npt.NDArray[np.float64] = np.sum(r2 * r2, axis=-1)
        x_local: npt.NDArray[np.float64] = np.sum(r * tangent, axis=-1)
        y_local: npt.NDArray[np.float64] = np.sum(r * normal, axis=-1)
        vx_local: npt.NDArray[np.float64] = np.log(r_squared / r2_squared) / (4 * np.pi * l)
        vy_local: npt.NDArray[np.float64] = np.arctan(l * y_local / (r_squared - l * x_local)) / (2 * np.pi * l)
        vx = vx_local * tangent[..., 0] + vy_local * normal[..., 0]
        vy = vx_local * tangent[..., 1] + vy_local * normal[..., 1]
        return np.stack((vx, vy), axis=-1)
    
    @_velocity_impl.register(tuple)
    @staticmethod
    def _(r: tuple[float, float], panal: tuple[float, float]) -> tuple[float, float]:
        r_array = np.array(r).reshape(1, -1)
        panal_array = np.array(panal).reshape(1, -1)
        return tuple(SourcePotential._velocity_impl(r_array, panal_array).tolist()[0])
    
    @_velocity_impl.register(list)
    @staticmethod
    def _(r: list[float], panal: list[float]) -> list[float]:
        r_array = np.array(r).reshape(1, -1)
        panal_array = np.array(panal).reshape(1, -1)
        return SourcePotential._velocity_impl(r_array, panal_array).tolist()[0]
    
    @_velocity_impl.register(float)
    @staticmethod
    def _(x: float, y: float, panal_x: float, panal_y: float) -> tuple[float, float]:
        r = np.array([x, y]).reshape(1, -1)
        panal = np.array([panal_x, panal_y]).reshape(1, -1)
        return tuple(SourcePotential._velocity_impl(r, panal).tolist()[0])
    
    @overload
    @staticmethod
    def velocity(
        r: npt.NDArray[np.float64],
        panal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    @overload
    @staticmethod
    def velocity(
        r: tuple[float, float],
        panal: tuple[float, float]
    ) -> tuple[float, float]: ...

    @overload
    @staticmethod
    def velocity(
        r: list[float],
        panal: list[float]
    ) -> list[float]: ...

    @overload
    @staticmethod
    def velocity(
        x: float, y: float,
        panal_x: float, panal_y: float
    ) -> tuple[float, float]: ...

    @staticmethod
    def velocity(*args, **kwargs) -> npt.NDArray[np.float64] | tuple[float, float] | list[float]:
        """
        Calculate the accurate velocity at given points near the surface.
        """
        return SourcePotential._velocity_impl(*args, **kwargs)
    
