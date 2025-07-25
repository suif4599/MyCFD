import numpy as np
import numpy.typing as npt

from .potential import Potential
from functools import singledispatch
from typing import overload
from ..reshape import reshape2, double_reshape2


class VortexPotential(Potential):
    """
    Class for vortex potential functions.\n
    Using vortex panels instead of source panels for better numerical stability.\n
    Far field velocity function: $\\vec{v} = \\frac{\\Gamma}{2\\pi} \\frac{(-y, x)}{x^2 + y^2}$.\n
    Velocity function: $\\vec{v} = \\frac{\\Gamma}{2\\pi L} \\left( -y, x \\right)$ **in local coordinates** (it returns in global coordinates).\n
    """

    @singledispatch
    @staticmethod
    def _far_velocity_impl(*args, **kwargs):
        raise NotImplementedError("Unsupported type for vortex far field velocity calculation.")
    
    @_far_velocity_impl.register(np.ndarray)
    @reshape2
    @staticmethod
    def _(
        points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r_squared = points[:, 0]**2 + points[:, 1]**2
        min_distance = 1e-10
        vx = np.zeros_like(points[:, 0])
        vy = np.zeros_like(points[:, 1])
        mask = r_squared > min_distance**2
        if np.any(mask):
            factor = 1.0 / (2 * np.pi * r_squared[mask])
            vx[mask] = -factor * points[mask, 1]
            vy[mask] = factor * points[mask, 0]
        near_singular = ~mask
        if np.any(near_singular):
            smoothed_r_squared = r_squared[near_singular] + min_distance**2
            factor = 1.0 / (2 * np.pi * smoothed_r_squared)
            vx[near_singular] = -factor * points[near_singular, 1]
            vy[near_singular] = factor * points[near_singular, 0]
        return np.column_stack((vx, vy))
    
    @_far_velocity_impl.register(tuple)
    @staticmethod
    def _(point: tuple[float, float]) -> tuple[float, float]:
        point_array = np.array([point], dtype=np.float64)
        result = VortexPotential._far_velocity_impl(point_array)
        return float(result[0, 0]), float(result[0, 1])
    
    @_far_velocity_impl.register(list)
    @staticmethod
    def _(point: list[float]) -> list[float]:
        point_array = np.array([point], dtype=np.float64)
        result = VortexPotential._far_velocity_impl(point_array)
        return [float(result[0, 0]), float(result[0, 1])]

    @_far_velocity_impl.register(float)
    @staticmethod
    def _(x: float, y: float) -> tuple[float, float]:
        point_array = np.array([[x, y]], dtype=np.float64)
        result = VortexPotential._far_velocity_impl(point_array)
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
        return VortexPotential._far_velocity_impl(*args, **kwargs)
    
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
        
        epsilon = 1e-14
        denominator = r_squared - l * x_local
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        
        r_squared_safe = np.maximum(r_squared, epsilon)
        r2_squared_safe = np.maximum(r2_squared, epsilon)
        
        vx_local: npt.NDArray[np.float64] = -np.arctan(l * y_local / denominator) / (2 * np.pi * l)
        vy_local: npt.NDArray[np.float64] = np.log(r_squared_safe / r2_squared_safe) / (4 * np.pi * l)
        vx = vx_local * tangent[..., 0] + vy_local * normal[..., 0]
        vy = vx_local * tangent[..., 1] + vy_local * normal[..., 1]
        return np.stack((vx, vy), axis=-1)
    
    @_velocity_impl.register(tuple)
    @staticmethod
    def _(r: tuple[float, float], panal: tuple[float, float]) -> tuple[float, float]:
        r_array = np.array(r).reshape(1, -1)
        panal_array = np.array(panal).reshape(1, -1)
        return tuple(VortexPotential._velocity_impl(r_array, panal_array).tolist()[0])
    
    @_velocity_impl.register(list)
    @staticmethod
    def _(r: list[float], panal: list[float]) -> list[float]:
        r_array = np.array(r).reshape(1, -1)
        panal_array = np.array(panal).reshape(1, -1)
        return VortexPotential._velocity_impl(r_array, panal_array).tolist()[0]
    
    @_velocity_impl.register(float)
    @staticmethod
    def _(x: float, y: float, panal_x: float, panal_y: float) -> tuple[float, float]:
        r = np.array([x, y]).reshape(1, -1)
        panal = np.array([panal_x, panal_y]).reshape(1, -1)
        return tuple(VortexPotential._velocity_impl(r, panal).tolist()[0])
    
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
        return VortexPotential._velocity_impl(*args, **kwargs)
    
