import numpy as np
import numpy.typing as npt

from functools import singledispatch
from abc import ABCMeta, abstractmethod
from typing import overload


class Potential(metaclass=ABCMeta):
    """
    Class for potential functions
    """

    @singledispatch
    @staticmethod
    def _far_velocity_impl(*args, **kwargs):
        raise NotImplementedError("Unsupported type for far field velocity calculation.")
    
    @_far_velocity_impl.register(np.ndarray)
    @staticmethod
    def _(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Unsupported type for far field velocity calculation.")
    
    @_far_velocity_impl.register(tuple)
    @staticmethod
    def _(point: tuple[float, float]) -> tuple[float, float]:
        raise NotImplementedError("Unsupported type for far field velocity calculation.")
    
    @_far_velocity_impl.register(list)
    @staticmethod
    def _(point: list[float]) -> list[float]:
        raise NotImplementedError("Unsupported type for far field velocity calculation.")

    @_far_velocity_impl.register(float)
    @staticmethod
    def _(x: float, y: float) -> tuple[float, float]:
        raise NotImplementedError("Unsupported type for far field velocity calculation.")
    
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
    @abstractmethod
    def far_velocity(*args, **kwargs) -> npt.NDArray[np.float64] | tuple[float, float] | list[float]:
        """
        Calculate the far field velocity at given points (less accurate near surfaces).
        """
        return Potential._far_velocity_impl(*args, **kwargs)
    
    @singledispatch
    @staticmethod
    def _velocity_impl(*args, **kwargs):
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(np.ndarray)
    @staticmethod
    def _(
        r: npt.NDArray[np.float64],
        panal: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(tuple)
    @staticmethod
    def _(
        r: tuple[float, float],
        panal: tuple[float, float],
    ) -> tuple[float, float]:
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(list)
    @staticmethod
    def _(
        r: list[float],
        panal: list[float],
    ) -> list[float]:
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(float)
    @staticmethod
    def _(
        x: float,
        y: float,
        panal_x: float,
        panal_y: float,
    ) -> float:
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
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
        panal: tuple[float, float],
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
        x: float,
        y: float,
        panal_x: float,
        panal_y: float,
    ) -> tuple[float, float]: ...

    @staticmethod
    @abstractmethod
    def velocity(*args, **kwargs) -> npt.NDArray[np.float64] | tuple[float, float] | list[float]:
        """
        Calculate the accurate velocity at given points near the surface of the airfoil.

        @param r: Point - LeftEdge
        @param panal: RightEdge - LeftEdge
        """
        return Potential._velocity_impl(*args, **kwargs)
