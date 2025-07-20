import numpy as np
import numpy.typing as npt

from collections.abc import Callable
from typing import Any

def reshape2(
    func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    def wrapper(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if points.ndim < 2:
            raise ValueError("Input points must be at least 2-dimensional.")
        if points.shape[-1] != 2:
            raise ValueError("Input points must have shape (..., 2).")
        shape = points.shape[:-1]
        result = func(points.reshape(-1, 2))
        if result.ndim == 1:
            result = result.reshape(shape)
        else:
            result = result.reshape(*shape, -1)
        return result
    return wrapper

def reshape2_method(
    func: Callable[[Any, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
):
    def wrapper(self: Any, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if points.ndim < 2:
            raise ValueError("Input points must be at least 2-dimensional.")
        if points.shape[-1] != 2:
            raise ValueError("Input points must have shape (..., 2).")
        shape = points.shape[:-1]
        result = func(self, points.reshape(-1, 2))
        if result.ndim == 1:
            result = result.reshape(shape)
        else:
            result = result.reshape(*shape, -1)
        return result
    return wrapper

def double_reshape2(
    func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    def wrapper(points1: npt.NDArray[np.float64], points2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if points1.ndim < 2 or points2.ndim < 2:
            raise ValueError("Input points must be at least 2-dimensional.")
        if points1.shape[-1] != 2 or points2.shape[-1] != 2:
            raise ValueError("Input points must have shape (..., 2).")
        shape1 = points1.shape[:-1]
        shape2 = points2.shape[:-1]
        result = func(points1.reshape(-1, 2), points2.reshape(-1, 2))
        if result.ndim == 1:
            result = result.reshape(shape1 + shape2)
        else:
            result = result.reshape(*shape1, *shape2, -1)
        return result
    return wrapper  
