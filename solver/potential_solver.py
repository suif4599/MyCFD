from typing import Any
from collections.abc import Callable

from .base import Var, Const, Output
from .solver import Solver
from simulator import PanelMethod
from tools import Potential, Airfoil
import numpy as np
import numpy.typing as npt

TrueArray = np.array([True], dtype=np.bool)
FalseArray = np.array([False], dtype=np.bool)


class PotentialSolver(Solver):
    """
    Solver for potential flow problems.
    """
    def __init__(self, potential: Potential):
        super().__init__(
            _input=[
                Const("airfoil", Airfoil),
                Const("aoa", float),
                Const("rho", float),
                Const("v_{inf}", float),
                Const("p_{inf}", float),
                Var("delta^{*}_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Var("delta^{*}_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
            ],
            _output=[
                Var("U_{e,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Var("U_{e,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("p_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("p_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
            ]
        )
        self._potential = potential

    def solve(self, _input: dict[Var, Any]) -> dict[Var, Any]:
        """
        _input=[
            Const("airfoil", Airfoil),
            Const("aoa", float),
            Const("rho", float),
            Const("v_{inf}", float),
            Const("p_{inf}", float),
            Var("delta^{*}_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Var("delta^{*}_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
        ],
        _output=[
            Var("U_{e,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Var("U_{e,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("p_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("p_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
        ]
        """
        airfoil: Airfoil = _input[Const("airfoil")]
        aoa: float = _input[Const("aoa")]
        rho: float = _input[Const("rho")]
        v_inf: float = _input[Const("v_{inf}")]
        p_inf: float = _input[Const("p_{inf}")]
        delta_upper: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = _input[Var("delta^{*}_{upper}")]
        delta_lower: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = _input[Var("delta^{*}_{lower}")]

        airfoil = airfoil.expand(
            (
                delta_upper,
                delta_lower
            )
        )
        pm = PanelMethod(
            airfoil=airfoil
        )
        pm.compute(
            potential_func=self._potential,
            attack_angle=aoa,
            velocity=v_inf,
            pressure=p_inf,
            rho=rho,
            apply_kutta_condition=True
        )

        def create_surface_function(
            is_upper: bool,
            is_velocity: bool
        ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:

            def surface_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                arr = TrueArray if is_upper else FalseArray
                pos, ind = airfoil.position_along_edge(x, arr)
                eps = 1e-6
                pos += eps * airfoil.norm[ind]
                if is_velocity:
                    velocity_vec = pm.velocity(pos)
                    result: npt.NDArray[np.float64] = np.linalg.norm(velocity_vec, axis=1)
                    invalid_mask = np.isnan(result) | (result <= 0.0)
                    if np.any(invalid_mask):
                        result = PotentialSolver._ensure_valid(result, invalid_mask)
                else:  # pressure
                    result = pm.pressure(pos)
                    invalid_mask = np.isnan(result) | np.isinf(result)
                    if np.any(invalid_mask):
                        result = PotentialSolver._ensure_valid(result, invalid_mask, default_value=p_inf)
                return result
            return surface_func
        
        U_e_upper = create_surface_function(True, True)
        U_e_lower = create_surface_function(False, True)
        p_upper = create_surface_function(True, False)
        p_lower = create_surface_function(False, False)
        
        return {
            Var("U_{e,upper}"): U_e_upper,
            Var("U_{e,lower}"): U_e_lower,
            Var("p_{upper}"): p_upper,
            Var("p_{lower}"): p_lower,
        }
    
    @staticmethod
    def _ensure_valid(
        values: npt.NDArray[np.float64], 
        invalid_mask: npt.NDArray[np.bool], 
        default_value: float = 1e-6,
        max_neighbors: int = 5
    ) -> npt.NDArray[np.float64]:
        """
        Ensure all values are valid by replacing invalid values with neighbor averages.
        
        @param values: The original array of values.
        @param invalid_mask: Boolean mask indicating which values are invalid (True for invalid).
        @param default_value: Value to use if all values are invalid or no valid neighbors are found. Default is 1e-6.
        @param max_neighbors: Maximum number of nearest valid neighbors to use for averaging. Default is 5.
        @return: Array where invalid values are replaced by weighted averages of nearest valid neighbors.
        """
        if not np.any(invalid_mask):
            return values
        result = values.copy()
        valid_mask = ~invalid_mask
        if not np.any(valid_mask):
            result[:] = default_value
            return result
        
        invalid_indices = np.where(invalid_mask)[0]
        valid_indices = np.where(valid_mask)[0]
        distances = np.abs(invalid_indices[:, np.newaxis] - valid_indices[np.newaxis, :])
        n_neighbors = min(max_neighbors, len(valid_indices))
        closest_neighbor_indices = np.argpartition(distances, n_neighbors-1, axis=1)[:, :n_neighbors]
        row_indices = np.arange(len(invalid_indices))[:, np.newaxis]
        neighbor_distances = distances[row_indices, closest_neighbor_indices]
        neighbor_valid_indices = valid_indices[closest_neighbor_indices]
        neighbor_values = values[neighbor_valid_indices]
        weights = 1.0 / (neighbor_distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        weighted_averages = np.sum(weights * neighbor_values, axis=1)
        result[invalid_indices] = weighted_averages
        return result


