from typing import Any, cast
from collections.abc import Callable

from .base import Var, Const, Output
from .solver import Solver
from simulator import PanelMethod
from tools import Potential, Airfoil
import numpy as np
import numpy.typing as npt
from warnings import warn

TrueArray = np.array([True], dtype=bool)
FalseArray = np.array([False], dtype=bool)


class PotentialSolver(Solver):
    """
    Solver for potential flow problems.
    """
    def __init__(
            self,
            potential: Potential,
            sampling_scale_factor: float = 0.01,
            apply_kutta_condition: bool = False,
            apply_leading_edge_condition: bool = False):
        super().__init__(
            _input=[
                Const("airfoil", Airfoil),
                Const("aoa", float),
                Const("rho", float),
                Const("v_{inf}", float),
                Const("p_{inf}", float),
                Var("delta^{*}", npt.NDArray[np.float64])
            ],
            _output=[
                Var("U_{e}", npt.NDArray[np.float64]),
                Output("delta^{*}_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("delta^{*}_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("U_{e,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("U_{e,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("p_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("p_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
            ]
        )
        self._potential = potential
        self._sampling_scale_factor = sampling_scale_factor
        self._apply_kutta_condition = apply_kutta_condition
        self._apply_leading_edge_condition = apply_leading_edge_condition

    def solve(self, _input: dict[Var[Any], Any], need_output: bool) -> dict[Var[Any], Any]:
        """
        _input=[
            Const("airfoil", Airfoil),
            Const("aoa", float),
            Const("rho", float),
            Const("v_{inf}", float),
            Const("p_{inf}", float),
            Var("delta^{*}", npt.NDArray[np.float64])
        ],
        _output=[
            Var("U_{e}", npt.NDArray[np.float64]),
            Output("delta^{*}_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("delta^{*}_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("U_{e,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("U_{e,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("p_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
            Output("p_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
        ]
        """
        airfoil = cast(Airfoil, _input[Const("airfoil")])
        aoa = cast(float, _input[Const("aoa")])
        rho = cast(float, _input[Const("rho")])
        v_inf = cast(float, _input[Const("v_{inf}")])
        p_inf = cast(float, _input[Const("p_{inf}")])
        delta_star_array = cast(npt.NDArray[np.float64], _input[Var("delta^{*}")])

        pm = PanelMethod(
            airfoil=airfoil,
            sampling_scale_factor=self._sampling_scale_factor
        )
        pm.compute(
            potential_func=self._potential,
            attack_angle=aoa,
            velocity=v_inf,
            pressure=p_inf,
            rho=rho,
            apply_kutta_condition=self._apply_kutta_condition,
            apply_leading_edge_condition=self._apply_leading_edge_condition,
            expand=delta_star_array
        )

        # Compute velocity at panel midpoints
        rotated_airfoil = pm._rotated_airfoil
        if rotated_airfoil is None:
            raise ValueError("Panel method computation failed - no rotated airfoil available")
            
        sampling_offset = self._sampling_scale_factor * rotated_airfoil.chord
        pos = rotated_airfoil.midpoint + delta_star_array.reshape(-1, 1) * rotated_airfoil.norm + sampling_offset * rotated_airfoil.norm
        velocity_vec = pm.velocity(pos)
        
        # Compute tangential velocity component (U_e)
        tangent_vec = rotated_airfoil.tangent
        U_e_array: npt.NDArray[np.float64] = np.sum(velocity_vec * tangent_vec, axis=1)
        
        # For upper surface, take absolute value for physical velocity magnitude
        tail_index = rotated_airfoil.tail_index
        if tail_index < len(U_e_array):
            U_e_array[tail_index:] = np.abs(U_e_array[tail_index:])
        
        invalid_mask = np.isnan(U_e_array)
        if np.any(invalid_mask):
            U_e_array = PotentialSolver._ensure_valid(U_e_array, invalid_mask)

        p_array: npt.NDArray[np.float64] = pm.pressure(pos)
        
        p_invalid_mask = np.isnan(p_array) | np.isinf(p_array)
        if np.any(p_invalid_mask):
            p_array = PotentialSolver._ensure_valid(p_array, p_invalid_mask, default_value=p_inf)

        result: dict[Var[Any], Any] = {
            Var("U_{e}"): U_e_array
        }

        if need_output:
            def delta_upper_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                upper_indices = np.arange(airfoil.tail_index, len(airfoil.tangent_length))
                upper_s = airfoil.tangent_length[-1] - airfoil.tangent_length[upper_indices]
                upper_s_sorted = upper_s[::-1]
                upper_delta_sorted = delta_star_array[upper_indices][::-1]
                return np.interp(x, upper_s_sorted, upper_delta_sorted)
            
            def delta_lower_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                lower_indices = np.arange(0, airfoil.tail_index + 1)
                lower_s = airfoil.tangent_length[lower_indices]
                return np.interp(x, lower_s, delta_star_array[lower_indices])

            def U_e_upper_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                upper_indices = np.arange(airfoil.tail_index, len(airfoil.tangent_length))
                upper_s = airfoil.tangent_length[-1] - airfoil.tangent_length[upper_indices]
                upper_s_sorted = upper_s[::-1]
                upper_U_e_sorted = U_e_array[upper_indices][::-1]
                return np.interp(x, upper_s_sorted, upper_U_e_sorted)
            
            def U_e_lower_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                lower_indices = np.arange(0, airfoil.tail_index + 1)
                lower_s = airfoil.tangent_length[lower_indices]
                return np.interp(x, lower_s, U_e_array[lower_indices])

            def p_upper_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                upper_indices = np.arange(airfoil.tail_index, len(airfoil.tangent_length))
                upper_s = airfoil.tangent_length[-1] - airfoil.tangent_length[upper_indices]
                upper_s_sorted = upper_s[::-1]
                upper_p_sorted = p_array[upper_indices][::-1]
                return np.interp(x, upper_s_sorted, upper_p_sorted)
            
            def p_lower_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                lower_indices = np.arange(0, airfoil.tail_index + 1)
                lower_s = airfoil.tangent_length[lower_indices]
                return np.interp(x, lower_s, p_array[lower_indices])

            result.update({
                Output("delta^{*}_{upper}"): delta_upper_func,
                Output("delta^{*}_{lower}"): delta_lower_func,
                Output("U_{e,upper}"): U_e_upper_func,
                Output("U_{e,lower}"): U_e_lower_func,
                Output("p_{upper}"): p_upper_func,
                Output("p_{lower}"): p_lower_func,
            })
        
        return result
    
    @staticmethod
    def _ensure_valid(
        values: npt.NDArray[np.float64], 
        invalid_mask: npt.NDArray[np.bool_], 
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
        warn(f"Found {np.sum(invalid_mask)} invalid values, replacing with neighbor averages", RuntimeWarning)
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


