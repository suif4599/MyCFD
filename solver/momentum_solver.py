from typing import Any, cast
from collections.abc import Callable

from .base import Var, Const, Output
from .solver import Solver
from simulator import MomentumIntegralMethod
from tools import Airfoil
import numpy as np
import numpy.typing as npt
from typing import TypeVar

T = TypeVar('T')

class MomentumSolver(Solver):
    def __init__(self):
        super().__init__(
            _input=[
                Const("airfoil", Airfoil),
                Const("aoa", float),
                Const("nu", float),
                Var("U_{e}", npt.NDArray[np.float64])
            ],
            _output=[
                Var("delta^{*}", npt.NDArray[np.float64]),
                Output("delta^{*}_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("delta^{*}_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("U_{e,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("U_{e,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("c_{f,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("c_{f,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
            ]
        )
    
    def solve(
        self,
        _input: dict[Var[T], T],
        need_output: bool
    ) -> dict[Var, Any]:
        airfoil = cast(Airfoil, _input[Const("airfoil")])
        aoa = cast(float, _input[Const("aoa")])
        nu = cast(float, _input[Const("nu")])
        U_e_array = cast(npt.NDArray[np.float64], _input[Var("U_{e}")])

        airfoil_rotated = airfoil.rotate(-aoa)

        mi = MomentumIntegralMethod(
            airfoil=airfoil_rotated
        )
        mi.compute(
            nu=nu,
            U_e_array=U_e_array
        )
        
        delta_star_upper_func = mi.delta_star_upper
        delta_star_lower_func = mi.delta_star_lower
        
        def U_e_upper_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            tail_ind = airfoil_rotated.tail_index
            upper_U_e = U_e_array[tail_ind:]
            upper_s_coords = np.zeros(len(upper_U_e) + 1)
            for i in range(len(upper_U_e)):
                panel_idx = tail_ind + i
                if panel_idx < len(airfoil_rotated.length):
                    upper_s_coords[i + 1] = upper_s_coords[i] + airfoil_rotated.length[panel_idx]
            upper_midpoint_s = (upper_s_coords[:-1] + upper_s_coords[1:]) / 2
            return np.interp(x, upper_midpoint_s, upper_U_e)
        
        def U_e_lower_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            tail_ind = airfoil_rotated.tail_index
            lower_U_e = U_e_array[:tail_ind]
            lower_s_coords = np.zeros(tail_ind + 1)
            for i in range(tail_ind):
                lower_s_coords[i + 1] = lower_s_coords[i] + airfoil_rotated.length[i]
            lower_midpoint_s = (lower_s_coords[:-1] + lower_s_coords[1:]) / 2
            return np.interp(x, lower_midpoint_s, lower_U_e)

        n = len(airfoil_rotated.length)
        delta_star_array = np.zeros(n, dtype=np.float64)
        
        tail_ind = airfoil_rotated.tail_index
        lower_indices = np.arange(0, tail_ind)
        lower_s_coords = np.zeros(tail_ind + 1)
        for i in range(tail_ind):
            lower_s_coords[i + 1] = lower_s_coords[i] + airfoil_rotated.length[i]
        lower_midpoint_s = (lower_s_coords[:-1] + lower_s_coords[1:]) / 2
        delta_star_array[lower_indices] = delta_star_lower_func(lower_midpoint_s)
        
        upper_indices = np.arange(tail_ind, n)
        upper_s_coords = np.zeros(len(upper_indices) + 1)
        for i, panel_idx in enumerate(upper_indices):
            if panel_idx < len(airfoil_rotated.length):
                upper_s_coords[i + 1] = upper_s_coords[i] + airfoil_rotated.length[panel_idx]
        upper_midpoint_s = (upper_s_coords[:-1] + upper_s_coords[1:]) / 2
        delta_star_array[upper_indices] = delta_star_upper_func(upper_midpoint_s)

        result: dict[Var[Any], Any] = {
            Var("delta^{*}"): delta_star_array
        }

        if need_output:
            c_f_upper = mi.cf_upper
            c_f_lower = mi.cf_lower
            
            result.update({
                Output("delta^{*}_{upper}"): delta_star_upper_func,
                Output("delta^{*}_{lower}"): delta_star_lower_func,
                Output("U_{e,upper}"): U_e_upper_func,
                Output("U_{e,lower}"): U_e_lower_func,
                Output("c_{f,upper}"): c_f_upper,
                Output("c_{f,lower}"): c_f_lower
            })
        
        return result