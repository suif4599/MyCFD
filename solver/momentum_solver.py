from typing import Any
from collections.abc import Callable

from .base import Var, Const, Output
from .solver import Solver
from simulator import MomentumIntegralMethod
from tools import Potential, Airfoil
import numpy as np
import numpy.typing as npt


class MomentumSolver(Solver):
    def __init__(self):
        super().__init__(
            _input=[
                Const("airfoil", Airfoil),
                Const("aoa", float),
                Const("nu", float),
                Var("U_{e,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Var("U_{e,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
            ],
            _output=[
                Var("delta^{*}_{upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Var("delta^{*}_{lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("c_{f,upper}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]),
                Output("c_{f,lower}", Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]])
            ]
        )
    
    def solve(self, _input: dict[Var, Any]) -> dict[Var, Any]:
        airfoil: Airfoil = _input[Const("airfoil")]
        aoa: float = _input[Const("aoa")]
        nu: float = _input[Const("nu")]
        U_e_upper: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = _input[Var("U_{e,upper}")]
        U_e_lower: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = _input[Var("U_{e,lower}")]
        airfoil.inplace_rotate(-aoa)

        mi = MomentumIntegralMethod(
            airfoil=airfoil
        )
        mi.compute(
            nu=nu,
            U_e_upper=U_e_upper,
            U_e_lower=U_e_lower
        )
        
        delta_upper = mi.delta_upper
        delta_lower = mi.delta_lower
        c_f_upper = mi.cf_upper
        c_f_lower = mi.cf_lower
        return {
            Var("delta^{*}_{upper}"): delta_upper,
            Var("delta^{*}_{lower}"): delta_lower,
            Output("c_{f,upper}"): c_f_upper,
            Output("c_{f,lower}"): c_f_lower
        }