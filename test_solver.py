import numpy as np
import numpy.typing as npt

from tools import Airfoil, VortexPotential
from collections.abc import Callable
from solver import PotentialSolver, MomentumSolver, Var, Const, Chain
from tools.utils import naca_4_digit_f

naca_4_digit = Airfoil.from_function(
    func=naca_4_digit_f(
        c=3.0,
        m=0.02,
        p=0.4,
        t=0.12
    ),
    x_range=(0.0, 1.0),
    n_points=100,
    max_edge_length=0.02,
    min_edge_length=0.01
)

chain = PotentialSolver(
    potential=VortexPotential()
) | MomentumSolver()

def p(i: int, self: Chain):
    delta_upper: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = self.get(Var("delta^{*}_{upper}"))
    delta_lower: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = self.get(Var("delta^{*}_{lower}"))
    delta_upper_values = delta_upper(naca_4_digit.tangent_length[-1] - naca_4_digit.tangent_length[naca_4_digit.tail_index:])
    delta_lower_values = delta_lower(naca_4_digit.tangent_length[:naca_4_digit.tail_index])
    # Print the results
    print(f"Iteration {i}:")
    print("Delta Upper Mean:", np.mean(delta_upper_values))
    print("Delta Lower Mean:", np.mean(delta_lower_values))
    print("Delta Upper Std Dev:", np.std(delta_upper_values))
    print("Delta Lower Std Dev:", np.std(delta_lower_values))
    print("Delta Upper Max:", np.max(delta_upper_values))
    print("Delta Lower Max:", np.max(delta_lower_values))
    return True

chain.solve_iter(
    {
        Const("airfoil"): naca_4_digit,
        Const("aoa"): 0.1,
        Const("rho"): 1.225,
        Const("v_{inf}"): 170.0,
        Const("p_{inf}"): 101325.0,
        Const("nu"): 1.8e-5,
        Var("delta^{*}_{upper}"): lambda x: np.zeros_like(x),
        Var("delta^{*}_{lower}"): lambda x: np.zeros_like(x),
    },
    max_iterations=100,
    enable_tqdm=False,
    callback=p,
)
p(-1, chain)