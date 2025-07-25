import numpy as np
import numpy.typing as npt

from tools import Airfoil, VortexPotential
from solver import PotentialSolver, MomentumSolver, Var, Const, Chain, Output
from solver import ConvergenceAnalyzer, StatisticMetric
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
    potential=VortexPotential(),
    sampling_scale_factor=0.02,
    apply_kutta_condition=True,
    apply_leading_edge_condition=True
) | MomentumSolver()

convergence_metrics = [
    StatisticMetric.MEAN(Var("delta^{*}"), "Delta_Star_Mean"),
    StatisticMetric.STD(Var("delta^{*}"), "Delta_Star_Std"),
    StatisticMetric.MAX(Var("delta^{*}"), "Delta_Star_Max"),
    StatisticMetric.MEAN(Var("U_{e}"), "U_e_Mean"),
    StatisticMetric.STD(Var("U_{e}"), "U_e_Std"),
]

analyzer = ConvergenceAnalyzer(convergence_metrics)
chain.set_convergence_analyzer(analyzer)

def p(i: int, self: Chain):
    airfoil = self[Const("airfoil", Airfoil)]
    U_e = self[Var("U_{e}", npt.NDArray[np.float64])]
    delta_star = self[Var("delta^{*}", npt.NDArray[np.float64])]

    U_e[:airfoil.tail_index]
    
    print(f"Iteration {i}:")
    print("U_e_lower:", U_e[:airfoil.tail_index][:5])
    print("U_e_upper:", U_e[airfoil.tail_index:][::-1][:5])
    print("Delta_star_lower:", delta_star[:airfoil.tail_index][:5])
    print("Delta_star_upper:", delta_star[airfoil.tail_index:][::-1][:5])
    return True

chain.solve_iter(
    {
        Const("airfoil", Airfoil): naca_4_digit,
        Const("aoa", float): 0.1,
        Const("rho", float): 1.225,
        Const("v_{inf}", float): 170.0,
        Const("p_{inf}", float): 101325.0,
        Const("nu", float): 1.8e-5,
        Var("delta^{*}", npt.NDArray[np.float64]): np.zeros(len(naca_4_digit.length), dtype=np.float64),
    },
    max_iterations=20,
    # enable_tqdm=False,
    # callback=p,
)
p(-1, chain)

print("\n" + "="*80)
print("Convergence Analysis")
print("="*80)
print(chain.get_convergence_report())

chain.plot_convergence(save_path="convergence_analysis.png", show=True)