import numpy as np
import numpy.typing as npt

from tools import SourcePotential, Airfoil, VortexPotential
from solver import PotentialSolver, Var, Const, Chain, Output
from view import View
from PyQt6.QtWidgets import QApplication
from collections.abc import Callable
from tools.utils import naca_4_digit_f
from typing import cast


half_cylinder = Airfoil(
    points=np.array(
        [[-np.cos(2 * np.pi * i / 100) + 1, np.sin(2 * np.pi * i / 100)] for i in range(50)],
        dtype=np.float64
    ),
    max_edge_length=0.02,
)
cylinder = Airfoil(
    points=np.array(
        [[-np.cos(2 * np.pi * i / 100) + 1, np.sin(2 * np.pi * i / 100)] for i in range(100)],
        dtype=np.float64
    ),
    max_edge_length=0.02,
)
triangle = Airfoil(
    points=np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ],
        dtype=np.float64
    ),
    max_edge_length=2
)


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
    sampling_scale_factor=0.05,
    apply_kutta_condition=True,
    apply_leading_edge_condition=True
)

result = chain.solve(
    {
        Const("airfoil"): naca_4_digit,
        Const("aoa"): 0.1,
        Const("rho"): 1.225,
        Const("v_{inf}"): 170.0,
        Const("p_{inf}"): 101325.0,
        Const("nu"): 1.8e-5,
        Var("delta^{*}"): np.zeros(len(naca_4_digit.length), dtype=np.float64),
    },
    need_output=True
)

U_e = cast(npt.NDArray[np.float64], result.get(Var("U_{e}")))

print("U_e_lower:", U_e[:naca_4_digit.tail_index][:10])
print("U_e_upper:", U_e[naca_4_digit.tail_index:][::-1][:10])

tangent_lengths = naca_4_digit.tangent_length
tail_index = naca_4_digit.tail_index
panel_midpoint_lengths = np.zeros(len(naca_4_digit.length))
panel_midpoint_lengths[0] = naca_4_digit.length[0] / 2
for i in range(1, len(naca_4_digit.length)):
    panel_midpoint_lengths[i] = tangent_lengths[i-1] + naca_4_digit.length[i] / 2
s_lower = panel_midpoint_lengths[:tail_index]
s_upper = panel_midpoint_lengths[tail_index:]
total_length = tangent_lengths[-1]
lower_surface_length = tangent_lengths[tail_index-1] if tail_index > 0 else 0
upper_surface_length = total_length - lower_surface_length
s_upper_from_le = upper_surface_length - (s_upper - lower_surface_length)

# draw U_e by matplotlib using arc length distance
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(s_lower, U_e[:tail_index], label='Lower Surface', marker='o', markersize=3, linewidth=2)
plt.plot(s_upper_from_le, U_e[tail_index:], label='Upper Surface', marker='x', markersize=3, linewidth=2)
plt.title('Velocity Distribution on NACA 4-Digit Airfoil')
plt.xlabel('Distance along surface from Leading Edge (s)')
plt.ylabel('Edge Velocity (U_e)')
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
