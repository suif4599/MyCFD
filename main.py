import numpy as np
import numpy.typing as npt

from tools import SourcePotential, Airfoil, VortexPotential
from simulator import PanelMethod
from view import View
from PyQt5.QtWidgets import QApplication
from collections.abc import Callable
from tools.utils import naca_4_digit_f

####### Airfoil.expand 还不能很好地处理尾部的情况，需要修改

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

panel_method = PanelMethod(
    airfoil=naca_4_digit
)
panel_method.compute(
    potential_func=VortexPotential(),
    attack_angle=0.1,
    velocity=170.0,
    pressure=101325.0,
    rho=1.225,
    apply_kutta_condition=True
)
print("Force:", panel_method.force)
print("Cl/Cd:", panel_method.coef_lift / panel_method.coef_drag)
app = QApplication([])
view = View(
    field_func=panel_method.velocity,
    width=2000,
    height=1600,
    x_range=(-4, 4),
    y_range=(-4, 4),
    calc_scale=0.25,
    airfoil=panel_method.airfoil
)
view.show()
exit(app.exec_())


# from solver import PotentialSolver, Var
# solver = PotentialSolver(VortexPotential())
# ret = solver.solve(
#     {
#         Var("airfoil", Airfoil): naca_4_digit,
#         Var("aoa", float): 0.2,
#         Var("rho", float): 1.225,
#         Var("v_{inf}", float): 170.0,
#         Var("p_{inf}", float): 101325.0,
#         Var("delta", float): 0.0,
#     }
# )

