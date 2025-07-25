import numpy as np
import numpy.typing as npt

from tools import SourcePotential, Airfoil, VortexPotential
from simulator import PanelMethod
from view import View
from PyQt6.QtWidgets import QApplication
from collections.abc import Callable
from tools.utils import naca_4_digit_f

# TODO: 气流在前缘不能自然分开
# TODO: 势流求解器对于异常数据应该提供插值
# TODO: Var对于泛型的支持不够好

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
    # center_potential=VortexPotential(),
    attack_angle=0.1,
    velocity=170.0,
    pressure=101325.0,
    rho=1.225,
    apply_kutta_condition=True,
    apply_leading_edge_condition=True,
    expand=0.05,
)
print("Force:", panel_method.force)
print("Cl/Cd:", panel_method.coef_lift / panel_method.coef_drag)
app = QApplication([])
view = View(
    field_func=panel_method.velocity,
    width=1000,
    height=1000,
    # x_range=(-2, 4),
    # y_range=(-2, 2),
    x_range=(-0.3, 0.2),
    y_range=(-0.25, 0.25),
    calc_scale=0.5,
    airfoil=panel_method.airfoil
)
view.show()
exit(app.exec())

