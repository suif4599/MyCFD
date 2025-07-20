from view.airfoil_visualizer import AirfoilVisualizer
from tools.airfoil import Airfoil
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
naca_4_digit.inplace_rotate(-0.2)
viz = AirfoilVisualizer(figsize=(12, 8))
viz.add_airfoil(
    naca_4_digit,
    label="NACA 4-Digit Airfoil",
    color="blue"
)
viz.add_airfoil(
    naca_4_digit.expand((lambda x: x / 10, lambda x: x / 10)),
    label="Expanded(func) NACA 4-Digit Airfoil",
    color="red"
)
viz.add_airfoil(
    naca_4_digit.expand(0.05 * naca_4_digit.chord),
    label="Expanded(0.05 * chord) NACA 4-Digit Airfoil",
    color="green"
)
viz.plot()