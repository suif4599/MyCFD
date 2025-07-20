from tools import Airfoil
from simulator import PanelMethod
from tools import VortexPotential

import numpy as np

def test_airfoil():
    # Test with a simple set of points
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    airfoil = Airfoil(points=points, max_edge_length=5.0)
    
    print("Airfoil Points:")
    print(airfoil.points)

    print("Airfoil Panel:")
    print(airfoil.panel)

    print("Airfoil Length:")
    print(airfoil.length)

    print("Airfoil Norm:")
    print(airfoil.norm)

    print("Airfoil Tangent:")
    print(airfoil.tangent)

    panel_method = PanelMethod(airfoil=airfoil)
    panel_method.compute(
        potential_func=VortexPotential(),
        attack_angle=0.0,
        velocity=1.0,
        pressure=101325.0,
        rho=1.225,
        apply_kutta_condition=True
    )


    