import numpy as np

from tools import VortexPotential

def test_vortex_potential():
    vortex = VortexPotential()

    panel = (0.1, 0.05)
    point = (3, 4)

    velocity = vortex.velocity(point, panel)
    far_velocity = vortex.far_velocity(point)

    print(f"Velocity: {velocity}")
    print(f"Far Field Velocity: {far_velocity}")