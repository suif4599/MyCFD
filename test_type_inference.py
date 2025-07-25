from solver.base import Var, Const, Output
from collections.abc import Callable
from typing import cast

print("\n" f"{'Auto Inference Example':=^50}" "\n")

x = Var("x", list[int]) # Var[list[int]] is inferred
y = Const("y", dict[float, float]) # Const[dict[float, float]] is inferred
z = cast(
    Output[Callable[[list[int]], float]],
    Output("z", Callable[[list[int]], float])
) # Auto Inference gets Output[Callable] but not the full type in vscode 1.102.1 with pylance latest version


print(x) # Var(x, list[int])
print(y) # Const(y, dict[float, float])
print(z) # Output(z, collections.abc.Callable[[list[int]], float])


print("\n" f"{'Singleton Example':=^50}" "\n")

x2 = Var("x", list[int])

try:
    y2 = Output("y", dict[float, float])
except TypeError as e:
    y2 = Const("y", dict[float, float])
    print(f"TypeError: {e}")

try:
    z2 = Output("z", Callable[[list[float]], float])
except TypeError as e:
    z2 = cast(
        Output[Callable[[list[int]], float]],
        Output("z", Callable[[list[int]], float])
    )
    print(f"TypeError: {e}")

print(x2 is x and y2 is y and z2 is z)  # True


print("\n" f"{'Auto Inference with existing Var':=^50}" "\n")

x3 = cast(
    Var[int],
    Var("x")
) # without _type, Var[int] cannot be auto-inferred

try:
    y3 = Output("y")
except TypeError as e:
    y3 = cast(
        Const[dict[float, float]],
        Const("y")
    )
    print(f"TypeError: {e}")

print(x3)  # Var(x, int)
print(y3)  # Const(y, dict[float, float])
