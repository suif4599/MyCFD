from typing import Any, TypeAlias


class VarMeta(type):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raw: "Var" = super().__call__(*args, **kwds)
        if raw in Var._have_allocated:
            return Var._have_allocated[Var._have_allocated.index(raw)]
        Var._have_allocated.append(raw)
        return raw

class Var(metaclass=VarMeta):
    """
    Class representing a variable in a solver.
    """
    _name: str
    _have_allocated: list["Var"] = []

    def __init__(self, name: str, _type: TypeAlias | None = None):
        self._name = name.strip()

    def __str__(self) -> str:
        return f"Var({self._name})"
    
    def __repr__(self) -> str:
        return f"Var({self._name})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Var):
            return NotImplemented
        return self._name == other._name
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    @property
    def name(self) -> str:
        return self._name

class Const(Var):
    """
    Class representing a constant variable in a solver.
    """
    def __repr__(self) -> str:
        return f"Const({self._name})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Const):
            return NotImplemented
        if isinstance(other, Var) and not isinstance(other, Const):
            return False
        return self._name == other._name
    
    def __hash__(self) -> int:
        return super().__hash__()

class Output(Var):
    """
    Class representing an output variable in a solver.
    """
    def __repr__(self) -> str:
        return f"Output({self._name})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Output):
            return NotImplemented
        if isinstance(other, Var) and not isinstance(other, Output):
            return False
        return self._name == other._name
    
    def __hash__(self) -> int:
        return super().__hash__()

