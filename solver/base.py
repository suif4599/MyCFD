"""
This module implements a singleton pattern for variables with the same name, 
ensuring consistent typing and class definitions across instances.

Key behaviors:
1. Reusing existing instance:
   - When creating `Var("x", SomeType)` followed by `Var("x")`, 
     the existing instance is reused (type inferred as `SomeType`)
   - Explicit type must match if provided: `Var("x", SomeType)` 
     returns existing instance when types match

2. Type conflict prevention:
   - `Var("x", AnotherType)` raises `TypeError` if existing instance 
     has different type

3. Class conflict prevention:
   - Creating different classes with same name (e.g., `Const("x", SomeType)` 
     when `Var("x", SomeType)` exists) raises `TypeError`
   - Applies to all subclasses (Const, Output, etc.)

Generic Type Considerations:
- Type inference works for concrete types but complex generics may require
  explicit casting due to limitations in type inference systems
- When reusing existing variables without explicit type, the type parameter 
  may not be inferred correctly and may require casting

Implementation notes:
- All variable instances are tracked in `Var._have_allocated`
- Matching considers: (name.strip(), type, class)
- Type can be either explicit type or instance value type

Example scenarios:
1. Valid reuse with type inference:
   a = Var("x", SomeType)   # Creates new instance
   b = Var("x")             # Reuses a (type inferred as SomeType)

2. Type conflict:
   Var("x", SomeType)
   Var("x", AnotherType)    # Raises TypeError

3. Class conflict:
   Var("x", SomeType)
   Const("x", SomeType)     # Raises TypeError

4. Complex generic handling:
   # May require explicit casting for full type information
   c = cast(Var[ComplexType], Var("x", ComplexType))
"""

from typing import Any, cast, TypeVar, Generic

T = TypeVar('T')

class VarMeta(type):
    def __call__(
        cls,
        name: str,
        _type: type[T] | T | None = None
    ) -> "Var[T]":
        key = (name.strip(), _type, cls)

        for obj in Var._have_allocated:
            if (obj.name, obj.type, obj.__class__) == key:
                return obj
            if obj.name == name.strip():
                if _type is None:
                    _type = obj.type
                    if obj.__class__ == cls:
                        return obj
                if obj.type != _type:
                    raise TypeError(
                        f"Variable '{obj}' already exists with different value type:"
                        f"\n\tFailed to create {cls.__name__} with name '{name}' and type '{_type}'"
                    )
                if obj.__class__ != cls:
                    raise TypeError(
                        f"Variable '{obj}' already exists with different class:"
                        f"\n\tFailed to create {cls.__name__} with name '{name}' and type '{_type}'"
                    )

        raw: "Var" = super().__call__(name, _type or Any)
        raw._type = _type
        Var._have_allocated.append(raw)
        return raw

class Var(Generic[T], metaclass=VarMeta):
    """
    Class representing a variable in a solver.
    Generic type T represents the type of data this variable holds.
    """
    _name: str
    _type: T
    _have_allocated: list["Var"] = []

    def __init__(self, name: str, _type: type[T] | T | None = None) -> None:
        self._name = name.strip()
        self._type = cast(T, _type or Any)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.type})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Var):
            return NotImplemented
        return self._name == other._name and self.__class__ == other.__class__
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def type(self) -> T:
        return self._type

class ConstMeta(VarMeta):
    def __call__(cls, name: str, _type: type[T] | T | None = None) -> "Const[T]":
        return cast(Const[T], super().__call__(name, _type))

class Const(Var[T], metaclass=ConstMeta):
    """
    Class representing a constant variable in a solver.
    Generic type T represents the type of constant data this variable holds.
    """

class OutputMeta(VarMeta):
    def __call__(cls, name: str, _type: type[T] | T | None = None) -> "Output[T]":
        return cast(Output[T], super().__call__(name, _type))

class Output(Var[T], metaclass=OutputMeta):
    """
    Class representing an output variable in a solver.
    Generic type T represents the type of output data this variable holds.
    """
