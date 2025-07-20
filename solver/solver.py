from collections.abc import Sequence, Callable
from abc import abstractmethod, ABCMeta
from typing import Any

from .base import Var, Const, Output


class Chain:
    def __init__(self, *solvers: "Solver"):
        self._solvers = solvers
        self._values: dict[Var, Any] = {}

    def __repr__(self) -> str:
        return f"Chain({', '.join(repr(solver) for solver in self._solvers)})"
    
    def _check_valid(self) -> None:
        """
        Check if the solver chain is valid
        """
        # Closure check: ensure all required variables have corresponding outputs
        required_vars: set[Var] = set()
        for solver in self._solvers:
            for var in solver.input:
                if isinstance(var, Var) and not isinstance(var, (Const, Output)):
                    required_vars.add(var)

        available_outputs: set[Var] = set()
        for solver in self._solvers:
            for var in solver.output:
                if isinstance(var, Var):
                    available_outputs.add(var)
        
        uncovered_vars = required_vars - available_outputs
        if uncovered_vars:
            raise ValueError(f"Solver chain is not closed, missing outputs for variables: {list(uncovered_vars)}")
        
        # Initial condition check: ensure all required variables can be computed
        available_vars: set[Var] = set(self._values.keys())
        max_simulation_iterations = len(self._solvers) * 2
        for _ in range(max_simulation_iterations):
            new_vars_added = False
            for solver in self._solvers:
                can_run = all(var in available_vars for var in solver.input)
                if can_run:
                    for var in solver.output:
                        if var not in available_vars:
                            available_vars.add(var)
                            new_vars_added = True
            if not new_vars_added:
                break
        
        all_required_vars: set[Var] = set()
        for solver in self._solvers:
            for var in solver.input:
                if isinstance(var, Var) and not isinstance(var, Output):
                    all_required_vars.add(var)
        
        missing_vars = all_required_vars - available_vars
        if missing_vars:
            raise ValueError(f"Solver chain cannot be solved, missing initial conditions: {list(missing_vars)}")

    def solve_iter(
        self,
        _input: dict[Var, Any],
        max_iterations: int = 100,
        enable_tqdm: bool = True,
        callback: Callable[[int, "Chain"], bool] | None = None 
    ) -> None:
        """
        Solve the solver chain iteratively for a fixed number of iterations.
        
        @param _input: Initial input values for the solver chain.
        @param max_iterations: Maximum number of iterations to run.
        @param enable_tqdm: Whether to use tqdm for progress display.
        @param callback: Optional callback function to be called after each iteration,
            which accepts a number and a Chain object and returns True to continue, False to stop.
        """
        self._values = _input.copy()
        self._check_valid()
        
        if enable_tqdm:
            from tqdm import tqdm
            iterations = tqdm(range(1, max_iterations + 1), desc="Solving Chain")
        else:
            iterations = range(1, max_iterations + 1)
        for i in iterations:
            for solver in self._solvers:
                solver_input: dict[Var, Any] = {}
                for var in solver.input:
                    if var in self._values:
                        solver_input[var] = self._values[var]
                solver_output = solver.solve(solver_input)
                self._values.update(solver_output)
            if callback:
                if not callback(i, self):
                    break
    
    def __or__(self, value: Any) -> "Chain":
        if not isinstance(value, Chain):
            raise TypeError("Can only combine with another Chain instance.")
        return Chain(*(self._solvers + value._solvers))
    
    def __getitem__(self, key: Var) -> Any:
        if key not in self._values:
            raise KeyError(f"Variable {key} not found in solver chain values.")
        return self._values[key]
    
    def get(self, key: Var, default: Any = None) -> Any:
        return self._values.get(key, default)
    
    def keys(self):
        return self._values.keys()
    
    def values(self):
        return self._values.values()
    
    def items(self):
        return self._values.items()


class Solver(Chain, metaclass=ABCMeta):
    """
    Base class for a solver that takes input variables and produces output variables.
    """
    def __init__(self, _input: Sequence[Var], _output: Sequence[Var]):
        # Check that inputs don't contain Output types
        for var in _input:
            if isinstance(var, Output):
                raise ValueError(f"Solver input cannot contain Output type: {var}")
        
        # Check that outputs don't contain Const types
        for var in _output:
            if isinstance(var, Const):
                raise ValueError(f"Solver output cannot contain Const type: {var}")
        
        self._input = list(_input)
        self._output = list(_output)
        super().__init__(self)

    @abstractmethod
    def solve(self, _input: dict[Var, Any]) -> dict[Var, Any]:
        """
        Solve one step of the equations based on the input variables.
        This method should be implemented by subclasses.
        """
        pass

    @property
    def input(self) -> Sequence[Var]:
        return self._input
    
    @property
    def output(self) -> Sequence[Var]:
        return self._output

