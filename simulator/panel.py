import numpy as np
import numpy.typing as npt

from tools import Airfoil, Potential
from typing import overload, cast
from functools import singledispatchmethod
from tools.reshape import reshape2_method
from warnings import warn

class PanelMethod:
    """
    Class for the panel method to compute potential flow around an airfoil.\n

    Sources are on vertices and equations are on midpoints of segments.\n
    """

    _airfoil: Airfoil
    _rotated_airfoil: Airfoil | None = None # Airfoil rotated by the attack angle
    _potential_func: Potential | None = None # Potential function to use for the panel method
    _q: npt.NDArray[np.float64] | None = None # Source strengths for each panel
    _attack_angle: float # Angle of attack
    _velocity: float # Freestream velocity
    _pressure: float # Freestream pressure
    _rho: float # Freestream density
    _force: tuple[float, float] | None = None # Total force on the airfoil

    def __init__(
        self,
        airfoil: Airfoil
    ):
        self._airfoil = airfoil

    def compute(
        self,
        potential_func: Potential,
        velocity: float,
        pressure: float,
        rho: float,
        attack_angle: float = 0.0,
        apply_kutta_condition: bool = False,
    ) -> None:
        """
        Calculates the potential flow around the airfoil using the panel method.
        
        @param potential_func: Potential function to use
        @param attack_angle: Angle of attack in radians
        @param velocity: Freestream velocity magnitude
        @param apply_kutta_condition: Whether to apply Kutta condition for realistic flow
        """
        self._rotated_airfoil = self._airfoil.rotate(-attack_angle)
        self._potential_func = potential_func
        self._attack_angle = attack_angle
        self._velocity = velocity
        self._pressure = pressure
        self._rho = rho
        self._force = None
        airfoil = self._rotated_airfoil
        n = airfoil.points.shape[0]
        
        vector = airfoil.midpoint.reshape(-1, 1, 2) - airfoil.points.reshape(1, -1, 2)

        # coef = (
        #     airfoil.norm.reshape(-1, 1, 2) @ potential_func.far_velocity(vector).swapaxes(-1, -2)
        # ).reshape(-1, n)

        coef = (
            airfoil.norm.reshape(-1, 1, 2) @ \
                potential_func.velocity(
                    vector, airfoil.panel.reshape(1, -1, 2)
                ).swapaxes(-1, -2)
        ).reshape(-1, n)

        v_norm = cast(npt.NDArray[np.float64], (
            -np.array([velocity, 0.0], dtype=np.float64).reshape(1, 2) @ airfoil.norm.T
        ).reshape(-1, 1))

        if apply_kutta_condition:
            tail_idx = self.airfoil.tail_index
            if tail_idx == n - 1 or tail_idx == 0:
                raise ValueError(
                    "Kutta condition requires a valid trailing edge point, "
                    "but the tail index is at the first or last point."
                )
            kutta_row = np.zeros(n)
            kutta_row[tail_idx - 1: tail_idx + 1] = 1.0
            # coef[tail_idx, :] = kutta_row
            # v_norm[tail_idx] = 0.0
            # coef[tail_idx - 1, :] = kutta_row
            # v_norm[tail_idx - 1] = 0.0
            coef[-1, :] = kutta_row
            v_norm[-1] = 0.0
        
        try:
            condition_number = np.linalg.cond(coef)
            if condition_number < 1e10:
                self._q = np.asarray(np.linalg.solve(coef, v_norm), dtype=np.float64).reshape(-1)
            else:
                # Use SVD-based pseudoinverse
                warn(f"Warning: Ill-conditioned system (cond={condition_number:.2e}), using SVD")
                U, s, Vt = np.linalg.svd(coef, full_matrices=False)
                
                threshold = 1e-12 * s[0] if len(s) > 0 else 1e-12
                s_inv = np.where(s > threshold, 1.0/s, 0.0)
                coef_pinv = Vt.T @ np.diag(s_inv) @ U.T
                self._q = (coef_pinv @ v_norm).reshape(-1)
        except np.linalg.LinAlgError:
            warn("Warning: LinAlgError encountered, using regularized least squares")
            regularization_param = 1e-8
            A_reg = coef.T @ coef + regularization_param * np.eye(n)
            self._q = np.asarray(np.linalg.solve(A_reg, coef.T @ v_norm), dtype=np.float64).reshape(-1)
    
    @singledispatchmethod
    def _velocity_impl(self, *args, **kwargs):
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(np.ndarray)
    @reshape2_method
    def _(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self._q is not None
        assert self._potential_func is not None
        assert self._rotated_airfoil is not None
        vel_contributions = self._potential_func.velocity(
            points.reshape(-1, 1, 2) - self.airfoil.points.reshape(1, -1, 2),
            self.airfoil.panel.reshape(1, -1, 2)
        )
        panel_velocity = (vel_contributions * self._q.reshape(1, -1, 1)).sum(axis=1)
        n_points = points.shape[0] if points.ndim == 2 else points.reshape(-1, 2).shape[0]
        inflow_velocity = np.array([
            self._velocity, 0.0
        ], dtype=np.float64)
        inflow_velocity_array = np.tile(inflow_velocity, (n_points, 1))
        return panel_velocity + inflow_velocity_array
    
    @_velocity_impl.register(tuple)
    def _(self, point: tuple[float, float]) -> tuple[float, float]:
        assert self._q is not None
        assert self._potential_func is not None
        assert self._rotated_airfoil is not None
        vel_contributions = self._potential_func.velocity(
            np.array(point).reshape(1, 2) - self.airfoil.points,
            self.airfoil.panel
        )
        panel_velocity = (vel_contributions * self._q.reshape(-1, 1)).sum(axis=0)
        inflow_velocity = np.array([
            self._velocity, 0.0
        ])
        total_velocity = panel_velocity + inflow_velocity
        return total_velocity[0], total_velocity[1]
    
    @_velocity_impl.register(list)
    def _(self, point: list[float]) -> list[float]:
        result = self._velocity_impl(tuple(point))
        return [result[0], result[1]]
    
    @_velocity_impl.register(float)
    def _(self, x: float, y: float) -> tuple[float, float]:
        return self._velocity_impl((x, y))
    
    @overload
    def velocity(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    @overload
    def velocity(self, point: tuple[float, float]) -> tuple[float, float]: ...

    @overload
    def velocity(self, point: list[float]) -> list[float]: ...

    @overload
    def velocity(self, x: float, y: float) -> tuple[float, float]: ...

    def velocity(self, *args, **kwargs) -> npt.NDArray[np.float64] | tuple[float, float] | list[float]:
        """
        Returns the velocity of the airfoil.
        """
        if self._q is None or self._potential_func is None:
            raise ValueError("Panel method has not been computed yet.")
        return self._velocity_impl(*args, **kwargs)
    
    @singledispatchmethod
    def _pressure_impl(self, *args, **kwargs):
        raise NotImplementedError("Unsupported type for pressure calculation.")
    
    @_pressure_impl.register(np.ndarray)
    @reshape2_method
    def _(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        velocity = self.velocity(points)
        return self._pressure - self._rho * (
            velocity[:, 0]**2 + velocity[:, 1]**2 - self._velocity**2
        ) / 2.0
    
    @_pressure_impl.register(tuple)
    def _(self, point: tuple[float, float]) -> float:
        velocity = self.velocity(point)
        return self._pressure - self._rho * (
            velocity[0]**2 + velocity[1]**2 - self._velocity**2
        ) / 2.0
    
    @_pressure_impl.register(list)
    def _(self, point: list[float]) -> float:
        return self._pressure_impl(
            tuple(point)
        )
    
    @_pressure_impl.register(float)
    def _(self, x: float, y: float) -> float:
        return self._pressure_impl(
            (x, y)
        )
    
    @overload
    def pressure(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    @overload
    def pressure(self, point: tuple[float, float]) -> float: ...

    @overload
    def pressure(self, point: list[float]) -> float: ...

    @overload
    def pressure(self, x: float, y: float) -> float: ...

    def pressure(self, *args, **kwargs) -> npt.NDArray[np.float64] | float:
        """
        Returns the pressure at the given points or point.
        """
        if self._q is None or self._potential_func is None:
            raise ValueError("Panel method has not been computed yet.")
        return self._pressure_impl(*args, **kwargs)

    @property
    def airfoil(self) -> Airfoil:
        if self._rotated_airfoil is None:
            return self._airfoil
        return self._rotated_airfoil
    
    @property
    def force(self) -> tuple[float, float]:
        """
        Returns the lift force on the airfoil.
        """
        if self._force is None:
            if self._q is None or self._potential_func is None:
                raise ValueError("Panel method has not been computed yet.")
            expanded_airfoil = self.airfoil.expand(0.05 * self.airfoil.chord)
            force = self.pressure(expanded_airfoil.midpoint) * expanded_airfoil.length
            self._force = tuple((force.reshape(1, -1) @ expanded_airfoil.norm).tolist()[0])
            self._force = -self._force[0], -self._force[1]
        return self._force
    
    @property
    def lift(self) -> float:
        """
        Returns the lift force on the airfoil.
        """
        return self.force[1]
    
    @property
    def drag(self) -> float:
        """
        Returns the drag force on the airfoil.
        """
        return self.force[0]
    
    @property
    def coef_lift(self) -> float:
        """
        Returns the coefficient of lift.
        """
        return self.lift / (0.5 * self._rho * self._velocity**2 * self.airfoil.chord)
    
    @property
    def coef_drag(self) -> float:
        """
        Returns the coefficient of drag.
        """
        return self.drag / (0.5 * self._rho * self._velocity**2 * self.airfoil.chord)
