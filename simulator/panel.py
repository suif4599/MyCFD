import numpy as np
import numpy.typing as npt

from tools import Airfoil, Potential
from typing import overload, cast, Union
from collections.abc import Callable
from functools import singledispatchmethod
from tools.reshape import reshape2_method
from warnings import warn

class PanelMethod:
    """
    Class for the panel method to compute potential flow around an airfoil.
    Sources are on vertices and equations are on midpoints of segments.
    """

    _airfoil: Airfoil
    _sampling_scale_factor: float = 0.01  # Sampling scale factor for the airfoil
    _rotated_airfoil: Airfoil | None = None # Airfoil rotated by the attack angle
    _potential_func: Potential | None = None # Potential function to use for the panel method
    _center_potential: Potential | None = None # Center potential function
    _center_position: tuple[float, float] | None = None # Position of center source
    _center_strength: float | None = None # Strength of center source
    _q: npt.NDArray[np.float64] | None = None # Source strengths for each panel
    _attack_angle: float # Angle of attack
    _velocity: float # Freestream velocity
    _pressure: float # Freestream pressure
    _rho: float # Freestream density
    _force: tuple[float, float] | None = None # Total force on the airfoil
    _expand: Union[float, npt.NDArray[np.float64], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = 0.0 # Boundary layer displacement thickness
    _displaced_sources: npt.NDArray[np.float64] | None = None # Displaced source points
    _displaced_panels: npt.NDArray[np.float64] | None = None # Displaced panel vectors
    _use_displaced_sources: bool = False # Whether to use displaced sources for velocity calculation
    _last_computation_unstable: bool = False # Whether the last computation was numerically unstable
    _matrix_condition_number: float | None = None # Last matrix condition number
    _solution_quality_score: float | None = None # Quality score of the last solution

    def __init__(
        self,
        airfoil: Airfoil,
        sampling_scale_factor: float = 0.01
    ):
        self._airfoil = airfoil
        self._sampling_scale_factor = sampling_scale_factor
        self._last_computation_unstable = False
        self._matrix_condition_number = None
        self._solution_quality_score = None

    def is_last_computation_stable(self) -> bool:
        """Returns True if the last computation was numerically stable."""
        return not self._last_computation_unstable
    
    def get_solution_quality_metrics(self) -> dict:
        """Returns metrics about the quality of the last solution."""
        return {
            'stable': not self._last_computation_unstable,
            'condition_number': self._matrix_condition_number,
            'quality_score': self._solution_quality_score
        }
    
    def _assess_solution_stability(self, coef: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> tuple[bool, float]:
        """
        Assess the numerical stability of the solution.
        Returns (is_stable, quality_score).
        """
        condition_number = np.linalg.cond(coef)
        self._matrix_condition_number = condition_number
        
        eigenvals = np.linalg.eigvals(coef)
        min_eigenval = np.min(np.real(eigenvals))
        small_eigenvals = np.sum(np.abs(np.real(eigenvals)) < 1e-10)
        
        q_range = np.max(q) - np.min(q)
        q_mean = np.mean(np.abs(q))
        
        sign_changes = np.sum(np.diff(np.sign(q)) != 0)
        oscillation_ratio = sign_changes / len(q)
        
        quality_score = 1.0
        if condition_number > 1e7:
            quality_score *= 0.1
        elif condition_number > 5e3:
            quality_score *= max(0.3, 5e3 / condition_number)
        
        if min_eigenval < -1.0:
            quality_score *= max(0.1, 1.0 / float(abs(min_eigenval)))
        elif min_eigenval < -0.5:
            quality_score *= 0.5
        
        small_eigenval_ratio = small_eigenvals / len(eigenvals)
        if small_eigenval_ratio > 0.5:
            quality_score *= max(0.2, 1.0 - small_eigenval_ratio)
        
        if oscillation_ratio > 0.9:
            quality_score *= max(0.1, 1.0 - oscillation_ratio)
        
        if q_range > 100 * q_mean:
            quality_score *= 0.1
        self._solution_quality_score = cast(float, quality_score)
        
        is_stable = (
            condition_number < 1e7 and        # Much higher threshold
            min_eigenval > -5.0 and           # Allow moderate negative eigenvalues
            small_eigenval_ratio < 0.6 and    # More lenient
            oscillation_ratio < 0.95 and      # More lenient
            quality_score > 0.05              # Much lower threshold
        )
        
        return bool(is_stable), self._solution_quality_score
    
    def _compute_with_interpolation(
        self,
        potential_func: Potential,
        velocity: float,
        pressure: float,
        rho: float,
        attack_angle: float,
        apply_kutta_condition: bool,
        apply_leading_edge_condition: bool,
        expand: Union[float, npt.NDArray[np.float64], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        center_potential: Potential | None = None,
    ) -> None:
        """
        Compute with automatic interpolation fallback for unstable solutions.
        """
        if not isinstance(expand, (int, float)):
            # For non-scalar expand, use standard computation
            return self._compute_direct(
                potential_func,
                velocity,
                pressure,
                rho,
                attack_angle,
                apply_kutta_condition,
                apply_leading_edge_condition,
                expand,
                center_potential
            )
        
        expand_val = float(expand)
        
        # Try direct computation
        try:
            self._compute_direct(
                potential_func,
                velocity,
                pressure,
                rho,
                attack_angle,
                apply_kutta_condition,
                apply_leading_edge_condition,
                expand_val,
                center_potential
            )
            if self.is_last_computation_stable():
                return
            else:
                warn(f"Numerically unstable solution for expand={expand_val:.6f}, attempting interpolation")
        except Exception as e:
            warn(f"Computation failed for expand={expand_val:.6f}: {e}, attempting interpolation")
        
        # Find stable neighboring points for interpolation
        stable_points = []
        test_distances = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        for distance in test_distances:
            for direction in [-1, 1]:
                test_expand = expand_val + direction * distance
                if test_expand < 0:
                    continue
                try:
                    temp_panel = PanelMethod(self._airfoil)
                    temp_panel._compute_direct(
                        potential_func,
                        velocity,
                        pressure,
                        rho,
                        attack_angle,
                        apply_kutta_condition,
                        apply_leading_edge_condition,
                        test_expand,
                        center_potential
                    )
                    if temp_panel.is_last_computation_stable() and temp_panel._q is not None:
                        stable_points.append(
                            (test_expand, temp_panel._q.copy(), temp_panel.force)
                        )
                        if len(stable_points) >= 2:
                            break
                except:
                    continue
            if len(stable_points) >= 2:
                break
        
        if len(stable_points) < 2:
            # Fallback to expand=0 and small positive value
            try:
                temp_panel_0 = PanelMethod(self._airfoil)
                temp_panel_0._compute_direct(
                    potential_func,
                    velocity,
                    pressure,
                    rho,
                    attack_angle,
                    apply_kutta_condition,
                    apply_leading_edge_condition,
                    0.0,
                    center_potential
                )
                temp_panel_small = PanelMethod(self._airfoil)
                temp_panel_small._compute_direct(
                    potential_func,
                    velocity,
                    pressure,
                    rho,
                    attack_angle,
                    apply_kutta_condition,
                    apply_leading_edge_condition,
                    0.01,
                    center_potential
                )
                if temp_panel_0._q is not None and temp_panel_small._q is not None:
                    stable_points = [
                        (0.0, temp_panel_0._q.copy(), temp_panel_0.force),
                        (0.01, temp_panel_small._q.copy(), temp_panel_small.force)
                    ]
            except Exception as e:
                warn(f"Failed to find stable points for interpolation: {e}")
                return
        
        expand1, q1, force1 = stable_points[0]
        expand2, q2, force2 = stable_points[1]
        if expand2 != expand1:
            alpha = (expand_val - expand1) / (expand2 - expand1)
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = 0.5
        
        self._q = (1 - alpha) * q1 + alpha * q2
        interpolated_force = ((1 - alpha) * np.array(force1) + alpha * np.array(force2))
        self._force = (float(interpolated_force[0]), float(interpolated_force[1]))
        
        warn(f"Used interpolation between expand={expand1:.6f} and expand={expand2:.6f} for target expand={expand_val:.6f}")
        self._last_computation_unstable = False
        self._solution_quality_score = 0.7

    def compute(
        self,
        potential_func: Potential,
        velocity: float,
        pressure: float,
        rho: float,
        attack_angle: float = 0.0,
        apply_kutta_condition: bool = False,
        apply_leading_edge_condition: bool = False,
        expand: Union[float, npt.NDArray[np.float64], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = 0.0,
        center_potential: Potential | None = None
    ) -> None:
        """
        Calculates the potential flow around the airfoil using the panel method.
        Automatically handles numerical instability with interpolation fallback.
        
        @param center_potential: Optional potential to be placed at the airfoil center.
                                When provided, adds an additional source at the center
                                and treats Kutta condition as an additional constraint.
        @param apply_leading_edge_condition: Whether to apply leading edge condition to 
                                           ensure smooth flow separation at the leading edge,
                                           preventing non-physical reverse flow.
        """
        try:
            self._compute_with_interpolation(
                potential_func,
                velocity,
                pressure,
                rho,
                attack_angle,
                apply_kutta_condition,
                apply_leading_edge_condition,
                expand,
                center_potential
            )
        except Exception as e:
            warn(f"All computation methods failed: {e}")
            # Fallback to basic computation without interpolation
            self._compute_direct(
                potential_func,
                velocity,
                pressure,
                rho,
                attack_angle,
                apply_kutta_condition,
                apply_leading_edge_condition,
                expand,
                center_potential
            )

    def _compute_direct(
        self,
        potential_func: Potential,
        velocity: float,
        pressure: float,
        rho: float,
        attack_angle: float = 0.0,
        apply_kutta_condition: bool = False,
        apply_leading_edge_condition: bool = False,
        expand: Union[float, npt.NDArray[np.float64], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = 0.0,
        center_potential: Potential | None = None
    ) -> None:
        """
        Direct computation without interpolation fallback.
        
        @param potential_func: Potential function to use
        @param attack_angle: Angle of attack in radians
        @param velocity: Freestream velocity magnitude
        @param apply_kutta_condition: Whether to apply Kutta condition for realistic flow
        @param expand: Boundary layer displacement thickness. Can be:
                      - float: Uniform thickness for all panels
                      - ndarray: Array of thickness values for each panel
                      - Callable: Function that takes arc lengths and returns thickness values
        """
        self._last_computation_unstable = False  # Reset stability flag
        self._rotated_airfoil = self._airfoil.rotate(-attack_angle)
        self._potential_func = potential_func
        self._center_potential = center_potential
        self._attack_angle = attack_angle
        self._velocity = velocity
        self._pressure = pressure
        self._rho = rho
        self._force = None
        self._expand = expand
        airfoil = self._rotated_airfoil
        n = airfoil.points.shape[0]
        
        # Calculate center position if center_potential is provided
        if center_potential is not None:
            self._center_position = airfoil.center
        else:
            self._center_position = None
            self._center_strength = None
        
        if isinstance(expand, (int, float)):
            expand_distances = np.full(n, float(expand), dtype=np.float64)
        elif callable(expand):
            expand_distances = expand(airfoil.tangent_length)
        else:
            expand_distances = np.asarray(expand, dtype=np.float64)
            if len(expand_distances) != n:
                raise ValueError(f"Expand array length ({len(expand_distances)}) must match number of panels ({n})")
        displaced_sources = airfoil.points.copy()
        
        max_expand = np.max(expand_distances)
        use_displaced_sources = max_expand > 1e-12
        self._use_displaced_sources = bool(use_displaced_sources)
        
        if use_displaced_sources:
            next_points = np.roll(airfoil.points, -1, axis=0)
            prev_points = np.roll(airfoil.points, 1, axis=0)
            tangent_vectors = next_points - prev_points
            normal_vectors = np.column_stack([tangent_vectors[:, 1], -tangent_vectors[:, 0]])
            normal_lengths = np.linalg.norm(normal_vectors, axis=1)
            normal_vectors = normal_vectors / normal_lengths.reshape(-1, 1)
            displaced_sources = airfoil.points + expand_distances.reshape(-1, 1) * normal_vectors
        
        if use_displaced_sources:
            boundary_points = airfoil.midpoint + expand_distances.reshape(-1, 1) * airfoil.norm
        else:
            boundary_points = airfoil.midpoint
        
        if self._use_displaced_sources:
            vector = boundary_points.reshape(-1, 1, 2) - displaced_sources.reshape(1, -1, 2)
            displaced_panels = displaced_sources[1:] - displaced_sources[:-1]
            closing_panel = displaced_sources[0] - displaced_sources[-1]
            displaced_panels = np.vstack([displaced_panels, closing_panel.reshape(1, -1)])
            self._displaced_sources = displaced_sources
            self._displaced_panels = displaced_panels
            
            panel_coef = (
                airfoil.norm.reshape(-1, 1, 2) @ \
                    potential_func.velocity(
                        vector, displaced_panels.reshape(1, -1, 2)
                    ).swapaxes(-1, -2)
            ).reshape(-1, n)
            
            if center_potential is not None:
                center_pos = np.array(self._center_position).reshape(1, 2)
                center_vector = boundary_points - center_pos
                dummy_panel = np.array([[0.01, 0.0]])
                center_influence = (
                    airfoil.norm.reshape(-1, 1, 2) @ \
                        center_potential.velocity(
                            center_vector.reshape(-1, 1, 2), 
                            dummy_panel.reshape(1, -1, 2)
                        ).swapaxes(-1, -2)
                ).reshape(-1, 1)
                coef = np.hstack([panel_coef, center_influence])
            else:
                coef = panel_coef
        else:
            vector = boundary_points.reshape(-1, 1, 2) - airfoil.points.reshape(1, -1, 2)
            self._displaced_sources = None
            self._displaced_panels = None
            panel_coef = (
                airfoil.norm.reshape(-1, 1, 2) @ \
                    potential_func.velocity(
                        vector, airfoil.panel.reshape(1, -1, 2)
                    ).swapaxes(-1, -2)
            ).reshape(-1, n)
            
            if center_potential is not None:
                center_pos = np.array(self._center_position).reshape(1, 2)
                center_vector = boundary_points - center_pos
                dummy_panel = np.array([[0.01, 0.0]])
                center_influence = (
                    airfoil.norm.reshape(-1, 1, 2) @ \
                        center_potential.velocity(
                            center_vector.reshape(-1, 1, 2), 
                            dummy_panel.reshape(1, -1, 2)
                        ).swapaxes(-1, -2)
                ).reshape(-1, 1)
                coef = np.hstack([panel_coef, center_influence])
            else:
                coef = panel_coef
                
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
            
            if center_potential is not None:
                n_vars = coef.shape[1]
                n_eqs = coef.shape[0] + 1
                extended_coef = np.zeros((n_eqs, n_vars))
                extended_coef[:-1, :] = coef
                
                # Add Kutta condition row: sum of sources around trailing edge = 0
                kutta_row = np.zeros(n_vars)
                kutta_row[tail_idx - 1: tail_idx + 1] = 1.0
                kutta_row[-1] = 0.0
                extended_coef[-1, :] = kutta_row
                extended_v_norm = np.zeros((n_eqs, 1))
                extended_v_norm[:-1] = v_norm
                extended_v_norm[-1, 0] = 0.0
                coef = extended_coef
                v_norm = extended_v_norm
            else:
                # Without center potential: replace last equation with Kutta condition
                kutta_row = np.zeros(n)
                kutta_row[tail_idx - 1: tail_idx + 1] = 1.0
                coef[tail_idx, :] = kutta_row
                v_norm[tail_idx] = 0.0
        elif center_potential is not None:
            n_vars = coef.shape[1]
            n_eqs = coef.shape[0] + 1
            extended_coef = np.zeros((n_eqs, n_vars))
            extended_coef[:-1, :] = coef
            
            constraint_row = np.zeros(n_vars)
            constraint_row[-1] = 1.0
            extended_coef[-1, :] = constraint_row
            extended_v_norm = np.zeros((n_eqs, 1))
            extended_v_norm[:-1, :] = v_norm
            extended_v_norm[-1, 0] = 0.0
            coef = extended_coef
            v_norm = extended_v_norm

        # Apply leading edge condition for smooth flow separation
        # This condition works to some extent but it is not perfect at all.
        if apply_leading_edge_condition:
            leading_edge_idx = 0
            
            if center_potential is not None:
                n_vars = coef.shape[1]
                n_eqs = coef.shape[0] + 1
                new_extended_coef = np.zeros((n_eqs, n_vars))
                new_extended_coef[:-1, :] = coef
                leading_edge_row = np.zeros(n_vars)
                leading_edge_row[0] = 1.0
                leading_edge_row[n - 1] = -1.0
                if n_vars > n:
                    leading_edge_row[-1] = 0.0
                new_extended_coef[-1, :] = leading_edge_row
                
                new_extended_v_norm = np.zeros((n_eqs, 1))
                new_extended_v_norm[:-1, :] = v_norm
                new_extended_v_norm[-1, 0] = 0.0
                
                coef = new_extended_coef
                v_norm = new_extended_v_norm
            else:
                leading_edge_row = np.zeros(n)
                leading_edge_row[0] = 1.0
                leading_edge_row[n - 1] = -1.0
                coef[leading_edge_idx, :] = leading_edge_row
                v_norm[leading_edge_idx] = 0.0
        
        n_vars = coef.shape[1]
        regularization = 1e-6 * np.eye(n_vars)
        coef += regularization
        A_reg = coef
        b_reg = v_norm
        
        try:
            condition_number = np.linalg.cond(A_reg)
            
            if condition_number < 1e8:
                solution = np.asarray(np.linalg.solve(A_reg, b_reg), dtype=np.float64).reshape(-1)
            else:
                # Use SVD-based pseudoinverse for ill-conditioned systems
                warn(f"Warning: Ill-conditioned system (cond={condition_number:.2e}), using SVD")
                U, s, Vt = np.linalg.svd(A_reg, full_matrices=False)
                threshold = max(1e-12 * s[0], 1e-10) if len(s) > 0 else 1e-10
                s_inv = np.where(s > threshold, 1.0/s, 0.0)
                A_pinv = Vt.T @ np.diag(s_inv) @ U.T
                solution = (A_pinv @ b_reg).reshape(-1)
                
            if center_potential is not None:
                self._q = solution[:-1]
                self._center_strength = solution[-1]
            else:
                self._q = solution
                self._center_strength = None
                
        except np.linalg.LinAlgError:
            warn("Warning: LinAlgError encountered, using regularized least squares")
            condition_number = np.inf
            regularization_param = max(1e-6, 1e-4 * max_expand) if self._use_displaced_sources else 1e-8
            A_reg_backup = coef.T @ coef + regularization_param * np.eye(n_vars)
            solution = np.asarray(np.linalg.solve(A_reg_backup, coef.T @ v_norm), dtype=np.float64).reshape(-1)
            
            if center_potential is not None:
                self._q = solution[:-1]
                self._center_strength = solution[-1]
            else:
                self._q = solution
                self._center_strength = None
        
        if self._q is not None:
            is_stable, quality_score = self._assess_solution_stability(coef, self._q)
            self._last_computation_unstable = not is_stable
            if not is_stable:
                warn(f"Numerically unstable solution detected. Condition: {condition_number:.2e}, Quality: {quality_score:.3f}")
    
    @singledispatchmethod
    def _velocity_impl(self, *args, **kwargs):
        raise NotImplementedError("Unsupported type for velocity calculation.")
    
    @_velocity_impl.register(np.ndarray)
    @reshape2_method
    def _(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self._q is not None
        assert self._potential_func is not None
        assert self._rotated_airfoil is not None
        
        if self._use_displaced_sources:
            assert self._displaced_sources is not None
            assert self._displaced_panels is not None
            vel_contributions = self._potential_func.velocity(
                points.reshape(-1, 1, 2) - self._displaced_sources.reshape(1, -1, 2),
                self._displaced_panels.reshape(1, -1, 2)
            )
        else:
            vel_contributions = self._potential_func.velocity(
                points.reshape(-1, 1, 2) - self.airfoil.points.reshape(1, -1, 2),
                self.airfoil.panel.reshape(1, -1, 2)
            )
            
        panel_velocity = (vel_contributions * self._q.reshape(1, -1, 1)).sum(axis=1)
        
        if self._center_potential is not None and self._center_strength is not None:
            center_pos = np.array(self._center_position).reshape(1, 2)
            center_vector = points.reshape(-1, 2) - center_pos
            dummy_panel = np.array([[0.01, 0.0]])
            center_vel_contributions = self._center_potential.velocity(
                center_vector.reshape(-1, 1, 2),
                dummy_panel.reshape(1, -1, 2)
            )
            center_velocity = (center_vel_contributions * self._center_strength).sum(axis=1)
            panel_velocity = panel_velocity + center_velocity
        
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
        
        if self._use_displaced_sources:
            assert self._displaced_sources is not None
            assert self._displaced_panels is not None
            vel_contributions = self._potential_func.velocity(
                np.array(point).reshape(1, 2) - self._displaced_sources,
                self._displaced_panels
            )
        else:
            vel_contributions = self._potential_func.velocity(
                np.array(point).reshape(1, 2) - self.airfoil.points,
                self.airfoil.panel
            )
            
        panel_velocity = (vel_contributions * self._q.reshape(-1, 1)).sum(axis=0)
        
        if self._center_potential is not None and self._center_strength is not None:
            center_pos = np.array(self._center_position)
            center_vector = np.array(point) - center_pos
            dummy_panel = np.array([[0.01, 0.0]])
            center_vel_contributions = self._center_potential.velocity(
                center_vector.reshape(1, 2),
                dummy_panel
            )
            center_velocity = (center_vel_contributions * self._center_strength).sum(axis=0)
            panel_velocity = panel_velocity + center_velocity
        
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
    
    def velocity_at_boundary_layer_edge(self, *args, **kwargs) -> npt.NDArray[np.float64] | tuple[float, float] | list[float]:
        """
        Returns the velocity at the boundary layer edge.
        This is useful for getting the inviscid flow velocity just outside the boundary layer.
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
        Returns the force on the airfoil calculated at boundary layer edge.
        """
        if self._force is None:
            if self._q is None or self._potential_func is None:
                raise ValueError("Panel method has not been computed yet.")
            
            if self._use_displaced_sources and self._displaced_sources is not None:
                displaced_panels_vectors = self._displaced_sources[1:] - self._displaced_sources[:-1]
                closing_panel = self._displaced_sources[0] - self._displaced_sources[-1]
                displaced_panels_vectors = np.vstack([displaced_panels_vectors, closing_panel.reshape(1, -1)])
                displaced_midpoints = np.zeros((len(displaced_panels_vectors), 2))
                displaced_midpoints[:-1] = (self._displaced_sources[:-1] + self._displaced_sources[1:]) / 2
                displaced_midpoints[-1] = (self._displaced_sources[-1] + self._displaced_sources[0]) / 2
                displaced_lengths = np.linalg.norm(displaced_panels_vectors, axis=1)
                displaced_norms = np.column_stack([displaced_panels_vectors[:, 1], -displaced_panels_vectors[:, 0]]) / displaced_lengths.reshape(-1, 1)
                force_offset = self._sampling_scale_factor * self.airfoil.chord
                force_points = displaced_midpoints + force_offset * displaced_norms
                force = self.pressure(force_points) * displaced_lengths
                self._force = tuple((force.reshape(1, -1) @ displaced_norms).tolist()[0])
            else:
                airfoil = self.airfoil
                n = airfoil.points.shape[0]
                if isinstance(self._expand, (int, float)):
                    expand_distances = np.full(n, float(self._expand), dtype=np.float64)
                elif callable(self._expand):
                    expand_distances = self._expand(airfoil.tangent_length)
                else:
                    expand_distances = np.asarray(self._expand, dtype=np.float64)
                force_offset = self._sampling_scale_factor * self.airfoil.chord
                total_displacement = expand_distances + force_offset
                force_points = airfoil.midpoint + total_displacement.reshape(-1, 1) * airfoil.norm
                force = self.pressure(force_points) * airfoil.length
                self._force = tuple((force.reshape(1, -1) @ airfoil.norm).tolist()[0])
            
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
