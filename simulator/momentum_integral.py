import numpy as np
import numpy.typing as npt

from tools import Airfoil
from collections.abc import Callable
from typing import Any
from warnings import warn

C1_120 = 1 / 120.0
C1_9072 = 1 / 9072.0
C1_945 = 1 / 945.0
C37_315 = 37 / 315.0



class MomentumIntegralMethod:
    """
    Class for computing the momentum integral method for boundary layer flow.
    """
    def __init__(self, airfoil: Airfoil):
        self.airfoil = airfoil

    def compute(
        self,
        nu: float,
        U_e_array: npt.NDArray[np.float64]
    ):
        """
        Using (corrected formulations for laminar flow):
        - Pohlhausen velocity profile:
        $$
            f \\left( \\eta  \\right) = 2 \\eta - 2 \\eta ^{3} + \\eta ^{4} + \\frac{\\lambda }{6} \\eta \\left( 1 - \\eta  \\right)^{3}
        $$
        - Pohlhausen skin friction (laminar):
        $$
            c_{f} = \\frac{2 \\nu}{U_{e} \\delta} \\left( 2 + \\frac{\\lambda}{6} \\right)
        $$
        - Blasius solution (corrected for displacement thickness):
        $$
            \\delta^{*} \\left( x_{0} \\right) = 1.7208 \\sqrt{\\frac{\\nu x_{0}}{U_{e}\\left( x_{0} \\right)}}
        $$
        - Shape factor (corrected definition):
        $$
            H = \\frac{\\delta^{*}}{\\theta}
        $$

        Momentum equation: $\\frac{d \\theta }{ds} + \\frac{\\theta }{U_{e}} \\frac{d U_{e}}{ds} \\left( 2 + H \\right) = \\frac{c_{f}}{2}$
        
        @param nu: kinematic viscosity
        @param U_e_array: array of edge velocities at panel midpoints, same indexing as airfoil panels
        """
        airfoil = self.airfoil
        tail_ind = airfoil.tail_index

        # Lower surface: panels 0 to tail_ind-1
        # Upper surface: panels tail_ind to end  
        U_e_lower_panels = U_e_array[:tail_ind]
        U_e_upper_panels = U_e_array[tail_ind:]

        # Lower Surface
        lower_s_coords = np.zeros(tail_ind + 1)
        for i in range(tail_ind):
            lower_s_coords[i + 1] = lower_s_coords[i] + airfoil.length[i]
        
        panel_midpoint_s = (lower_s_coords[:-1] + lower_s_coords[1:]) / 2
        U_e_lower_interp = lambda s: np.interp(s, panel_midpoint_s, U_e_lower_panels)
        U_e_lower_val = U_e_lower_interp(lower_s_coords)
        dUe_ds_lower = np.zeros_like(U_e_lower_val)
        dUe_ds_lower[0] = (U_e_lower_val[1] - U_e_lower_val[0]) / (lower_s_coords[1] - lower_s_coords[0])
        dUe_ds_lower[-1] = (U_e_lower_val[-1] - U_e_lower_val[-2]) / (lower_s_coords[-1] - lower_s_coords[-2])
        for i in range(1, len(dUe_ds_lower) - 1):
            ds = lower_s_coords[i+1] - lower_s_coords[i-1]
            dUe_ds_lower[i] = (U_e_lower_val[i+1] - U_e_lower_val[i-1]) / ds

        delta_lower = np.zeros_like(U_e_lower_val)
        delta_star_lower = np.zeros_like(U_e_lower_val)
        lambda_lower = np.zeros_like(U_e_lower_val)
        theta_lower = np.zeros_like(U_e_lower_val)
        half_cf_lower = np.zeros_like(U_e_lower_val)
        H_lower = np.zeros_like(U_e_lower_val)
        
        # Find the first positive U_e value to start calculations
        first_positive_idx = None
        for idx in range(len(U_e_lower_val)):
            if U_e_lower_val[idx] > 0:
                first_positive_idx = idx
                break
        
        if first_positive_idx is None:
            # All U_e values are non-positive, use fallback
            first_positive_idx = 1
            warn("No positive U_e values found on lower surface, using index 1 as fallback")
        
        if first_positive_idx == 0:
            first_positive_idx = 1
        if first_positive_idx < len(lower_s_coords) - 1:
            s_distance = lower_s_coords[first_positive_idx]
            if s_distance > 0:
                ini = np.sqrt(nu * s_distance / max(U_e_lower_val[first_positive_idx], 1e-6))
            else:
                ini = np.sqrt(nu * airfoil.length[0] / max(U_e_lower_val[first_positive_idx], 1e-6))
            theta_lower[first_positive_idx] = 0.664 * ini
        else:
            ini = np.sqrt(nu * airfoil.length[0] / max(U_e_lower_val[1], 1e-6))
            theta_lower[1] = 0.664 * ini
            first_positive_idx = 1
        
        max_step_factor = 0.5
        min_step_factor = 0.01
        target_theta_change = 0.1
        current_s = lower_s_coords[first_positive_idx]
        i = first_positive_idx
        iteration_count = 0
        while current_s < lower_s_coords[-1] and i < len(U_e_lower_val) - 1:
            iteration_count += 1
            U_e_current = U_e_lower_interp(current_s)
            ds_small = min(current_s * 1e-6, 1e-4)
            if current_s + ds_small < lower_s_coords[-1]:
                dUe_ds_current = (U_e_lower_interp(current_s + ds_small) - U_e_current) / ds_small
            else:
                dUe_ds_current = (U_e_current - U_e_lower_interp(current_s - ds_small)) / ds_small
            
            if i == first_positive_idx:
                # Use Blasius initial condition
                s_distance = lower_s_coords[i]
                if s_distance > 0:
                    delta_lower[i] = 5.0 * np.sqrt(nu * s_distance / U_e_current)
                else:
                    delta_lower[i] = 5.0 * ini
                theta_coef = theta_lower[i] / delta_lower[i]
            else:
                lambda_prev = np.clip(lambda_lower[i-1], -12.0, 12.0)
                theta_coef = C37_315 - C1_945 * lambda_prev - C1_9072 * lambda_prev ** 2
                theta_coef = max(float(theta_coef), 1e-6)
                delta_lower[i] = theta_lower[i] / theta_coef
            
            delta_lower[i] = max(delta_lower[i], 1e-8)
            if U_e_current <= 1e-6:
                warn(f"Near-zero or negative velocity detected at i={i}, U_e={U_e_current:.6e}, using minimum value")
                U_e_current = 1e-6
            
            lambda_lower[i] = delta_lower[i] ** 2 / nu * dUe_ds_current
            lambda_lower[i] = np.clip(lambda_lower[i], -12.0, 12.0)
            half_cf_lower[i] = nu / (U_e_current * delta_lower[i]) * (2 + lambda_lower[i] / 6)
            if abs(half_cf_lower[i]) > 1.0:
                warn(f"Large skin friction coefficient detected at i={i}, cf={2*half_cf_lower[i]:.6e}")
                half_cf_lower[i] = np.sign(half_cf_lower[i]) * min(abs(half_cf_lower[i]), 1.0)
            delta_star_coeff = 0.3 - C1_120 * lambda_lower[i]
            delta_star_coeff = max(delta_star_coeff, 1e-6)
            H_lower[i] = delta_star_coeff / theta_coef
            H_lower[i] = np.clip(H_lower[i], 1.1, 10.0)
            dtheta_ds = half_cf_lower[i] - (2 + H_lower[i]) * theta_lower[i] / U_e_current * dUe_ds_current
            
            if abs(dtheta_ds) > 1e3 or abs(theta_lower[i]) > 0.1:
                warn(f"Unstable numerical behavior detected at i={i}, s={current_s:.6f}, dtheta_ds={dtheta_ds:.6e}, theta={theta_lower[i]:.6e}, stopping iteration")
                stopped_at_index = i
                break
            if abs(dtheta_ds) > 1e-12:
                suggested_ds = min(target_theta_change / abs(dtheta_ds), 0.1 * lower_s_coords[-1])
                base_ds = lower_s_coords[min(i+1, len(lower_s_coords)-1)] - current_s
                if base_ds <= 0:
                    base_ds = lower_s_coords[-1] - current_s
                step_factor = np.clip(suggested_ds / base_ds, min_step_factor, max_step_factor)
                adaptive_ds = step_factor * base_ds
            else:
                adaptive_ds = lower_s_coords[min(i+1, len(lower_s_coords)-1)] - current_s
                if adaptive_ds <= 0:
                    adaptive_ds = (lower_s_coords[-1] - current_s) * 0.1
            
            remaining_distance = lower_s_coords[-1] - current_s
            adaptive_ds = min(adaptive_ds, remaining_distance)
            theta_new = theta_lower[i] + dtheta_ds * adaptive_ds
            theta_new = np.clip(theta_new, 1e-10, 0.01)
            
            current_s += adaptive_ds
            i += 1
            if i < len(theta_lower):
                theta_lower[i] = theta_new
        else:
            stopped_at_index = len(theta_lower)
        
        self._interpolate_remaining_values(
            theta_lower, delta_lower, lambda_lower, half_cf_lower, H_lower,
            stopped_at_index, tail_ind
        )
            
        delta_star_coeff_vec = 0.3 - C1_120 * lambda_lower
        delta_star_lower = delta_lower * delta_star_coeff_vec
        
        invalid_mask = np.isnan(delta_star_lower) | np.isinf(delta_star_lower) | (delta_star_lower < 0)
        if np.any(invalid_mask):
            warn(f"Found {np.sum(invalid_mask)} invalid delta_star values, replacing with interpolation")
            valid_indices = np.where(~invalid_mask)[0]
            if len(valid_indices) > 0:
                delta_star_lower[invalid_mask] = np.interp(
                    np.where(invalid_mask)[0], 
                    valid_indices, 
                    delta_star_lower[valid_indices]
                )
            else:
                delta_star_lower[invalid_mask] = 1e-6
        
        # Interpolate displacement thickness for points before first_positive_idx
        # This handles both negative U_e points and the leading edge point (index 0)
        if first_positive_idx > 0:
            # Use approximately 10% of total points for fitting, but at least 3 points
            total_valid_points = len(delta_star_lower) - first_positive_idx
            n_fit_points = max(3, int(0.1 * total_valid_points))
            fit_indices = []
            fit_s_coords = []
            fit_delta_star = []
            for idx in range(first_positive_idx, min(first_positive_idx + n_fit_points, len(delta_star_lower))):
                if delta_star_lower[idx] > 0:  # Valid calculated point
                    fit_indices.append(idx)
                    fit_s_coords.append(lower_s_coords[idx])
                    fit_delta_star.append(delta_star_lower[idx])
            
            if len(fit_indices) >= 2:
                # Fit delta_star = ln(1 + a*x) where x is distance along airfoil
                fit_s_array = np.array(fit_s_coords)
                fit_delta_array = np.array(fit_delta_star)
                
                def objective(a):
                    if a <= 0:
                        return 1e6  # Penalty for non-positive 'a'
                    try:
                        predicted = np.log(1 + a * fit_s_array)
                        return np.sum((predicted - fit_delta_array) ** 2)
                    except:
                        return 1e6
                
                # Initial guess for 'a' based on the first point
                if fit_s_array[0] > 0 and fit_delta_array[0] > 0:
                    a_initial = (np.exp(fit_delta_array[0]) - 1) / fit_s_array[0]
                else:
                    a_initial = 1.0
                
                a_initial = max(a_initial, 0.1)
                a_candidates = np.linspace(a_initial * 0.1, a_initial * 10, 100)
                a_candidates = a_candidates[a_candidates > 0]
                best_a = a_initial
                best_error = objective(a_initial)
                for a in a_candidates:
                    error = objective(a)
                    if error < best_error:
                        best_error = error
                        best_a = a
                
                for idx in range(first_positive_idx):
                    s_point = lower_s_coords[idx]
                    if s_point >= 0:
                        try:
                            delta_star_fitted = np.log(1 + best_a * s_point)
                            delta_star_lower[idx] = max(delta_star_fitted, 0.0)
                        except:
                            delta_star_lower[idx] = 0.0
                    else:
                        delta_star_lower[idx] = 0.0
            else:
                # Fallback: not enough points for fitting, use simple scaling
                if len(fit_indices) == 1:
                    ref_delta_star = fit_delta_star[0]
                    ref_s = fit_s_coords[0]
                    if ref_s > 0 and ref_delta_star > 0:
                        a_est = (np.exp(ref_delta_star) - 1) / ref_s
                    else:
                        a_est = 1.0
                else:
                    a_est = 1.0
                
                a_est = max(a_est, 0.1)
                for idx in range(first_positive_idx):
                    s_point = lower_s_coords[idx]
                    if s_point >= 0:
                        try:
                            delta_star_fitted = np.log(1 + a_est * s_point)
                            delta_star_lower[idx] = max(delta_star_fitted, 0.0)
                        except:
                            delta_star_lower[idx] = 0.0
                    else:
                        delta_star_lower[idx] = 0.0
            
        # Upper Surface
        n_upper = len(airfoil.points) - tail_ind
        upper_tangent_lengths = np.zeros(n_upper)
        for i in range(n_upper - 1):
            panel_idx = tail_ind + i
            upper_tangent_lengths[i + 1] = upper_tangent_lengths[i] + airfoil.length[panel_idx]
        U_e_upper_val = U_e_upper_panels
        dUe_ds_upper = np.zeros_like(U_e_upper_val)
        dUe_ds_upper[0] = (U_e_upper_val[1] - U_e_upper_val[0]) / airfoil.length[tail_ind]
        dUe_ds_upper[-1] = (U_e_upper_val[-1] - U_e_upper_val[-2]) / airfoil.length[len(airfoil.points) - 1]
        for i in range(1, len(dUe_ds_upper) - 1):
            panel_idx1 = tail_ind + i - 1
            panel_idx2 = tail_ind + i
            dUe_ds_upper[i] = (U_e_upper_val[i + 1] - U_e_upper_val[i - 1]) / (airfoil.length[panel_idx1] + airfoil.length[panel_idx2])

        n_upper_panels = len(U_e_upper_panels)
        upper_s_coords = np.zeros(n_upper_panels + 1)
        for i in range(n_upper_panels):
            panel_idx = tail_ind + i
            if panel_idx < len(airfoil.length):
                upper_s_coords[i + 1] = upper_s_coords[i] + airfoil.length[panel_idx]
            else:
                upper_s_coords[i + 1] = upper_s_coords[i] + (upper_s_coords[i] - upper_s_coords[i-1] if i > 0 else 0.01)
        
        upper_panel_midpoint_s = (upper_s_coords[:-1] + upper_s_coords[1:]) / 2
        U_e_upper_interp = lambda s: np.interp(s, upper_panel_midpoint_s, U_e_upper_panels)
        
        delta_upper = np.zeros(n_upper_panels + 1)
        delta_star_upper = np.zeros(n_upper_panels + 1)
        lambda_upper = np.zeros(n_upper_panels + 1)
        theta_upper = np.zeros(n_upper_panels + 1)
        half_cf_upper = np.zeros(n_upper_panels + 1)
        H_upper = np.zeros(n_upper_panels + 1)
        
        if len(U_e_upper_val) > 1:
            ini_upper = np.sqrt(nu * airfoil.length[tail_ind] / U_e_upper_val[1])
            theta_upper[1] = 0.664 * ini_upper

            current_s_upper = upper_s_coords[1]
            i = 1
            while current_s_upper < upper_s_coords[-1] and i < len(theta_upper) - 1:
                U_e_current = U_e_upper_interp(current_s_upper)
                ds_small = min(current_s_upper * 1e-6, 1e-4)
                if current_s_upper + ds_small < upper_s_coords[-1]:
                    dUe_ds_current = (U_e_upper_interp(current_s_upper + ds_small) - U_e_current) / ds_small
                else:
                    dUe_ds_current = (U_e_current - U_e_upper_interp(current_s_upper - ds_small)) / ds_small
                
                if i == 1:
                    delta_upper[i] = 5.0 * ini_upper
                    theta_coef = theta_upper[1] / delta_upper[1]
                else:
                    lambda_prev = np.clip(lambda_upper[i-1], -12.0, 12.0)
                    theta_coef = C37_315 - C1_945 * lambda_prev - C1_9072 * lambda_prev ** 2
                    theta_coef = max(float(theta_coef), 1e-6)
                    delta_upper[i] = theta_upper[i] / theta_coef
                
                delta_upper[i] = max(delta_upper[i], 1e-8)
                lambda_upper[i] = delta_upper[i] ** 2 / nu * dUe_ds_current
                lambda_upper[i] = np.clip(lambda_upper[i], -12.0, 12.0)
                half_cf_upper[i] = nu / (U_e_current * delta_upper[i]) * (2 + lambda_upper[i] / 6)
                delta_star_coeff = 0.3 - C1_120 * lambda_upper[i]
                delta_star_coeff = max(delta_star_coeff, 1e-6)
                H_upper[i] = delta_star_coeff / theta_coef
                dtheta_ds = half_cf_upper[i] - (2 + H_upper[i]) * theta_upper[i] / U_e_current * dUe_ds_current
                
                if abs(dtheta_ds) > 1e-12:
                    suggested_ds = target_theta_change / abs(dtheta_ds)
                    base_ds = upper_s_coords[min(i+1, len(upper_s_coords)-1)] - current_s_upper
                    if base_ds <= 0:
                        base_ds = upper_s_coords[-1] - current_s_upper
                    step_factor = np.clip(suggested_ds / base_ds, min_step_factor, max_step_factor)
                    adaptive_ds = step_factor * base_ds
                else:
                    adaptive_ds = upper_s_coords[min(i+1, len(upper_s_coords)-1)] - current_s_upper
                    if adaptive_ds <= 0:
                        adaptive_ds = (upper_s_coords[-1] - current_s_upper) * 0.1
                
                remaining_distance = upper_s_coords[-1] - current_s_upper
                adaptive_ds = min(adaptive_ds, remaining_distance)
                theta_new = theta_upper[i] + dtheta_ds * adaptive_ds
                theta_new = max(theta_new, 1e-10)
                current_s_upper += adaptive_ds
                i += 1
                if i < len(theta_upper):
                    theta_upper[i] = theta_new
                if abs(dtheta_ds) > 1e6 or theta_new > 1.0:
                    warn(f"Unstable numerical behavior detected at i={i}, s={current_s_upper:.6f}, stopping iteration")
                    stopped_at_index_upper = i
                    break
            else:
                stopped_at_index_upper = len(theta_upper)
            
            self._interpolate_remaining_values(
                theta_upper, delta_upper, lambda_upper, half_cf_upper, H_upper,
                stopped_at_index_upper, len(theta_upper)
            )
        
        delta_star_coeff_vec_upper = 0.3 - C1_120 * lambda_upper
        delta_star_upper = delta_upper * delta_star_coeff_vec_upper
        
        # Replace any remaining NaN values
        for arr_name, arr in [
            ("delta_star_lower", delta_star_lower), ("delta_star_upper", delta_star_upper),
            ("theta_lower", theta_lower), ("theta_upper", theta_upper),
            ("delta_lower", delta_lower), ("delta_upper", delta_upper),
            ("H_lower", H_lower), ("H_upper", H_upper),
            ("half_cf_lower", half_cf_lower), ("half_cf_upper", half_cf_upper)
        ]:
            invalid_mask = np.isnan(arr) | np.isinf(arr)
            if np.any(invalid_mask):
                warn(f"Found {np.sum(invalid_mask)} invalid values in {arr_name}, replacing with safe defaults")
                if "delta_star" in arr_name:
                    arr[invalid_mask] = 1e-6
                elif "theta" in arr_name:
                    arr[invalid_mask] = 1e-8
                elif "delta" in arr_name:
                    arr[invalid_mask] = 1e-6
                elif "H_" in arr_name:
                    arr[invalid_mask] = 2.5
                elif "cf" in arr_name:
                    arr[invalid_mask] = 1e-6

        self._delta_lower_data = delta_lower
        self._delta_star_lower_data = delta_star_lower
        self._theta_lower_data = theta_lower  
        self._H_lower_data = H_lower
        self._cf_lower_data = 2 * half_cf_lower
        self._delta_upper_data = delta_upper
        self._delta_star_upper_data = delta_star_upper
        self._theta_upper_data = theta_upper
        self._H_upper_data = H_upper
        self._cf_upper_data = 2 * half_cf_upper
        self._lower_tangent_lengths = lower_s_coords
        self._upper_tangent_lengths = upper_s_coords
    
    def _interpolate_remaining_values(self, theta_array, delta_array, lambda_array, half_cf_array, H_array, 
                                    stopped_index, total_length):
        """
        Interpolates remaining values for the boundary layer if the computation stopped early.
        """
        if stopped_index >= total_length - 1:
            return
        
        valid_end = min(stopped_index, len(theta_array))
        if valid_end < 2:
            warn("Not enough valid data points to interpolate remaining values.")
            for i in range(valid_end, min(total_length, len(theta_array))):
                theta_array[i] = 1e-6
                delta_array[i] = 1e-5
                lambda_array[i] = 0.0
                half_cf_array[i] = 1e-6
                H_array[i] = 2.5
            return
        
        last_valid_idx = valid_end - 1
        
        theta_array[last_valid_idx] = np.clip(theta_array[last_valid_idx], 1e-10, 0.01)
        delta_array[last_valid_idx] = np.clip(delta_array[last_valid_idx], 1e-8, 0.1)
        lambda_array[last_valid_idx] = np.clip(lambda_array[last_valid_idx], -12.0, 12.0)
        half_cf_array[last_valid_idx] = np.clip(half_cf_array[last_valid_idx], -1.0, 1.0)
        H_array[last_valid_idx] = np.clip(H_array[last_valid_idx], 1.1, 10.0)
        
        # Use simple exponential decay to trailing edge
        target_theta = 1e-8
        target_delta = 1e-7
        
        remaining_length = min(total_length, len(theta_array)) - valid_end
        if remaining_length <= 0:
            return
            
        for idx, i in enumerate(range(valid_end, min(total_length, len(theta_array)))):
            relative_pos = (idx + 1) / remaining_length
            decay_factor = np.exp(-2.0 * relative_pos)
            
            theta_array[i] = max(target_theta, theta_array[last_valid_idx] * decay_factor)
            delta_array[i] = max(target_delta, delta_array[last_valid_idx] * decay_factor)
            lambda_array[i] = lambda_array[last_valid_idx] * decay_factor
            half_cf_array[i] = half_cf_array[last_valid_idx] * decay_factor
            H_array[i] = np.clip(H_array[last_valid_idx] * (1 - relative_pos) + 2.5 * relative_pos, 1.1, 10.0)
    
    @property
    def delta_lower(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for lower surface boundary layer thickness"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._lower_tangent_lengths, self._delta_lower_data)
        return interpolate_func
    
    @property
    def theta_lower(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for lower surface momentum thickness"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._lower_tangent_lengths, self._theta_lower_data)
        return interpolate_func
    
    @property
    def H_lower(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for lower surface shape factor"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._lower_tangent_lengths, self._H_lower_data)
        return interpolate_func
    
    @property
    def cf_lower(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for lower surface skin friction coefficient"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._lower_tangent_lengths, self._cf_lower_data)
        return interpolate_func
    
    @property
    def delta_star_lower(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for lower surface displacement thickness"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._lower_tangent_lengths, self._delta_star_lower_data)
        return interpolate_func
    
    @property
    def delta_upper(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for upper surface boundary layer thickness"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._upper_tangent_lengths, self._delta_upper_data)
        return interpolate_func
    
    @property
    def theta_upper(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for upper surface momentum thickness"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._upper_tangent_lengths, self._theta_upper_data)
        return interpolate_func
    
    @property
    def H_upper(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for upper surface shape factor"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._upper_tangent_lengths, self._H_upper_data)
        return interpolate_func
    
    @property
    def cf_upper(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for upper surface skin friction coefficient"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._upper_tangent_lengths, self._cf_upper_data)
        return interpolate_func
    
    @property
    def delta_star_upper(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Returns interpolation function for upper surface displacement thickness"""
        def interpolate_func(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.interp(s, self._upper_tangent_lengths, self._delta_star_upper_data)
        return interpolate_func
