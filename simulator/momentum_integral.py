import numpy as np
import numpy.typing as npt

from tools import Airfoil
from collections.abc import Callable
from typing import Any

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
        U_e_upper: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        U_e_lower: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ):
        """
        Using
        - Pohlhausen:
        $$
            f \\left( \\eta  \\right) = 2 \\eta - 2 \\eta ^{3} + \\eta ^{4} + \\frac{\\lambda }{6} \\eta \\left( 1 - \\eta  \\right)^{3}
        $$
        - Ludwieg-Tillmann:
        $$
            c_{f} = \\frac{0.246}{10^{0.678H}} \\cdot Re_{\\theta }^{-0.268}
        $$
        - Blasius:
        $$
            \\delta \\left( x_{0} \\right) = 1.7208 \\sqrt{\\frac{\\nu x_{0}}{U_{e}\\left( x_{0} \\right)}}
        $$

        Equation: $\\frac{d \\theta }{ds} + \\frac{\\theta }{U_{e}} \\frac{d U_{e}}{ds} \\left( 2 + H \\right) = \\frac{c_{f}}{2}$
        """
        airfoil = self.airfoil
        tail_ind = airfoil.tail_index

        # lower surface
        U_e_lower_val = U_e_lower(
            np.insert(
                self.airfoil.tangent_length[:tail_ind],
                0,
                0.0
            )
        )
        dUe_ds_lower = (U_e_lower_val[1:] - U_e_lower_val[:-1]) / airfoil.length[:tail_ind]

        delta_lower = np.zeros_like(U_e_lower_val)
        delta_star_lower = np.zeros_like(U_e_lower_val)
        lambda_lower = np.zeros_like(U_e_lower_val)
        theta_lower = np.zeros_like(U_e_lower_val)
        cf_lower = np.zeros_like(U_e_lower_val)
        H_lower = np.zeros_like(U_e_lower_val)
        
        # Blasius
        delta_lower[1] = 1.7208 * np.sqrt(nu * airfoil.length[0] / U_e_lower_val[1])
        theta_lower[1] = 0.664 * np.sqrt(nu * airfoil.length[0] / U_e_lower_val[1])

        for i in range(1, min(tail_ind, len(U_e_lower_val) - 1)):
            if i-1 < len(dUe_ds_lower) and delta_lower[i] > 0:
                lambda_lower[i] = (delta_lower[i] ** 2 / nu) * dUe_ds_lower[i-1]
                lambda_lower[i] = np.clip(lambda_lower[i], -10.0, 10.0)
            else:
                lambda_lower[i] = 0.0
            
            denominator = 0.3 - C1_120 * lambda_lower[i]  # 3/10 - lambda/120
            if abs(denominator) < 1e-10:
                denominator = 1e-10 if denominator >= 0 else -1e-10
            H_lower[i] = (C37_315 - C1_9072 * lambda_lower[i] ** 2 - C1_945 * lambda_lower[i]) / denominator
            H_lower[i] = np.clip(H_lower[i], 1.4, 10.0)
            
            # H = theta/delta*
            if H_lower[i] > 0:
                delta_star_lower[i] = theta_lower[i] / H_lower[i]
            else:
                delta_star_lower[i] = theta_lower[i] / 2.0
            
            # c_f = 0.246 * 10 ** (-0.678 * H) * Re_theta ** (-0.268)
            Re_theta = max(U_e_lower_val[i] * theta_lower[i] / nu, 1.0)
            cf_lower[i] = 0.246 * 10 ** (-0.678 * H_lower[i]) * Re_theta ** (-0.268)
            cf_lower[i] = max(cf_lower[i], 1e-6)
            
            # Solve
            if i-1 < len(dUe_ds_lower):
                dtheta_ds = cf_lower[i] / 2 - theta_lower[i] / U_e_lower_val[i] * (2 + H_lower[i]) * dUe_ds_lower[i-1]
            else:
                dtheta_ds = cf_lower[i] / 2

            step_size = airfoil.length[i]
            max_theta_change = 0.05 * theta_lower[i] if theta_lower[i] > 0 else 1e-8
            if abs(dtheta_ds * step_size) > max_theta_change:
                dtheta_ds = np.sign(dtheta_ds) * max_theta_change / step_size
                
            if i + 1 < len(theta_lower):
                theta_lower[i + 1] = max(theta_lower[i] + dtheta_ds * step_size, 1e-10)
                theta_coeff = C37_315 - C1_9072 * lambda_lower[i] ** 2 - C1_945 * lambda_lower[i]
                if abs(theta_coeff) > 1e-10:
                    delta_lower[i + 1] = theta_lower[i + 1] / theta_coeff
                else:
                    delta_lower[i + 1] = theta_lower[i + 1] / 0.37
        
        # upper surface
        upper_surface_total_length = self.airfoil.tangent_length[-1] - self.airfoil.tangent_length[tail_ind]
        n_upper_points = len(self.airfoil.points) - tail_ind
        upper_distances = np.linspace(0, upper_surface_total_length * 0.99, n_upper_points, dtype=np.float64)
        
        U_e_upper_val = U_e_upper(upper_distances)[::-1]
        upper_lengths = airfoil.length[tail_ind:][::-1]
        dUe_ds_upper = (U_e_upper_val[1:] - U_e_upper_val[:-1]) / upper_lengths[:-1]

        delta_upper = np.zeros_like(U_e_upper_val)
        delta_star_upper = np.zeros_like(U_e_upper_val)
        lambda_upper = np.zeros_like(U_e_upper_val)
        theta_upper = np.zeros_like(U_e_upper_val)
        cf_upper = np.zeros_like(U_e_upper_val)
        H_upper = np.zeros_like(U_e_upper_val)
        
        U_e_safe = np.maximum(U_e_upper_val, 1e-3)
        leading_edge_distance_upper = airfoil.length[tail_ind - 1] if tail_ind > 0 else airfoil.length[0]
        delta_upper[1] = 1.7208 * np.sqrt(nu * leading_edge_distance_upper / U_e_safe[1])
        theta_upper[1] = 0.664 * np.sqrt(nu * leading_edge_distance_upper / U_e_safe[1])
        
        for i in range(1, min(len(U_e_safe) - 1, len(upper_lengths))):
            if i-1 < len(dUe_ds_upper) and delta_upper[i] > 0:
                lambda_upper[i] = (delta_upper[i] ** 2 / nu) * dUe_ds_upper[i-1]
                lambda_upper[i] = np.clip(lambda_upper[i], -10.0, 10.0)
            else:
                lambda_upper[i] = 0.0
            
            denominator = 0.3 - C1_120 * lambda_upper[i]
            if abs(denominator) < 1e-10:
                denominator = 1e-10 if denominator >= 0 else -1e-10
            H_upper[i] = (C37_315 - C1_9072 * lambda_upper[i] ** 2 - C1_945 * lambda_upper[i]) / denominator
            H_upper[i] = np.clip(H_upper[i], 1.4, 10.0)
            
            if H_upper[i] > 0:
                delta_star_upper[i] = theta_upper[i] / H_upper[i]
            else:
                delta_star_upper[i] = theta_upper[i] / 2.0
            
            Re_theta = max(U_e_safe[i] * theta_upper[i] / nu, 1.0)
            cf_upper[i] = 0.246 * 10 ** (-0.678 * H_upper[i]) * Re_theta ** (-0.268)
            cf_upper[i] = max(cf_upper[i], 1e-6)
            
            if i-1 < len(dUe_ds_upper):
                dtheta_ds = cf_upper[i] / 2 - theta_upper[i] / U_e_safe[i] * (2 + H_upper[i]) * dUe_ds_upper[i-1]
            else:
                dtheta_ds = cf_upper[i] / 2
                
            step_size = upper_lengths[i]
            max_theta_change = 0.05 * theta_upper[i] if theta_upper[i] > 0 else 1e-8
            if abs(dtheta_ds * step_size) > max_theta_change:
                dtheta_ds = np.sign(dtheta_ds) * max_theta_change / step_size
                
            if i + 1 < len(theta_upper):
                theta_upper[i + 1] = max(theta_upper[i] + dtheta_ds * step_size, 1e-10)
                theta_coeff = C37_315 - C1_9072 * lambda_upper[i] ** 2 - C1_945 * lambda_upper[i]
                if abs(theta_coeff) > 1e-10:
                    delta_upper[i + 1] = theta_upper[i + 1] / theta_coeff
                else:
                    delta_upper[i + 1] = theta_upper[i + 1] / 0.37

        self._delta_lower_data = delta_lower
        self._delta_star_lower_data = delta_star_lower
        self._theta_lower_data = theta_lower  
        self._H_lower_data = H_lower
        self._cf_lower_data = cf_lower
        self._delta_upper_data = delta_upper
        self._delta_star_upper_data = delta_star_upper
        self._theta_upper_data = theta_upper
        self._H_upper_data = H_upper
        self._cf_upper_data = cf_upper
        
        self._lower_tangent_lengths = np.insert(
            self.airfoil.tangent_length[:tail_ind],
            0,
            0.0
        )
        self._upper_tangent_lengths = upper_distances
    
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
