import numpy as np
import numpy.typing as npt


from collections.abc import Callable

def naca_4_digit_f(
        c: float = 1.0,
        m: float = 0.02,
        p: float = 0.4,
        t: float = 0.12
    ) -> tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ...]:
    """
    Generate a single point on NACA 4-digit airfoil

    Let

    $$
        \\begin{cases}
            & y_{c} = 
            \\begin{cases}
                \\frac{m}{p^{2}} \\left( 2 p x - x^{2} \\right), \\quad 0 \\le x \\le p
                \\\\ \\frac{m}{\\left( 1 - p \\right)^{2}} \\left[ \\left( 1 - 2 p \\right) + 2 p x - x^{2} \\right], \\quad p \\lt x \\le 1
            \\end{cases} 
            \\\\ & y_{t} = 5 t \\left( 0.2969 x^{\\frac{1}{2}} - 0.1260 x - 0.3516 x^{2} + 0.2843 x^{3} - 0.1015 x^{4} \\right)
            \\\\ & \\theta = \\arctan \\frac{dy_{c}}{dx} = \\arctan 
            \\begin{cases}
                \\frac{2 m}{p^{2}} \\left( p - x \\right), \\quad 0 \\le x \\le p
                \\\\ \\frac{2 m}{\\left( 1 - p \\right)^{2}} \\left( p - x \\right), \\quad p \\lt x \\le 1
            \\end{cases}
        \\end{cases} 
    $$

    And

    $$
        \\begin{cases}
            upper = 
            c \\begin{pmatrix}
                x - y_{t} \\sin \\theta 
                \\\\ y_{c} + y_{t} \\cos \\theta 
            \\end{pmatrix}
            \\\\ lower = 
            c \\begin{pmatrix}
                x + y_{t} \\sin \\theta 
                \\\\ y_{c} - y_{t} \\cos \\theta 
            \\end{pmatrix}
        \\end{cases} 
    $$
    """
    def upper_surface(x_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Ensure x_array is always a numpy array
        x_array = np.atleast_1d(x_array)
        y_c = np.zeros_like(x_array)
        y_t = 5 * t * (0.2969 * np.sqrt(x_array) - 0.1260 * x_array - 
                    0.3516 * x_array**2 + 0.2843 * x_array**3 - 
                    0.1015 * x_array**4)
        mask_le = (x_array <= p)
        mask_te = (x_array > p)
        y_c[mask_le] = (m / p**2) * (2 * p * x_array[mask_le] - x_array[mask_le]**2)
        y_c[mask_te] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_array[mask_te] - x_array[mask_te]**2)
        theta = np.zeros_like(x_array)
        theta[mask_le] = np.arctan((2 * m / p**2) * (p - x_array[mask_le]))
        theta[mask_te] = np.arctan((2 * m / (1 - p)**2) * (p - x_array[mask_te]))
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_upper = x_array - y_t * sin_theta
        y_upper = y_c + y_t * cos_theta
        return np.column_stack((x_upper, y_upper)) * c
    
    def lower_surface(x_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Ensure x_array is always a numpy array
        x_array = np.atleast_1d(x_array)
        y_c = np.zeros_like(x_array)
        y_t = 5 * t * (0.2969 * np.sqrt(x_array) - 0.1260 * x_array - 
                    0.3516 * x_array**2 + 0.2843 * x_array**3 - 
                    0.1015 * x_array**4)
        mask_le = (x_array <= p)
        mask_te = (x_array > p)
        y_c[mask_le] = (m / p**2) * (2 * p * x_array[mask_le] - x_array[mask_le]**2)
        y_c[mask_te] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_array[mask_te] - x_array[mask_te]**2)
        theta = np.zeros_like(x_array)
        theta[mask_le] = np.arctan((2 * m / p**2) * (p - x_array[mask_le]))
        theta[mask_te] = np.arctan((2 * m / (1 - p)**2) * (p - x_array[mask_te]))
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        x_lower = x_array + y_t * sin_theta
        y_lower = y_c - y_t * cos_theta
        return np.column_stack((x_lower, y_lower)) * c
    
    return upper_surface, lower_surface