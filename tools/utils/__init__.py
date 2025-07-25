import warnings

def format_warning(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"

warnings.formatwarning = format_warning

del warnings
del format_warning

from .naca import naca_4_digit_f

__all__ = [
    "naca_4_digit_f",
]