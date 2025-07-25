from typing import Any, Callable, Optional, Union, TypeVar, Generic, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from .base import Var, Const, Output
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

T = TypeVar('T')


class TypeMeta(type):
    def __call__(
        cls,
        var: Var[T],
        callback: Callable[[T], float],
        name: str | None = None
    ) -> "ConvergenceMetric[T]":
        if not isinstance(var, Var):
            raise TypeError(f"Expected Var type, got {type(var)}")
        if not callable(callback):
            raise TypeError(f"Expected callable for callback, got {type(callback)}")
        return cls(var=var, callback=callback, name=name)

class ConvergenceMetric(Generic[T]):
    """Convergence metric configuration with generic type support."""
    var: Var[T]
    callback: Callable[[T], float]
    name: str | None = None

    def __init__(
        self,
        var: Var[T],
        callback: Callable[[T], float],
        name: str | None
    ) -> None:
        self.var = var
        self.callback = callback
        if name is None:
            name = f"{self.var.name}_{getattr(self.callback, '__name__', 'custom')}"
        self.name = name
    
    @property
    def display_name(self) -> str:
        """Ensure a non-None name is returned."""
        return self.name or f"{self.var.name}_callback"


class StatisticMetric(Enum):
    """Statistical metric types that create ConvergenceMetric instances."""
    
    @staticmethod
    def MEAN(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a mean convergence metric."""
        if name is None:
            name = f"{var.name}_mean"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.mean(data)),
            name=name
        )
    
    @staticmethod
    def STD(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a standard deviation convergence metric."""
        if name is None:
            name = f"{var.name}_std"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.std(data)),
            name=name
        )
    
    @staticmethod
    def MAX(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a maximum convergence metric."""
        if name is None:
            name = f"{var.name}_max"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.max(data)),
            name=name
        )
    
    @staticmethod
    def MIN(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a minimum convergence metric."""
        if name is None:
            name = f"{var.name}_min"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.min(data)),
            name=name
        )
    
    @staticmethod
    def MEDIAN(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a median convergence metric."""
        if name is None:
            name = f"{var.name}_median"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.median(data)),
            name=name
        )
    
    @staticmethod
    def NORM_L1(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a L1 norm convergence metric."""
        if name is None:
            name = f"{var.name}_norm_l1"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.linalg.norm(data, ord=1)),
            name=name
        )
    
    @staticmethod
    def NORM_L2(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a L2 norm convergence metric."""
        if name is None:
            name = f"{var.name}_norm_l2"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.linalg.norm(data, ord=2)),
            name=name
        )
    
    @staticmethod
    def NORM_INF(var: Var[npt.NDArray[np.floating]], name: str | None = None) -> ConvergenceMetric[npt.NDArray[np.floating]]:
        """Create a infinity norm convergence metric."""
        if name is None:
            name = f"{var.name}_norm_inf"
        return ConvergenceMetric(
            var=var,
            callback=lambda data: float(np.linalg.norm(data, ord=np.inf)),
            name=name
        )


def custom_metric(var: Var, callback: Callable[[T], float], name: str | None = None) -> ConvergenceMetric[T]:
    """Create a custom convergence metric with user-defined callback."""
    return ConvergenceMetric(var=var, callback=callback, name=name)


class ConvergenceAnalyzer:
    """Convergence analyzer for iterative solutions."""
    
    metrics: list[ConvergenceMetric[Any]]
    history: dict[str, list[float]]
    iterations: list[int]
    
    def __init__(self, metrics: list[ConvergenceMetric[Any]]) -> None:
        """
        Initialize convergence analyzer.
        
        @param metrics: List of convergence metrics to analyze
        """
        for metric in metrics:
            if not isinstance(metric.var, Var):
                raise TypeError(f"Metric variable must be Var type, got {type(metric.var)}")
            if isinstance(metric.var, (Const, Output)):
                raise ValueError(f"Cannot analyze convergence for {type(metric.var).__name__} type: {metric.var}")
        
        self.metrics = metrics
        self.history: dict[str, list[float]] = {metric.display_name: [] for metric in metrics}
        self.iterations: list[int] = []
    
    def update(self, iteration: int, chain_values: dict[Var, Any]) -> None:
        """Update convergence data."""
        self.iterations.append(iteration)
        
        for metric in self.metrics:
            if metric.var not in chain_values:
                raise KeyError(f"Variable {metric.var} not found in chain values")
            
            data = chain_values[metric.var]
            statistic_value = metric.callback(data)
            self.history[metric.display_name].append(statistic_value)
    
    def analyze_convergence(self, window_size: int = 10, tolerance: float = 1e-6) -> dict[str, dict[str, Any]]:
        """
        Analyze convergence behavior.
        
        @param window_size: Window size for trend calculation
        @param tolerance: Convergence tolerance
        @return: Convergence analysis results
        """
        if len(self.iterations) < 2:
            return {}
        
        results: dict[str, dict[str, Any]] = {}
        
        for metric in self.metrics:
            history: npt.NDArray[np.floating] = np.array(self.history[metric.display_name])
            
            current_value: float = float(history[-1])
            initial_value: float = float(history[0])
            relative_change: float = abs(current_value - initial_value) / (abs(initial_value) + 1e-12)
            
            if len(history) >= window_size:
                recent_values: npt.NDArray[np.floating] = history[-window_size:]
                trend_slope: float = float(np.polyfit(range(window_size), recent_values, 1)[0])
                is_trending_down: bool = trend_slope < -tolerance
                is_trending_up: bool = trend_slope > tolerance
                is_stable: bool = abs(trend_slope) <= tolerance
            else:
                recent_values = history
                trend_slope = 0.0
                is_stable = False
                is_trending_down = False
                is_trending_up = False
            
            recent_std: float = float(np.std(recent_values))
            is_converged: bool = recent_std < tolerance and is_stable and len(history) >= window_size
            
            if len(history) >= 4:
                diff: npt.NDArray[np.floating] = np.diff(history)
                sign_changes: int = int(np.sum(np.diff(np.sign(diff)) != 0))
                oscillation_ratio: float = sign_changes / max(len(diff) - 1, 1)
                is_oscillating: bool = oscillation_ratio > 0.5
            else:
                oscillation_ratio = 0.0
                is_oscillating = False
            
            results[metric.display_name] = {
                'current_value': current_value,
                'initial_value': initial_value,
                'relative_change': relative_change,
                'trend_slope': trend_slope,
                'recent_std': recent_std,
                'is_converged': is_converged,
                'is_stable': is_stable,
                'is_trending_down': is_trending_down,
                'is_trending_up': is_trending_up,
                'is_oscillating': is_oscillating,
                'oscillation_ratio': oscillation_ratio,
                'num_iterations': len(history)
            }
        
        return results
    
    def generate_report(self, window_size: int = 10, tolerance: float = 1e-6) -> str:
        """Generate convergence analysis report."""
        analysis: dict[str, dict[str, Any]] = self.analyze_convergence(window_size, tolerance)
        
        if not analysis:
            return "Convergence Analysis: Insufficient data, requires at least 2 iterations"
        
        report_lines: list[str] = []
        report_lines.append("=" * 60)
        report_lines.append("Convergence Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Total iterations: {len(self.iterations)}")
        report_lines.append(f"Analysis window size: {window_size}")
        report_lines.append(f"Convergence tolerance: {tolerance:.2e}")
        report_lines.append("")
        
        for metric_name, result in analysis.items():
            report_lines.append(f"Metric: {metric_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"  Current value: {result['current_value']:.6e}")
            report_lines.append(f"  Initial value: {result['initial_value']:.6e}")
            report_lines.append(f"  Relative change: {result['relative_change']:.6e}")
            report_lines.append(f"  Trend slope: {result['trend_slope']:.6e}")
            report_lines.append(f"  Recent std dev: {result['recent_std']:.6e}")
            
            status: list[str] = []
            if result['is_converged']:
                status.append("Converged")
            if result['is_stable']:
                status.append("Stable")
            elif result['is_trending_down']:
                status.append("Decreasing")
            elif result['is_trending_up']:
                status.append("Increasing")
            if result['is_oscillating']:
                status.append(f"Oscillating (ratio: {result['oscillation_ratio']:.3f})")
            
            if status:
                report_lines.append(f"  Status: {', '.join(status)}")
            else:
                report_lines.append("  Status: Unstable")
            
            report_lines.append("")
        
        converged_metrics: list[str] = [name for name, result in analysis.items() if result['is_converged']]
        total_metrics: int = len(analysis)
        convergence_rate: float = len(converged_metrics) / total_metrics if total_metrics > 0 else 0
        
        report_lines.append("Overall Convergence Assessment:")
        report_lines.append("-" * 40)
        report_lines.append(f"  Converged metrics: {len(converged_metrics)}/{total_metrics}")
        report_lines.append(f"  Convergence rate: {convergence_rate:.1%}")
        
        if convergence_rate >= 0.8:
            report_lines.append("  Assessment: Overall converged")
        elif convergence_rate >= 0.5:
            report_lines.append("  Assessment: Partially converged")
        else:
            report_lines.append("  Assessment: Not yet converged")
        
        return "\n".join(report_lines)
    
    def plot_convergence(self, save_path: str | None = None, show: bool = True) -> None:
        """Plot convergence charts."""
        try:
            import matplotlib.pyplot as plt
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
        except ImportError:
            print("Warning: matplotlib not installed, cannot generate charts")
            return
        
        if len(self.iterations) < 2:
            print("Warning: Insufficient data for convergence chart")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        n_metrics: int = len(self.metrics)
        if n_metrics == 0:
            return
        
        if n_metrics == 1:
            rows, cols = 1, 1
        elif n_metrics <= 4:
            rows, cols = 2, 2
        elif n_metrics <= 6:
            rows, cols = 2, 3
        elif n_metrics <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        axes_result: 'Axes' | npt.NDArray[Any]
        axes_result = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))[1]
        
        # Normalize axes to always be a list using isinstance check
        if isinstance(axes_result, np.ndarray):
            axes: list['Axes'] = axes_result.flatten().tolist()
        else:
            axes = [axes_result]
        
        for i, metric in enumerate(self.metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            history: list[float] = self.history[metric.display_name]
            
            ax.plot(self.iterations, history, 'b-', linewidth=2, label=metric.display_name)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.set_title(f'{metric.display_name} Convergence History')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if len(history) >= 10:
                z: npt.NDArray[np.floating] = np.polyfit(self.iterations[-10:], history[-10:], 1)
                p = np.poly1d(z)
                ax.plot(self.iterations[-10:], p(self.iterations[-10:]), 'r--', 
                       alpha=0.7, label='Recent Trend')
                ax.legend()
        
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
