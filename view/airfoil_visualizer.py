import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Optional, Union, Tuple, Dict, Any
from tools.airfoil import Airfoil

class AirfoilVisualizer:
    """
    A class for visualizing airfoil shapes with various customization options.
    Supports displaying multiple airfoils with different styles and colors.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8), dpi: int = 100):
        """
        Initialize the airfoil visualizer.
        
        @param figsize: Figure size as (width, height) in inches
        @param dpi: Figure DPI for resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.airfoils: List[Dict[str, Any]] = []
        
    def add_airfoil(
        self,
        airfoil: Airfoil,
        label: Optional[str] = None,
        color: str = 'blue',
        alpha: float = 0.7,
        fill: bool = True,
        line_width: float = 2.0,
        line_style: str = '-',
        show_points: bool = False,
        point_size: float = 3.0,
        show_normals: bool = False,
        normal_scale: float = 0.05,
        normal_color: str = 'red',
        show_panels: bool = False,
        panel_color: str = 'green',
        show_midpoints: bool = False,
        midpoint_color: str = 'orange',
        midpoint_size: float = 2.0
    ) -> None:
        """
        Add an airfoil to the visualization.
        
        @param airfoil: The Airfoil object to visualize
        @param label: Label for the airfoil (for legend)
        @param color: Color for the airfoil outline and fill
        @param alpha: Transparency level (0.0 to 1.0)
        @param fill: Whether to fill the airfoil
        @param line_width: Width of the outline
        @param line_style: Style of the outline ('-', '--', '-.', ':')
        @param show_points: Whether to show individual points
        @param point_size: Size of the points
        @param show_normals: Whether to show normal vectors
        @param normal_scale: Scale factor for normal vector length
        @param normal_color: Color for normal vectors
        @param show_panels: Whether to show panel vectors
        @param panel_color: Color for panel vectors
        @param show_midpoints: Whether to show panel midpoints
        @param midpoint_color: Color for midpoints
        @param midpoint_size: Size of midpoint markers
        """
        airfoil_config = {
            'airfoil': airfoil,
            'label': label,
            'color': color,
            'alpha': alpha,
            'fill': fill,
            'line_width': line_width,
            'line_style': line_style,
            'show_points': show_points,
            'point_size': point_size,
            'show_normals': show_normals,
            'normal_scale': normal_scale,
            'normal_color': normal_color,
            'show_panels': show_panels,
            'panel_color': panel_color,
            'show_midpoints': show_midpoints,
            'midpoint_color': midpoint_color,
            'midpoint_size': midpoint_size
        }
        self.airfoils.append(airfoil_config)
    
    def clear_airfoils(self) -> None:
        """Clear all airfoils from the visualizer."""
        self.airfoils.clear()
    
    def _setup_figure(self) -> None:
        """Setup the matplotlib figure and axis."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            if self.ax is not None:
                self.ax.clear()
        
        # Ensure ax is not None for type checking
        assert self.ax is not None
        
        # Set equal aspect ratio to avoid distortion
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Airfoil Visualization')
    
    def _plot_airfoil_outline(self, config: Dict[str, Any]) -> None:
        """Plot the airfoil outline."""
        assert self.ax is not None  # Type guard
        
        airfoil = config['airfoil']
        points = airfoil.points
        
        # Close the curve for proper visualization
        closed_points = np.vstack([points, points[0]])
        
        # Plot filled polygon if requested
        if config['fill']:
            polygon = patches.Polygon(
                points,
                closed=True,
                facecolor=config['color'],
                alpha=config['alpha'],
                edgecolor=config['color'],
                linewidth=config['line_width'],
                linestyle=config['line_style'],
                label=config['label']
            )
            self.ax.add_patch(polygon)
        else:
            # Plot outline only
            self.ax.plot(
                closed_points[:, 0],
                closed_points[:, 1],
                color=config['color'],
                linewidth=config['line_width'],
                linestyle=config['line_style'],
                alpha=config['alpha'],
                label=config['label']
            )
    
    def _plot_points(self, config: Dict[str, Any]) -> None:
        """Plot individual points."""
        if not config['show_points']:
            return
            
        assert self.ax is not None  # Type guard
        
        airfoil = config['airfoil']
        points = airfoil.points
        
        self.ax.scatter(
            points[:, 0],
            points[:, 1],
            s=config['point_size']**2,
            c=config['color'],
            alpha=config['alpha'],
            zorder=10
        )
    
    def _plot_normals(self, config: Dict[str, Any]) -> None:
        """Plot normal vectors."""
        if not config['show_normals']:
            return
            
        assert self.ax is not None  # Type guard
        
        airfoil = config['airfoil']
        midpoints = airfoil.midpoint
        normals = airfoil.norm
        
        # Scale normals for visualization
        scaled_normals = normals * config['normal_scale']
        
        for i in range(len(midpoints)):
            start = midpoints[i]
            end = start + scaled_normals[i]
            self.ax.arrow(
                start[0], start[1],
                scaled_normals[i][0], scaled_normals[i][1],
                head_width=config['normal_scale']/10,
                head_length=config['normal_scale']/10,
                fc=config['normal_color'],
                ec=config['normal_color'],
                alpha=0.7,
                zorder=5
            )
    
    def _plot_panels(self, config: Dict[str, Any]) -> None:
        """Plot panel vectors."""
        if not config['show_panels']:
            return
            
        assert self.ax is not None  # Type guard
        
        airfoil = config['airfoil']
        points = airfoil.points
        
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            self.ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=config['panel_color'],
                linewidth=1.0,
                alpha=0.7,
                zorder=3
            )
    
    def _plot_midpoints(self, config: Dict[str, Any]) -> None:
        """Plot panel midpoints."""
        if not config['show_midpoints']:
            return
            
        assert self.ax is not None  # Type guard
        
        airfoil = config['airfoil']
        midpoints = airfoil.midpoint
        
        self.ax.scatter(
            midpoints[:, 0],
            midpoints[:, 1],
            s=config['midpoint_size']**2,
            c=config['midpoint_color'],
            alpha=0.8,
            zorder=8,
            marker='s'  # Square markers for midpoints
        )
    
    def _add_leading_trailing_markers(self, config: Dict[str, Any]) -> None:
        """Add markers for leading and trailing edges."""
        assert self.ax is not None  # Type guard
        
        airfoil = config['airfoil']
        points = airfoil.points
        
        # Leading edge (first point)
        self.ax.scatter(
            points[0, 0], points[0, 1],
            s=50, c='green', marker='o',
            alpha=0.9, zorder=15,
            edgecolors='darkgreen', linewidth=2
        )
        
        # Trailing edge
        tail_idx = airfoil.tail_index
        self.ax.scatter(
            points[tail_idx, 0], points[tail_idx, 1],
            s=50, c='red', marker='s',
            alpha=0.9, zorder=15,
            edgecolors='darkred', linewidth=2
        )
    
    def plot(
        self,
        show_legend: bool = True,
        show_leading_trailing: bool = True,
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot all added airfoils.
        
        @param show_legend: Whether to show the legend
        @param show_leading_trailing: Whether to mark leading and trailing edges
        @param title: Custom title for the plot
        @param xlim: X-axis limits as (min, max)
        @param ylim: Y-axis limits as (min, max)
        @param save_path: Path to save the figure (optional)
        @param show_plot: Whether to display the plot
        """
        if not self.airfoils:
            print("No airfoils to plot. Add airfoils using add_airfoil() first.")
            return
        
        self._setup_figure()
        assert self.ax is not None  # Type guard
        
        # Plot each airfoil
        for config in self.airfoils:
            self._plot_airfoil_outline(config)
            self._plot_points(config)
            self._plot_normals(config)
            self._plot_panels(config)
            self._plot_midpoints(config)
            
            if show_leading_trailing:
                self._add_leading_trailing_markers(config)
        
        # Customize plot
        if title:
            self.ax.set_title(title)
        
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        
        if show_legend and any(config['label'] for config in self.airfoils):
            self.ax.legend()
        
        # Add markers explanation if leading/trailing edges are shown
        if show_leading_trailing:
            # Add custom legend entries for leading/trailing edges
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                       markersize=8, markeredgecolor='darkgreen', markeredgewidth=2),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                       markersize=8, markeredgecolor='darkred', markeredgewidth=2)
            ]
            
            # Get existing legend
            handles, labels = self.ax.get_legend_handles_labels()
            handles.extend(custom_lines)
            labels.extend(['Leading Edge', 'Trailing Edge'])
            
            if show_legend:
                self.ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
    
    def plot_comparison(
        self,
        airfoils: List[Airfoil],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: str = "Airfoil Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Convenience method to plot multiple airfoils for comparison.
        
        @param airfoils: List of Airfoil objects to compare
        @param labels: List of labels for each airfoil
        @param colors: List of colors for each airfoil
        @param title: Plot title
        @param save_path: Path to save the figure
        """
        # Default colors if not provided
        if colors is None:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Default labels if not provided
        if labels is None:
            labels = [f'Airfoil {i+1}' for i in range(len(airfoils))]
        
        # Clear existing airfoils
        self.clear_airfoils()
        
        # Add all airfoils
        for i, airfoil in enumerate(airfoils):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f'Airfoil {i+1}'
            
            self.add_airfoil(
                airfoil=airfoil,
                label=label,
                color=color,
                alpha=0.6,
                fill=True,
                line_width=2.0
            )
        
        # Plot
        self.plot(title=title, save_path=save_path)


def demo_airfoil_visualizer():
    """
    Demonstration of the AirfoilVisualizer functionality.
    """
    # Create some sample airfoils using simple functions
    print("Creating sample airfoils...")
    
    # Create simple symmetric airfoil
    def simple_upper(x):
        return np.sqrt(x * (1 - x)) * 0.2
    
    def simple_lower(x):
        return -np.sqrt(x * (1 - x)) * 0.2
    
    symmetric_airfoil = Airfoil.from_function(
        [simple_upper, simple_lower],
        n_points=50,
        max_edge_length=0.1
    )
    
    # Create cambered airfoil
    def cambered_upper(x):
        return np.sqrt(x * (1 - x)) * 0.2 + 0.02 * x * (1 - x)
    
    def cambered_lower(x):
        return -np.sqrt(x * (1 - x)) * 0.2 + 0.02 * x * (1 - x)
    
    cambered_airfoil = Airfoil.from_function(
        [cambered_upper, cambered_lower],
        n_points=50,
        max_edge_length=0.1
    )
    
    # Create expanded version
    expanded_airfoil = symmetric_airfoil.expand(0.02)
    
    # Create visualizer
    visualizer = AirfoilVisualizer(figsize=(14, 10))
    
    # Example 1: Single airfoil with detailed visualization
    print("Creating detailed single airfoil visualization...")
    visualizer.add_airfoil(
        symmetric_airfoil,
        label='Symmetric Airfoil',
        color='blue',
        alpha=0.7,
        fill=True,
        show_points=True,
        show_normals=True,
        normal_scale=0.03,
        show_midpoints=True
    )
    
    visualizer.plot(
        title='Detailed Airfoil Visualization (Symmetric)',
        save_path='detailed_airfoil.png'
    )
    
    # Example 2: Multiple airfoils comparison
    print("Creating airfoil comparison...")
    visualizer.plot_comparison(
        airfoils=[symmetric_airfoil, cambered_airfoil, expanded_airfoil],
        labels=['Symmetric', 'Cambered', 'Expanded Symmetric'],
        colors=['blue', 'red', 'green'],
        title='Airfoil Comparison',
        save_path='airfoil_comparison.png'
    )
    
    # Example 3: Custom visualization with multiple configurations
    print("Creating custom multi-airfoil visualization...")
    visualizer.clear_airfoils()
    
    visualizer.add_airfoil(
        symmetric_airfoil,
        label='Original',
        color='blue',
        alpha=0.5,
        fill=True,
        line_width=2
    )
    
    visualizer.add_airfoil(
        expanded_airfoil,
        label='Expanded',
        color='red',
        alpha=0.3,
        fill=False,
        line_width=3,
        line_style='--'
    )
    
    visualizer.plot(
        title='Original vs Expanded Airfoil',
        save_path='original_vs_expanded.png'
    )
    
    print("Demonstration completed!")


if __name__ == "__main__":
    demo_airfoil_visualizer()
