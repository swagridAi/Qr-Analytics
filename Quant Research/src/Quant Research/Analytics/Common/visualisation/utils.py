"""
Visualization Utility Functions

This module provides common utility functions used by various visualization modules.
These functions handle common tasks like configuring axes, adding annotations,
creating multi-panel figures, and other shared functionality.
"""

# Standard library imports
import logging
from typing import Any, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.visualization.utils")

#------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------

def configure_date_axis(
    ax: Axes, 
    date_format: str = '%Y-%m-%d',
    major_interval: Optional[int] = None,
    minor_interval: Optional[int] = None,
    rot: int = 45
) -> Axes:
    """
    Configure the x-axis for date display.
    
    Args:
        ax: Matplotlib axes object
        date_format: Date format string
        major_interval: Interval for major tick marks (None for auto)
        minor_interval: Interval for minor tick marks (None for auto)
        rot: Rotation angle for tick labels
        
    Returns:
        Configured axes object
    """
    # Format the date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    
    # Set custom date intervals if provided
    if major_interval is not None:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=major_interval))
    
    if minor_interval is not None:
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=minor_interval))
    
    # Rotate labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rot, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, which='major')
    
    return ax


def add_horizontal_line(
    ax: Axes, 
    y_value: float, 
    **kwargs
) -> Line2D:
    """
    Add a horizontal line to a plot.
    
    Args:
        ax: Matplotlib axes object
        y_value: Y-coordinate of the horizontal line
        **kwargs: Additional arguments for ax.axhline
        
    Returns:
        The line object
    """
    # Set default parameters unless specified
    line_params = {
        'color': 'black',
        'linestyle': '--',
        'alpha': 0.5,
        'linewidth': 1.0,
        'zorder': 1
    }
    
    # Update with provided parameters
    line_params.update(kwargs)
    
    # Create the line
    line = ax.axhline(y=y_value, **line_params)
    
    return line


def add_vertical_line(
    ax: Axes, 
    x_value: Union[float, pd.Timestamp], 
    **kwargs
) -> Line2D:
    """
    Add a vertical line to a plot.
    
    Args:
        ax: Matplotlib axes object
        x_value: X-coordinate of the vertical line
        **kwargs: Additional arguments for ax.axvline
        
    Returns:
        The line object
    """
    # Set default parameters unless specified
    line_params = {
        'color': 'black',
        'linestyle': '--',
        'alpha': 0.5,
        'linewidth': 1.0,
        'zorder': 1
    }
    
    # Update with provided parameters
    line_params.update(kwargs)
    
    # Create the line
    line = ax.axvline(x=x_value, **line_params)
    
    return line


def add_annotations(
    ax: Axes,
    x_values: List[Union[float, pd.Timestamp]],
    y_values: List[float],
    texts: List[str],
    **kwargs
) -> List[Any]:
    """
    Add annotations to a plot.
    
    Args:
        ax: Matplotlib axes object
        x_values: List of x-coordinates
        y_values: List of y-coordinates
        texts: List of annotation texts
        **kwargs: Additional arguments for ax.annotate
        
    Returns:
        List of annotation objects
    """
    # Set default parameters unless specified
    annotation_params = {
        'xytext': (0, 10),
        'textcoords': 'offset points',
        'ha': 'center',
        'va': 'bottom',
        'fontsize': 9,
        'alpha': 0.8,
        'arrowprops': {'arrowstyle': '->', 'alpha': 0.6}
    }
    
    # Update with provided parameters
    annotation_params.update(kwargs)
    
    # Create the annotations
    annotations = []
    for x, y, text in zip(x_values, y_values, texts):
        annotation = ax.annotate(
            text, 
            xy=(x, y), 
            **annotation_params
        )
        annotations.append(annotation)
    
    return annotations


def add_legend(
    ax: Axes,
    loc: str = 'best',
    frameon: bool = True,
    framealpha: float = 0.8,
    **kwargs
) -> plt.legend:
    """
    Add a legend to a plot with sensible defaults.
    
    Args:
        ax: Matplotlib axes object
        loc: Legend location
        frameon: Whether to show the legend frame
        framealpha: Alpha transparency of the legend frame
        **kwargs: Additional arguments for ax.legend
        
    Returns:
        The legend object
    """
    # Set default parameters unless specified
    legend_params = {
        'loc': loc,
        'frameon': frameon,
        'framealpha': framealpha,
        'fancybox': True,
        'fontsize': 10
    }
    
    # Update with provided parameters
    legend_params.update(kwargs)
    
    # Create the legend
    legend = ax.legend(**legend_params)
    
    return legend


def format_y_axis(
    ax: Axes,
    y_type: str = 'numeric',
    **kwargs
) -> Axes:
    """
    Format the y-axis based on data type.
    
    Args:
        ax: Matplotlib axes object
        y_type: Type of y-axis ('numeric', 'percent', 'log', 'dollar')
        **kwargs: Additional formatting parameters
        
    Returns:
        Configured axes object
    """
    if y_type == 'percent':
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    elif y_type == 'log':
        ax.set_yscale('log')
    elif y_type == 'dollar':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    
    # Apply additional specific formatting
    if 'y_min' in kwargs:
        ax.set_ylim(bottom=kwargs['y_min'])
    if 'y_max' in kwargs:
        ax.set_ylim(top=kwargs['y_max'])
    if 'grid' in kwargs:
        ax.grid(kwargs['grid'], axis='y', linestyle='--', alpha=0.7)
    
    return ax


def save_figure(
    fig: Figure,
    filename: str,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    pad_inches: float = 0.1,
    transparent: bool = False,
    facecolor: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a figure with sensible defaults.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box in inches
        pad_inches: Padding in inches
        transparent: Whether to use a transparent background
        facecolor: Figure facecolor
        **kwargs: Additional arguments for fig.savefig
        
    Returns:
        None
    """
    # Set default parameters unless specified
    save_params = {
        'dpi': dpi,
        'bbox_inches': bbox_inches,
        'pad_inches': pad_inches,
        'transparent': transparent
    }
    
    # Add facecolor if provided
    if facecolor is not None:
        save_params['facecolor'] = facecolor
    
    # Update with provided parameters
    save_params.update(kwargs)
    
    # Save the figure
    try:
        fig.savefig(filename, **save_params)
        logger.info(f"Figure saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save figure to {filename}: {e}")


def create_color_map(
    values: np.ndarray,
    cmap_name: str = 'RdYlGn',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    reverse: bool = False
) -> Tuple[mcolors.Colormap, mcolors.Normalize]:
    """
    Create a color map for numeric values.
    
    Args:
        values: Array of values to map to colors
        cmap_name: Name of the colormap
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        reverse: Whether to reverse the colormap
        
    Returns:
        Tuple of (colormap, norm)
    """
    # Set default value range if not provided
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    
    # Create colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Reverse colormap if requested
    if reverse:
        cmap = cmap.reversed()
    
    # Create normalization
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    return cmap, norm


def create_multi_panel(
    rows: int = 1,
    cols: int = 1,
    height_ratios: Optional[List[float]] = None,
    width_ratios: Optional[List[float]] = None,
    figsize: Tuple[float, float] = None,
    dpi: int = 100,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs
) -> Tuple[Figure, Union[Axes, List[Axes], List[List[Axes]]]]:
    """
    Create a multi-panel figure with customizable layout.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        height_ratios: Relative heights of rows
        width_ratios: Relative widths of columns
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
        sharex: Whether to share x-axes
        sharey: Whether to share y-axes
        **kwargs: Additional arguments for plt.subplots
        
    Returns:
        Tuple of (figure, axes)
    """
    # Calculate default figure size if not provided
    if figsize is None:
        base_width = 8
        base_height = 6
        figsize = (base_width * cols, base_height * rows)
    
    # Set up gridspec parameters
    gridspec_kw = {}
    if height_ratios is not None:
        if len(height_ratios) != rows:
            raise ValueError(f"height_ratios must have length {rows}, got {len(height_ratios)}")
        gridspec_kw['height_ratios'] = height_ratios
    
    if width_ratios is not None:
        if len(width_ratios) != cols:
            raise ValueError(f"width_ratios must have length {cols}, got {len(width_ratios)}")
        gridspec_kw['width_ratios'] = width_ratios
    
    # Create the figure and axes
    fig, axes = plt.subplots(
        rows, cols,
        figsize=figsize,
        dpi=dpi,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
        **kwargs
    )
    
    # Adjust spacing
    plt.tight_layout()
    
    return fig, axes