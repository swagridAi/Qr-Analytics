"""
Visualization Utilities

This module provides visualization functions used across all analytics modules.
It creates consistent, professional-quality charts and plots for time series data,
statistical analysis, performance metrics, and market analysis.

Features:
- Time series plots (price, returns, volatility)
- Financial charts (candlestick, OHLC)
- Statistical visualizations (distributions, correlations)
- Performance visualizations (drawdowns, metrics)
- Signal and regime visualizations
- Multi-panel composition and layout utilities

Usage:
    ```python
    from quant_research.analytics.common.visualization import (
        plot_time_series,
        plot_candlestick,
        plot_distribution,
        plot_correlation_matrix,
        plot_drawdowns,
        create_multi_panel
    )
    
    # Create simple time series plot
    fig, ax = plot_time_series(price_data, title="Price Chart")
    
    # Create candlestick chart with volume
    fig, axes = plot_candlestick(ohlc_data, volume=True)
    
    # Create return distribution with normal overlay
    fig, ax = plot_distribution(returns, normal_overlay=True)
    
    # Create multi-panel figure with different plots
    fig, axes = create_multi_panel(
        rows=2, cols=1, 
        height_ratios=[3, 1],
        figsize=(12, 8)
    )
    ```
"""

# Standard library imports
import logging
import math
import warnings
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns
from scipy import stats

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.visualization")

# Default color palette
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Set default seaborn style
sns.set_style('whitegrid')

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
    x_value: Union[float, datetime], 
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
    x_values: List[Union[float, datetime]],
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


#------------------------------------------------------------------------
# Time Series Plots
#------------------------------------------------------------------------

def plot_time_series(
    data: Union[pd.Series, pd.DataFrame],
    columns: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = 'Date',
    ylabel: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None,
    linestyles: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    alpha: float = 1.0,
    y_type: str = 'numeric',
    highlight_regions: Optional[List[Tuple[datetime, datetime, str, float]]] = None,
    annotations: Optional[List[Tuple[datetime, float, str]]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot time series data with options for customization.
    
    Args:
        data: Time series data (Series or DataFrame)
        columns: Columns to plot (for DataFrame)
        labels: Labels for the plotted series
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        colors: Colors for each series
        figsize: Figure size (width, height) in inches
        ax: Existing axes to plot on
        linestyles: Line styles for each series
        markers: Markers for each series
        alpha: Alpha transparency
        y_type: Type of y-axis ('numeric', 'percent', 'log', 'dollar')
        highlight_regions: List of (start, end, color, alpha) for highlighting time periods
        annotations: List of (x, y, text) for annotations
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to DataFrame if necessary
    if isinstance(data, pd.Series):
        df = pd.DataFrame(data)
        series_name = data.name if data.name is not None else 'Value'
        df.columns = [series_name]
    else:
        df = data.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns or 'timestamp' in df.columns:
            # Try to set index from date/timestamp column
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            df = df.set_index(date_col)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("Unable to convert index to datetime")
    
    # Determine columns to plot
    if columns is None:
        if df.shape[1] <= 10:  # Plot all columns if 10 or fewer
            columns = df.columns.tolist()
        else:
            columns = df.columns[:10].tolist()  # Plot first 10 columns if more
            logger.warning(f"DataFrame has {df.shape[1]} columns, plotting only the first 10")
    
    # Create default labels if not provided
    if labels is None:
        labels = columns
    
    # Default colors if not provided
    if colors is None:
        colors = DEFAULT_COLORS
    
    # Use cycle if fewer colors than columns
    if len(colors) < len(columns):
        colors = colors * (len(columns) // len(colors) + 1)
    
    # Default linestyles if not provided
    if linestyles is None:
        linestyles = ['-'] * len(columns)
    
    # Use cycle if fewer linestyles than columns
    if len(linestyles) < len(columns):
        linestyles = linestyles * (len(columns) // len(linestyles) + 1)
    
    # Plot each series
    for i, col in enumerate(columns):
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        label = labels[i] if i < len(labels) else col
        
        marker = None if markers is None else markers[i % len(markers)]
        
        ax.plot(
            df.index, df[col],
            color=color,
            linestyle=ls,
            marker=marker,
            alpha=alpha,
            label=label,
            **kwargs
        )
    
    # Add highlight regions if provided
    if highlight_regions is not None:
        for start, end, color, region_alpha in highlight_regions:
            ax.axvspan(start, end, color=color, alpha=region_alpha)
    
    # Add annotations if provided
    if annotations is not None:
        add_annotations(
            ax,
            [a[0] for a in annotations],
            [a[1] for a in annotations],
            [a[2] for a in annotations]
        )
    
    # Set labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    
    # Format y-axis based on type
    format_y_axis(ax, y_type=y_type)
    
    # Format date axis
    configure_date_axis(ax)
    
    # Add legend if multiple series
    if len(columns) > 1:
        add_legend(ax)
    
    plt.tight_layout()
    
    return fig, ax


def plot_returns(
    returns: Union[pd.Series, pd.DataFrame],
    columns: Optional[List[str]] = None,
    cumulative: bool = True,
    log_scale: bool = False,
    compound: bool = True,
    benchmark: Optional[Union[pd.Series, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None,
    colors: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot returns or cumulative returns.
    
    Args:
        returns: Return series or DataFrame
        columns: Columns to plot (for DataFrame)
        cumulative: Whether to plot cumulative returns
        log_scale: Whether to use log scale
        compound: Whether to use compound or simple returns
        benchmark: Benchmark return series or column name
        title: Plot title
        figsize: Figure size (width, height) in inches
        ax: Existing axes to plot on
        colors: Colors for each series
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to DataFrame if necessary
    if isinstance(returns, pd.Series):
        if benchmark is not None and isinstance(benchmark, pd.Series):
            # If benchmark is a separate Series, combine into DataFrame
            df = pd.DataFrame({
                returns.name if returns.name is not None else 'Returns': returns,
                benchmark.name if benchmark.name is not None else 'Benchmark': benchmark
            })
        else:
            df = pd.DataFrame(returns)
            series_name = returns.name if returns.name is not None else 'Returns'
            df.columns = [series_name]
    else:
        df = returns.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            logger.warning("Unable to convert index to datetime")
    
    # Determine columns to plot
    if columns is None:
        if df.shape[1] <= 10:  # Plot all columns if 10 or fewer
            columns = df.columns.tolist()
        else:
            columns = df.columns[:10].tolist()  # Plot first 10 columns if more
            logger.warning(f"DataFrame has {df.shape[1]} columns, plotting only the first 10")
    
    # If benchmark is a column name, ensure it's in the list
    if benchmark is not None and isinstance(benchmark, str):
        if benchmark in df.columns and benchmark not in columns:
            columns.append(benchmark)
    
    # Default colors if not provided
    if colors is None:
        colors = DEFAULT_COLORS
    
    # Calculate cumulative returns if requested
    if cumulative:
        if compound:
            # Compound returns: prod(1+r) - 1
            cum_returns = (1 + df[columns]).cumprod() - 1
        else:
            # Simple returns: sum(r)
            cum_returns = df[columns].cumsum()
        
        plot_data = cum_returns
        ylabel = 'Cumulative Return'
    else:
        plot_data = df[columns]
        ylabel = 'Return'
    
    # Plot each series
    for i, col in enumerate(columns):
        color = colors[i % len(colors)]
        
        # Make benchmark dashed if separate
        ls = '--' if col == benchmark and isinstance(benchmark, str) else '-'
        
        ax.plot(
            plot_data.index,
            plot_data[col],
            color=color,
            linestyle=ls,
            label=col,
            **kwargs
        )
    
    # Add horizontal line at zero
    add_horizontal_line(ax, 0)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    
    if title is None:
        title = 'Cumulative Returns' if cumulative else 'Returns'
    ax.set_title(title)
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Format date axis
    configure_date_axis(ax)
    
    # Add legend if multiple series
    if len(columns) > 1:
        add_legend(ax)
    
    plt.tight_layout()
    
    return fig, ax


def plot_dual_axis(
    data: pd.DataFrame,
    y1_columns: List[str],
    y2_columns: List[str],
    y1_label: str = 'Primary Axis',
    y2_label: str = 'Secondary Axis',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    y1_colors: Optional[List[str]] = None,
    y2_colors: Optional[List[str]] = None,
    y1_style: Dict[str, Any] = None,
    y2_style: Dict[str, Any] = None,
    **kwargs
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Create a dual-axis plot for time series data.
    
    Args:
        data: DataFrame with time series data
        y1_columns: Columns to plot on primary y-axis
        y2_columns: Columns to plot on secondary y-axis
        y1_label: Label for primary y-axis
        y2_label: Label for secondary y-axis
        title: Plot title
        figsize: Figure size (width, height) in inches
        y1_colors: Colors for primary axis series
        y2_colors: Colors for secondary axis series
        y1_style: Additional style parameters for primary axis series
        y2_style: Additional style parameters for secondary axis series
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, (ax1, ax2))
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Create secondary axis
    ax2 = ax1.twinx()
    
    # Default colors if not provided
    if y1_colors is None:
        y1_colors = DEFAULT_COLORS[:len(y1_columns)]
    
    if y2_colors is None:
        y2_colors = DEFAULT_COLORS[len(y1_columns):len(y1_columns)+len(y2_columns)]
        # If we've run out of colors, use different shade of primary colors
        if len(y2_colors) < len(y2_columns):
            y2_colors = [f"C{i}" for i in range(len(y2_columns))]
    
    # Default styles if not provided
    if y1_style is None:
        y1_style = {'alpha': 0.8, 'linewidth': 2}
    
    if y2_style is None:
        y2_style = {'alpha': 0.8, 'linewidth': 2, 'linestyle': '--'}
    
    # Plot data on primary axis
    for i, col in enumerate(y1_columns):
        color = y1_colors[i % len(y1_colors)]
        ax1.plot(
            data.index, 
            data[col],
            color=color,
            label=col,
            **y1_style
        )
    
    # Plot data on secondary axis
    for i, col in enumerate(y2_columns):
        color = y2_colors[i % len(y2_colors)]
        ax2.plot(
            data.index,
            data[col],
            color=color,
            label=col,
            **y2_style
        )
    
    # Set labels and title
    ax1.set_xlabel('Date')
    ax1.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)
    
    if title is not None:
        plt.title(title)
    
    # Format date axis
    configure_date_axis(ax1)
    
    # Create a single legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def plot_area(
    data: Union[pd.Series, pd.DataFrame],
    columns: Optional[List[str]] = None,
    stacked: bool = False,
    normalized: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[List[str]] = None,
    alpha: float = 0.7,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create an area plot from time series data.
    
    Args:
        data: Time series data
        columns: Columns to include
        stacked: Whether to create a stacked area plot
        normalized: Whether to normalize values (for stacked=True)
        title: Plot title
        figsize: Figure size (width, height) in inches
        colors: Colors for each series
        alpha: Alpha transparency
        ax: Existing axes to plot on
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to DataFrame if necessary
    if isinstance(data, pd.Series):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            logger.warning("Unable to convert index to datetime")
    
    # Determine columns to plot
    if columns is None:
        columns = df.columns.tolist()
    
    # Extract the data to plot
    plot_data = df[columns]
    
    # Normalize if requested
    if normalized and len(columns) > 1:
        plot_data = plot_data.div(plot_data.sum(axis=1), axis=0)
    
    # Default colors if not provided
    if colors is None:
        colors = DEFAULT_COLORS[:len(columns)]
    
    # Create the area plot
    if stacked:
        plot_data.plot.area(
            ax=ax,
            stacked=True,
            alpha=alpha,
            color=colors,
            **kwargs
        )
    else:
        for i, col in enumerate(columns):
            ax.fill_between(
                plot_data.index,
                plot_data[col],
                alpha=alpha,
                color=colors[i % len(colors)],
                label=col,
                **kwargs
            )
    
    # Set labels and title
    ax.set_xlabel('Date')
    
    if normalized:
        ax.set_ylabel('Proportion')
        format_y_axis(ax, y_type='percent')
    else:
        ax.set_ylabel('Value')
    
    if title is not None:
        ax.set_title(title)
    
    # Format date axis
    configure_date_axis(ax)
    
    # Add legend if multiple series
    if len(columns) > 1:
        add_legend(ax)
    
    plt.tight_layout()
    
    return fig, ax


#------------------------------------------------------------------------
# Financial Charts
#------------------------------------------------------------------------

def plot_candlestick(
    data: pd.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: Optional[str] = 'volume',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    up_color: str = 'green',
    down_color: str = 'red',
    alpha: float = 0.6,
    volume: bool = True,
    volume_height_ratio: float = 0.2,
    **kwargs
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Create a candlestick chart with optional volume subplot.
    
    Args:
        data: OHLC data
        open_col: Column name for open prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
        volume_col: Column name for volume
        title: Plot title
        figsize: Figure size (width, height) in inches
        up_color: Color for up candles
        down_color: Color for down candles
        alpha: Alpha transparency
        volume: Whether to include volume subplot
        volume_height_ratio: Height ratio of volume to price subplot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Import matplotlib finance utilities
    try:
        from mplfinance.original_flavor import candlestick_ohlc
        from matplotlib.dates import date2num
    except ImportError:
        logger.error("mplfinance package is required for candlestick plots")
        raise ImportError("mplfinance package is required for candlestick plots")
    
    # Create figure and axes based on whether volume is included
    if volume and volume_col in data.columns:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=figsize,
            gridspec_kw={'height_ratios': [1-volume_height_ratio, volume_height_ratio]},
            sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    
    # Ensure OHLC columns exist
    required_cols = [open_col, high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except:
            logger.warning("Unable to convert index to datetime")
            raise ValueError("Data index must be convertible to datetime")
    
    # Convert data to OHLC format required by candlestick_ohlc
    ohlc_data = []
    for date, row in data.iterrows():
        date_num = date2num(date)
        open_price = row[open_col]
        high_price = row[high_col]
        low_price = row[low_col]
        close_price = row[close_col]
        ohlc_data.append([date_num, open_price, high_price, low_price, close_price])
    
    # Create the candlestick chart
    candlestick_ohlc(
        ax1,
        ohlc_data,
        colorup=up_color,
        colordown=down_color,
        alpha=alpha,
        width=0.6
    )
    
    # Add volume subplot if requested
    if volume and volume_col in data.columns and ax2 is not None:
        # Calculate up and down volume
        up_days = data[close_col] >= data[open_col]
        down_days = data[close_col] < data[open_col]
        
        # Plot volume with colors based on price movement
        ax2.bar(
            data.index[up_days],
            data[volume_col][up_days],
            color=up_color,
            alpha=alpha
        )
        ax2.bar(
            data.index[down_days],
            data[volume_col][down_days],
            color=down_color,
            alpha=alpha
        )
        
        # Format volume axis
        ax2.set_ylabel('Volume')
        
        # Format y-axis to avoid scientific notation
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Set labels and title
    ax1.set_ylabel('Price')
    if title is not None:
        ax1.set_title(title)
    
    # Format date axis
    configure_date_axis(ax1)
    
    # Format y-axis
    format_y_axis(ax1, y_type='numeric')
    
    fig.tight_layout()
    
    if ax2 is not None:
        return fig, [ax1, ax2]
    else:
        return fig, ax1


def plot_ohlc(
    data: pd.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create an OHLC (bars) chart.
    
    Args:
        data: OHLC data
        open_col: Column name for open prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
        title: Plot title
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Import matplotlib finance utilities
    try:
        from mplfinance.original_flavor import plot_day_summary_ohlc
        from matplotlib.dates import date2num
    except ImportError:
        logger.error("mplfinance package is required for OHLC plots")
        raise ImportError("mplfinance package is required for OHLC plots")
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure OHLC columns exist
    required_cols = [open_col, high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except:
            logger.warning("Unable to convert index to datetime")
            raise ValueError("Data index must be convertible to datetime")
    
    # Convert data to OHLC format required by plot_day_summary_ohlc
    ohlc_data = []
    for date, row in data.iterrows():
        date_num = date2num(date)
        open_price = row[open_col]
        high_price = row[high_col]
        low_price = row[low_col]
        close_price = row[close_col]
        ohlc_data.append([date_num, open_price, high_price, low_price, close_price])
    
    # Create the OHLC chart
    plot_day_summary_ohlc(
        ax,
        ohlc_data,
        ticksize=2,
        **kwargs
    )
    
    # Set labels and title
    ax.set_ylabel('Price')
    if title is not None:
        ax.set_title(title)
    
    # Format date axis
    configure_date_axis(ax)
    
    # Format y-axis
    format_y_axis(ax, y_type='numeric')
    
    fig.tight_layout()
    
    return fig, ax


def plot_volume_profile(
    data: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
    bins: int = 50,
    orientation: str = 'horizontal',
    figsize: Tuple[float, float] = (10, 8),
    color: str = 'blue',
    alpha: float = 0.6,
    title: Optional[str] = None,
    price_range: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a volume profile chart.
    
    Args:
        data: Price and volume data
        price_col: Column name for price
        volume_col: Column name for volume
        bins: Number of price bins
        orientation: Orientation of histogram ('horizontal' or 'vertical')
        figsize: Figure size (width, height) in inches
        color: Color for the histogram
        alpha: Alpha transparency
        title: Plot title
        price_range: Optional (min, max) range for price axis
        ax: Existing axes to plot on
        **kwargs: Additional arguments for histogram
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Ensure required columns exist
    required_cols = [price_col, volume_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract price and volume data
    prices = data[price_col].values
    volumes = data[volume_col].values
    
    # Set price range
    if price_range is None:
        price_min = prices.min()
        price_max = prices.max()
        # Add some margin
        margin = (price_max - price_min) * 0.05
        price_range = (price_min - margin, price_max + margin)
    
    # Create histogram weights based on volume
    weights = volumes
    
    # Create the histogram
    if orientation == 'horizontal':
        ax.hist(
            prices, 
            bins=bins, 
            weights=weights,
            orientation='horizontal',
            color=color,
            alpha=alpha,
            range=price_range,
            **kwargs
        )
        ax.set_xlabel('Volume')
        ax.set_ylabel('Price')
    else:  # vertical
        ax.hist(
            prices, 
            bins=bins, 
            weights=weights,
            orientation='vertical',
            color=color,
            alpha=alpha,
            range=price_range,
            **kwargs
        )
        ax.set_xlabel('Price')
        ax.set_ylabel('Volume')
    
    # Set title
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Volume Profile')
    
    plt.tight_layout()
    
    return fig, ax


def plot_technical_indicators(
    data: pd.DataFrame,
    indicators: Dict[str, Dict[str, Any]],
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume',
    figsize: Tuple[float, float] = (12, 9),
    num_panels: int = 3,
    **kwargs
) -> Tuple[Figure, List[Axes]]:
    """
    Plot price with technical indicators.
    
    Args:
        data: Price and indicator data
        indicators: Dictionary mapping indicators to plot params
        price_col: Column name for price
        volume_col: Column name for volume
        figsize: Figure size (width, height) in inches
        num_panels: Number of panels to use (excluding volume)
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, list of axes)
    """
    # Group indicators by panel
    panel_indicators = {}
    
    for name, params in indicators.items():
        panel = params.get('panel', 0)  # Default to main panel
        if panel not in panel_indicators:
            panel_indicators[panel] = []
        panel_indicators[panel].append((name, params))
    
    # Determine number of panels to create
    max_panel = max(panel_indicators.keys())
    num_panels = max(num_panels, max_panel + 1)
    
    # Add volume panel if required
    include_volume = volume_col in data.columns and volume_col is not None
    if include_volume:
        num_panels += 1
        volume_panel = num_panels - 1
    
    # Create figure and axes
    height_ratios = [3]  # Main price panel is larger
    height_ratios.extend([1] * (num_panels - 1))  # Other panels are smaller
    
    fig, axes = plt.subplots(
        num_panels, 1,
        figsize=figsize,
        gridspec_kw={'height_ratios': height_ratios},
        sharex=True
    )
    
    # Ensure axes is a list
    if num_panels == 1:
        axes = [axes]
    
    # Plot price on main panel
    axes[0].plot(
        data.index,
        data[price_col],
        label=price_col,
        color='black',
        linewidth=1.5
    )
    
    # Add indicators to each panel
    for panel, indicators_list in panel_indicators.items():
        for name, params in indicators_list:
            # Get indicator data
            if name in data.columns:
                ind_data = data[name]
            else:
                logger.warning(f"Indicator column '{name}' not found in data")
                continue
            
            # Get plot parameters
            color = params.get('color', DEFAULT_COLORS[panel % len(DEFAULT_COLORS)])
            alpha = params.get('alpha', 0.7)
            linestyle = params.get('linestyle', '-')
            plot_type = params.get('type', 'line')
            
            # Plot based on type
            if plot_type == 'line':
                axes[panel].plot(
                    data.index,
                    ind_data,
                    color=color,
                    alpha=alpha,
                    linestyle=linestyle,
                    label=name
                )
            elif plot_type == 'histogram':
                axes[panel].bar(
                    data.index,
                    ind_data,
                    color=color,
                    alpha=alpha,
                    label=name
                )
            elif plot_type == 'area':
                axes[panel].fill_between(
                    data.index,
                    ind_data,
                    color=color,
                    alpha=alpha,
                    label=name
                )
            elif plot_type == 'scatter':
                axes[panel].scatter(
                    data.index,
                    ind_data,
                    color=color,
                    alpha=alpha,
                    label=name
                )
            
            # Add reference lines if specified
            if 'ref_lines' in params:
                for value, line_params in params['ref_lines'].items():
                    add_horizontal_line(
                        axes[panel],
                        value,
                        **line_params
                    )
    
    # Add volume if requested
    if include_volume:
        volume_data = data[volume_col]
        
        # Color volume bars based on price change
        colors = []
        for i in range(len(data)):
            if i > 0 and data[price_col].iloc[i] >= data[price_col].iloc[i-1]:
                colors.append('green')
            else:
                colors.append('red')
        
        # Plot volume bars
        axes[volume_panel].bar(
            data.index,
            volume_data,
            color=colors,
            alpha=0.5,
            width=0.8
        )
        
        axes[volume_panel].set_ylabel('Volume')
        
        # Format y-axis to avoid scientific notation
        axes[volume_panel].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Set labels for each panel
    axes[0].set_title('Price and Technical Indicators')
    axes[0].set_ylabel('Price')
    
    for panel, indicators_list in panel_indicators.items():
        if panel > 0:  # Skip main panel
            # Get panel label from the first indicator if available
            if indicators_list and 'label' in indicators_list[0][1]:
                axes[panel].set_ylabel(indicators_list[0][1]['label'])
            else:
                indicators_in_panel = [name for name, _ in indicators_list]
                axes[panel].set_ylabel(' / '.join(indicators_in_panel))
    
    # Format date axis (only for bottom axis)
    configure_date_axis(axes[-1])
    
    # Add legends to each panel
    for i in range(num_panels):
        if i == volume_panel and include_volume:
            continue  # Skip legend for volume panel
        
        add_legend(axes[i])
    
    plt.tight_layout()
    
    return fig, axes


#------------------------------------------------------------------------
# Statistical Visualizations
#------------------------------------------------------------------------

def plot_distribution(
    data: Union[pd.Series, np.ndarray],
    bins: int = 50,
    kde: bool = True,
    normal_overlay: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    color: str = '#1f77b4',
    title: Optional[str] = None,
    stats_box: bool = True,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot the distribution of data with optional overlays.
    
    Args:
        data: Data for distribution analysis
        bins: Number of histogram bins
        kde: Whether to include KDE overlay
        normal_overlay: Whether to overlay normal distribution
        figsize: Figure size (width, height) in inches
        color: Color for the histogram
        title: Plot title
        stats_box: Whether to include statistics box
        ax: Existing axes to plot on
        **kwargs: Additional arguments for histogram
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to Series if numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Remove NaN values
    clean_data = data.dropna()
    
    # Plot histogram
    ax.hist(
        clean_data,
        bins=bins,
        alpha=0.6,
        color=color,
        density=True,
        label='Histogram',
        **kwargs
    )
    
    # Add KDE overlay if requested
    if kde:
        sns.kdeplot(
            clean_data,
            ax=ax,
            color='navy',
            linewidth=2,
            label='KDE'
        )
    
    # Add normal distribution overlay if requested
    if normal_overlay:
        mean = clean_data.mean()
        std = clean_data.std()
        x = np.linspace(
            max(clean_data.min(), mean - 4*std),
            min(clean_data.max(), mean + 4*std),
            100
        )
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
    
    # Add statistics box if requested
    if stats_box:
        mean = clean_data.mean()
        median = clean_data.median()
        std = clean_data.std()
        skew = stats.skew(clean_data)
        kurt = stats.kurtosis(clean_data)
        
        # Create stats text
        stats_text = (
            f"Mean: {mean:.4f}\n"
            f"Median: {median:.4f}\n"
            f"Std Dev: {std:.4f}\n"
            f"Skewness: {skew:.4f}\n"
            f"Kurtosis: {kurt:.4f}"
        )
        
        # Add text box
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox={'boxstyle': 'round', 'alpha': 0.8, 'facecolor': 'white'}
        )
    
    # Set labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Distribution')
    
    # Add legend
    legend_elements = [mpatches.Patch(color=color, alpha=0.6, label='Histogram')]
    if kde:
        legend_elements.append(Line2D([0], [0], color='navy', linewidth=2, label='KDE'))
    if normal_overlay:
        legend_elements.append(Line2D([0], [0], color='red', linewidth=2, label='Normal'))
    
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    return fig, ax


def plot_qq(
    data: Union[pd.Series, np.ndarray],
    dist: str = 'norm',
    figsize: Tuple[float, float] = (8, 8),
    color: str = '#1f77b4',
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a Q-Q plot to compare data to a theoretical distribution.
    
    Args:
        data: Data for Q-Q analysis
        dist: Theoretical distribution to compare against
        figsize: Figure size (width, height) in inches
        color: Color for the points
        title: Plot title
        ax: Existing axes to plot on
        **kwargs: Additional arguments for scatter plot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to numpy array if Series
    if isinstance(data, pd.Series):
        data = data.dropna().values
    elif isinstance(data, np.ndarray):
        data = data[~np.isnan(data)]
    
    # Create Q-Q plot
    osm, osr = stats.probplot(data, dist=dist, fit=True, plot=ax)
    
    # Get fit parameters
    slope, intercept, r_value = osr
    
    # Customize plot appearance
    # Replace default points with our own
    ax.clear()
    
    # Recreate the plot
    x = osm[0]
    y = osm[1]
    
    # Plot points
    ax.scatter(x, y, color=color, alpha=0.7, **kwargs)
    
    # Plot reference line
    ax.plot([min(x), max(x)], [slope * min(x) + intercept, slope * max(x) + intercept], 
           'r-', linewidth=2, label=f'R² = {r_value**2:.4f}')
    
    # Set labels and title
    dist_name = dist.capitalize() if dist in ['norm', 'exp'] else dist
    ax.set_xlabel(f'Theoretical {dist_name} Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{dist_name} Q-Q Plot')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson',
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    mask_upper: bool = False,
    title: Optional[str] = None,
    vmin: Optional[float] = -1.0,
    vmax: Optional[float] = 1.0,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a correlation matrix heatmap.
    
    Args:
        data: DataFrame with data to correlate
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size (width, height) in inches
        cmap: Colormap for heatmap
        annot: Whether to annotate cells with correlation values
        mask_upper: Whether to mask the upper triangle
        title: Plot title
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        **kwargs: Additional arguments for heatmap
        
    Returns:
        Tuple of (figure, axes)
    """
    # Calculate correlation matrix
    corr = data.corr(method=method)
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=annot,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax,
        **kwargs
    )
    
    # Set title
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{method.capitalize()} Correlation Matrix')
    
    plt.tight_layout()
    
    return fig, ax


def plot_scatter_matrix(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hist_kwds: Optional[Dict[str, Any]] = None,
    density_kwds: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[Figure, np.ndarray]:
    """
    Create a scatter matrix (pairs plot) for data exploration.
    
    Args:
        data: DataFrame with data to plot
        columns: Columns to include (None for all)
        figsize: Figure size (width, height) in inches
        hist_kwds: Keywords for histogram plots
        density_kwds: Keywords for density plots
        **kwargs: Additional arguments for scatter plots
        
    Returns:
        Tuple of (figure, array of axes)
    """
    # Select columns to include
    if columns is not None:
        plot_data = data[columns]
    else:
        plot_data = data
    
    # Determine figure size if not provided
    if figsize is None:
        n = len(plot_data.columns)
        figsize = (2 * n, 2 * n)
    
    # Set up default hist_kwds if not provided
    if hist_kwds is None:
        hist_kwds = {'bins': 20, 'alpha': 0.6}
    
    # Set up default density_kwds if not provided
    if density_kwds is None:
        density_kwds = {'alpha': 0.6}
    
    # Create scatter matrix
    axes = pd.plotting.scatter_matrix(
        plot_data,
        figsize=figsize,
        hist_kwds=hist_kwds,
        density_kwds=density_kwds,
        **kwargs
    )
    
    # Access figure from axes
    fig = plt.gcf()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axes


def plot_box(
    data: Union[pd.DataFrame, pd.Series],
    columns: Optional[List[str]] = None,
    by: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    vert: bool = True,
    notch: bool = False,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a box plot for distribution comparison.
    
    Args:
        data: Data to plot
        columns: Columns to include (if DataFrame)
        by: Column to group by
        figsize: Figure size (width, height) in inches
        vert: Whether to create vertical box plots
        notch: Whether to add notches
        title: Plot title
        ax: Existing axes to plot on
        **kwargs: Additional arguments for boxplot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert Series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    
    # Determine columns to plot
    if columns is None:
        if data.shape[1] <= 10:  # Plot all columns if 10 or fewer
            columns = data.columns.tolist()
        else:
            columns = data.columns[:10].tolist()  # Plot first 10 columns if more
            logger.warning(f"DataFrame has {data.shape[1]} columns, plotting only the first 10")
    
    # Prepare data for plotting
    if by is not None:
        # Grouped box plot
        plot_data = []
        labels = []
        
        for col in columns:
            grouped = data.groupby(by)[col]
            for group_name, group_data in grouped:
                plot_data.append(group_data.values)
                labels.append(f"{col} - {group_name}")
    else:
        # Simple box plot
        plot_data = [data[col].dropna() for col in columns]
        labels = columns
    
    # Create box plot
    ax.boxplot(
        plot_data,
        labels=labels,
        vert=vert,
        notch=notch,
        patch_artist=True,
        **kwargs
    )
    
    # Set labels and title
    if vert:
        ax.set_xlabel('Variable')
        ax.set_ylabel('Value')
    else:
        ax.set_xlabel('Value')
        ax.set_ylabel('Variable')
    
    if title is not None:
        ax.set_title(title)
    else:
        if by is not None:
            ax.set_title(f'Box Plot Grouped by {by}')
        else:
            ax.set_title('Box Plot')
    
    # Rotate x-axis labels for better readability if needed
    if len(labels) > 4 or any(len(str(label)) > 10 for label in labels):
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax


def plot_heatmap(
    data: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'viridis',
    annot: bool = True,
    cbar: bool = True,
    title: Optional[str] = None,
    transpose: bool = False,
    fmt: str = '.2f',
    linewidths: float = 0.5,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a heatmap to visualize matrix data.
    
    Args:
        data: DataFrame with matrix data
        figsize: Figure size (width, height) in inches
        cmap: Colormap for heatmap
        annot: Whether to annotate cells with values
        cbar: Whether to add a colorbar
        title: Plot title
        transpose: Whether to transpose the data
        fmt: String formatting code for annotations
        linewidths: Width of lines between cells
        ax: Existing axes to plot on
        **kwargs: Additional arguments for heatmap
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Transpose data if requested
    if transpose:
        plot_data = data.T
    else:
        plot_data = data
    
    # Create heatmap
    sns.heatmap(
        plot_data,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=linewidths,
        cbar=cbar,
        ax=ax,
        **kwargs
    )
    
    # Set title
    if title is not None:
        ax.set_title(title)
    
    # Rotate x-axis labels for better readability if needed
    if plot_data.shape[1] > 4 or any(len(str(col)) > 10 for col in plot_data.columns):
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax


#------------------------------------------------------------------------
# Performance Visualizations
#------------------------------------------------------------------------

def plot_drawdowns(
    drawdowns: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 6),
    top_n: int = 5,
    color: str = 'red',
    alpha: float = 0.3,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Visualize drawdowns in a time series.
    
    Args:
        drawdowns: DataFrame with drawdown information
        figsize: Figure size (width, height) in inches
        top_n: Number of largest drawdowns to highlight
        color: Color for drawdown visualization
        alpha: Alpha transparency for drawdown regions
        title: Plot title
        ax: Existing axes to plot on
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot cumulative returns
    if 'cum_returns' in drawdowns.columns:
        ax.plot(
            drawdowns.index,
            drawdowns['cum_returns'],
            color='black',
            linewidth=1.5,
            label='Cumulative Returns'
        )
    
    # Ensure the DataFrame has the necessary column
    if 'drawdown' not in drawdowns.columns:
        logger.warning("Drawdown DataFrame must contain 'drawdown' column")
        return fig, ax
    
    # Plot drawdown underwater chart
    ax.fill_between(
        drawdowns.index,
        0,
        drawdowns['drawdown'],
        where=drawdowns['drawdown'] < 0,
        color=color,
        alpha=alpha,
        label='Drawdown'
    )
    
    # Find and highlight the top N largest drawdowns
    if 'drawdown_group' in drawdowns.columns and top_n > 0:
        # Find the lowest point in each drawdown period
        grouped = drawdowns[drawdowns['is_drawdown']].groupby('drawdown_group')
        min_points = grouped['drawdown'].idxmin()
        
        # Get the top N largest drawdowns
        top_drawdowns = drawdowns.loc[min_points].sort_values('drawdown').head(top_n)
        
        # Extract start and end of each drawdown
        for group_id in top_drawdowns.index:
            group_data = drawdowns[drawdowns['drawdown_group'] == drawdowns.loc[group_id, 'drawdown_group']]
            
            # Only process if we have data
            if not group_data.empty:
                start_date = group_data.index[0]
                end_date = group_data.index[-1]
                max_dd = group_data['drawdown'].min()
                max_dd_date = group_data['drawdown'].idxmin()
                
                # Annotate maximum drawdown point
                ax.annotate(
                    f"{max_dd:.1%}",
                    xy=(max_dd_date, max_dd),
                    xytext=(0, -20),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    color='black',
                    arrowprops={'arrowstyle': '->', 'color': 'black'}
                )
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add horizontal line at zero
    add_horizontal_line(ax, 0)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns / Drawdown')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Drawdown Analysis')
    
    # Format date axis
    configure_date_axis(ax)
    
    # Add legend
    add_legend(ax)
    
    plt.tight_layout()
    
    return fig, ax


def plot_regime_overlay(
    data: pd.DataFrame,
    regimes: pd.Series,
    price_col: str = 'close',
    title: Optional[str] = None,
    regime_alpha: float = 0.2,
    regime_colors: Optional[Dict[int, str]] = None,
    regime_labels: Optional[Dict[int, str]] = None,
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a price chart with regime overlay.
    
    Args:
        data: Price data
        regimes: Series with regime identifiers
        price_col: Column name for price data
        title: Plot title
        regime_alpha: Alpha transparency for regime backgrounds
        regime_colors: Dictionary mapping regime IDs to colors
        regime_labels: Dictionary mapping regime IDs to labels
        figsize: Figure size (width, height) in inches
        ax: Existing axes to plot on
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Align data and regimes
    aligned_data = data[price_col].copy()
    aligned_regimes = regimes.copy()
    
    if aligned_regimes.index.equals(aligned_data.index):
        pass  # Already aligned
    else:
        # Try to reindex regimes to match data
        try:
            aligned_regimes = aligned_regimes.reindex(aligned_data.index, method='ffill')
        except Exception as e:
            logger.warning(f"Could not align regimes with data: {e}")
            return fig, ax
    
    # Plot price data
    ax.plot(
        aligned_data.index,
        aligned_data.values,
        linewidth=1.5,
        color='black',
        label=price_col,
        **kwargs
    )
    
    # Get unique regimes
    unique_regimes = aligned_regimes.unique()
    
    # Default regime colors and labels if not provided
    if regime_colors is None:
        cmap = cm.get_cmap('tab10')
        regime_colors = {regime: mcolors.rgb2hex(cmap(i % 10)) for i, regime in enumerate(unique_regimes)}
    
    if regime_labels is None:
        regime_labels = {regime: f"Regime {regime}" for regime in unique_regimes}
    
    # Find regime change points
    regime_changes = (aligned_regimes != aligned_regimes.shift(1)).astype(int)
    change_points = aligned_regimes.index[regime_changes == 1].tolist()
    
    # Add the first and last points
    if len(aligned_regimes) > 0:
        regime_periods = [aligned_regimes.index[0]] + change_points + [aligned_regimes.index[-1]]
    else:
        regime_periods = []
    
    # Create colored background for each regime
    legend_patches = []
    for i in range(len(regime_periods) - 1):
        start = regime_periods[i]
        end = regime_periods[i + 1]
        
        # Get regime during this period
        regime = aligned_regimes.loc[start]
        
        # Get color and label for this regime
        color = regime_colors.get(regime, 'gray')
        label = regime_labels.get(regime, f"Regime {regime}")
        
        # Add background
        ax.axvspan(
            start, end,
            alpha=regime_alpha,
            color=color,
            label=f"_nolegend_{label}"  # Avoid duplicate labels
        )
        
        # Create patch for legend
        patch = mpatches.Patch(
            color=color,
            alpha=regime_alpha,
            label=label
        )
        legend_patches.append(patch)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Price with Regime Overlay')
    
    # Format date axis
    configure_date_axis(ax)
    
    # Add legend with price and regime patches
    line_handle, = ax.plot([], [], color='black', label=price_col)
    handles = [line_handle] + legend_patches
    ax.legend(handles=handles, loc='best')
    
    plt.tight_layout()
    
    return fig, ax


def plot_signals(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    price_col: str = 'close',
    signal_col: str = 'signal',
    value_col: Optional[str] = 'value',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    buy_marker: str = '^',
    sell_marker: str = 'v',
    buy_color: str = 'green',
    sell_color: str = 'red',
    marker_size: int = 100,
    show_returns: bool = False,
    bottom_panel_ratio: float = 0.2,
    **kwargs
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Create a price chart with buy/sell signals.
    
    Args:
        data: Price data
        signals: Signal data
        price_col: Column name for price data
        signal_col: Column name for signal type (+1, -1, 0)
        value_col: Column name for signal value/strength
        title: Plot title
        figsize: Figure size (width, height) in inches
        buy_marker: Marker style for buy signals
        sell_marker: Marker style for sell signals
        buy_color: Color for buy signals
        sell_color: Color for sell signals
        marker_size: Size of signal markers
        show_returns: Whether to show returns in a lower panel
        bottom_panel_ratio: Height ratio of bottom panel
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Determine if we need two panels
    if show_returns:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=figsize,
            gridspec_kw={'height_ratios': [1-bottom_panel_ratio, bottom_panel_ratio]},
            sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    
    # Plot price data
    ax1.plot(
        data.index,
        data[price_col],
        linewidth=1.5,
        color='black',
        **kwargs
    )
    
    # Extract buy and sell signals
    if not signals.empty:
        # For buy signals
        buy_signals = signals[signals[signal_col] > 0]
        if not buy_signals.empty:
            buy_values = []
            for idx in buy_signals.index:
                # Find the corresponding price
                if idx in data.index:
                    buy_values.append(data.loc[idx, price_col])
                else:
                    # Find closest date
                    buy_values.append(data[price_col].iloc[data.index.get_indexer([idx], method='nearest')[0]])
            
            # Plot buy markers
            ax1.scatter(
                buy_signals.index,
                buy_values,
                color=buy_color,
                marker=buy_marker,
                s=marker_size,
                label='Buy Signal'
            )
        
        # For sell signals
        sell_signals = signals[signals[signal_col] < 0]
        if not sell_signals.empty:
            sell_values = []
            for idx in sell_signals.index:
                # Find the corresponding price
                if idx in data.index:
                    sell_values.append(data.loc[idx, price_col])
                else:
                    # Find closest date
                    sell_values.append(data[price_col].iloc[data.index.get_indexer([idx], method='nearest')[0]])
            
            # Plot sell markers
            ax1.scatter(
                sell_signals.index,
                sell_values,
                color=sell_color,
                marker=sell_marker,
                s=marker_size,
                label='Sell Signal'
            )
    
    # Show signal values in bottom panel if requested
    if show_returns and value_col in signals.columns:
        # Plot signal values/strength
        ax2.bar(
            signals.index,
            signals[value_col],
            color=signals[signal_col].apply(
                lambda x: buy_color if x > 0 else (sell_color if x < 0 else 'gray')
            ),
            alpha=0.7
        )
        
        # Add reference line at zero
        add_horizontal_line(ax2, 0)
        
        # Set y-label for bottom panel
        ax2.set_ylabel('Signal Strength')
    
    # Set labels and title
    ax1.set_ylabel('Price')
    
    if title is not None:
        ax1.set_title(title)
    else:
        ax1.set_title('Price Chart with Trading Signals')
    
    # Format date axis
    if ax2 is not None:
        configure_date_axis(ax2)
        ax2.set_xlabel('Date')
    else:
        configure_date_axis(ax1)
        ax1.set_xlabel('Date')
    
    # Add legend
    add_legend(ax1)
    
    plt.tight_layout()
    
    if ax2 is not None:
        return fig, [ax1, ax2]
    else:
        return fig, ax1


def plot_performance_metrics(
    metrics: Dict[str, float],
    figsize: Tuple[float, float] = (10, 6),
    color: str = '#1f77b4',
    horizontal: bool = True,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a bar chart of performance metrics.
    
    Args:
        metrics: Dictionary of metrics to plot
        figsize: Figure size (width, height) in inches
        color: Bar color
        horizontal: Whether to create horizontal bars
        title: Plot title
        ax: Existing axes to plot on
        **kwargs: Additional arguments for bar plot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Sort metrics by value
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_metrics]
    values = [item[1] for item in sorted_metrics]
    
    # Create the bar chart
    if horizontal:
        bars = ax.barh(
            labels,
            values,
            color=color,
            alpha=0.7,
            **kwargs
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01  # Slightly to the right of the bar
            
            # Format label based on value
            if abs(width) >= 100:
                label = f"{width:.0f}"
            elif abs(width) >= 10:
                label = f"{width:.1f}"
            elif abs(width) >= 1:
                label = f"{width:.2f}"
            else:
                label = f"{width:.3f}"
                
            ax.text(
                label_x_pos,
                bar.get_y() + bar.get_height()/2,
                label,
                va='center'
            )
        
        # Set labels
        ax.set_xlabel('Value')
        ax.set_ylabel('Metric')
    else:
        bars = ax.bar(
            labels,
            values,
            color=color,
            alpha=0.7,
            **kwargs
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label_y_pos = height * 1.01  # Slightly above the bar
            
            # Format label based on value
            if abs(height) >= 100:
                label = f"{height:.0f}"
            elif abs(height) >= 10:
                label = f"{height:.1f}"
            elif abs(height) >= 1:
                label = f"{height:.2f}"
            else:
                label = f"{height:.3f}"
                
            ax.text(
                bar.get_x() + bar.get_width()/2,
                label_y_pos,
                label,
                ha='center',
                va='bottom'
            )
        
        # Set labels
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add horizontal line at zero
    add_horizontal_line(ax, 0)
    
    # Set title
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Performance Metrics')
    
    plt.tight_layout()
    
    return fig, ax


def plot_rolling_metrics(
    metrics: Dict[str, pd.Series],
    window: int = 252,
    figsize: Tuple[float, float] = (12, 8),
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    grid: bool = True,
    legend_loc: str = 'best',
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot rolling performance metrics over time.
    
    Args:
        metrics: Dictionary mapping metric names to time series
        window: Rolling window size
        figsize: Figure size (width, height) in inches
        colors: List of colors for each metric
        title: Plot title
        grid: Whether to show grid
        legend_loc: Legend location
        **kwargs: Additional arguments for plotting
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        colors = DEFAULT_COLORS
    
    # Ensure all metrics have the same index
    all_indices = set()
    for series in metrics.values():
        all_indices = all_indices.union(set(series.index))
    
    common_index = sorted(list(all_indices))
    
    # Plot each metric
    for i, (name, series) in enumerate(metrics.items()):
        # Reindex series to common index
        aligned_series = series.reindex(common_index)
        
        # Calculate rolling metric
        rolling_metric = aligned_series.rolling(window=window, min_periods=window//2).mean()
        
        # Plot the rolling metric
        ax.plot(
            rolling_metric.index,
            rolling_metric.values,
            label=f"{name} ({window}-day)",
            color=colors[i % len(colors)],
            **kwargs
        )
    
    # Add horizontal line at zero if appropriate
    min_val = min([series.min() for series in metrics.values()])
    max_val = max([series.max() for series in metrics.values()])
    if min_val < 0 < max_val:
        add_horizontal_line(ax, 0)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Metric Value')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'Rolling Performance Metrics ({window}-day window)')
    
    # Add grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format date axis
    configure_date_axis(ax)
    
    # Add legend
    add_legend(ax, loc=legend_loc)
    
    plt.tight_layout()
    
    return fig, ax