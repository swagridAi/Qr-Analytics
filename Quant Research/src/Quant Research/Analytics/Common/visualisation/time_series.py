"""
Time Series Visualization Module

This module provides functions for creating various types of time series plots,
including line plots, return plots, and area charts. These functions are designed
to work with financial time series data.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Local imports
from .utils import (
    configure_date_axis,
    add_horizontal_line,
    add_annotations,
    add_legend,
    format_y_axis
)

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.visualization.time_series")

# Default color palette
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
    highlight_regions: Optional[List[Tuple[pd.Timestamp, pd.Timestamp, str, float]]] = None,
    annotations: Optional[List[Tuple[pd.Timestamp, float, str]]] = None,
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