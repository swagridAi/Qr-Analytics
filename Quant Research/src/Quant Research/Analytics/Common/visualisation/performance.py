"""
Performance Visualization Module

This module provides functions for visualizing financial performance metrics,
such as drawdowns, trading signals, regime overlays, and performance statistics.
"""

# Standard library imports
import logging
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

# Local imports
from .utils import (
    configure_date_axis,
    add_horizontal_line,
    add_annotations,
    add_legend
)

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.visualization.performance")

# Default color palette
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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