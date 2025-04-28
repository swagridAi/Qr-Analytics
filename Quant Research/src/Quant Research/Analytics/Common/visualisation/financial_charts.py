"""
Financial Charts Visualization Module

This module provides functions for creating financial charts such as candlestick charts,
OHLC (Open-High-Low-Close) charts, volume profiles, and technical indicators.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Local imports
from .utils import (
    configure_date_axis,
    add_horizontal_line,
    add_legend,
    format_y_axis
)

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.visualization.financial_charts")

# Default color palette
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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