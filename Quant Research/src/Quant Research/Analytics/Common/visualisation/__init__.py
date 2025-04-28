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

# Import utility functions
from .utils import (
    configure_date_axis,
    add_horizontal_line,
    add_vertical_line,
    add_annotations,
    add_legend,
    format_y_axis,
    save_figure,
    create_color_map,
    create_multi_panel
)

# Import time series plots
from .time_series import (
    plot_time_series,
    plot_returns,
    plot_dual_axis,
    plot_area
)

# Import financial charts
from .financial_charts import (
    plot_candlestick,
    plot_ohlc,
    plot_volume_profile,
    plot_technical_indicators
)

# Import statistical visualizations
from .statistical import (
    plot_distribution,
    plot_qq,
    plot_correlation_matrix,
    plot_scatter_matrix,
    plot_box,
    plot_heatmap
)

# Import performance visualizations
from .performance import (
    plot_drawdowns,
    plot_regime_overlay,
    plot_signals,
    plot_performance_metrics,
    plot_rolling_metrics
)

# Define what's available via `from visualization import *`
__all__ = [
    # Utility functions
    'configure_date_axis',
    'add_horizontal_line',
    'add_vertical_line',
    'add_annotations',
    'add_legend',
    'format_y_axis',
    'save_figure',
    'create_color_map',
    'create_multi_panel',
    
    # Time series plots
    'plot_time_series',
    'plot_returns',
    'plot_dual_axis',
    'plot_area',
    
    # Financial charts
    'plot_candlestick',
    'plot_ohlc',
    'plot_volume_profile',
    'plot_technical_indicators',
    
    # Statistical visualizations
    'plot_distribution',
    'plot_qq',
    'plot_correlation_matrix',
    'plot_scatter_matrix',
    'plot_box',
    'plot_heatmap',
    
    # Performance visualizations
    'plot_drawdowns',
    'plot_regime_overlay',
    'plot_signals',
    'plot_performance_metrics',
    'plot_rolling_metrics'
]