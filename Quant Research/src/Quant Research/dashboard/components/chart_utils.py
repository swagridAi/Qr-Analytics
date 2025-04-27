"""
Utility functions for chart components in the Quant Research dashboard.

This module provides shared utility functions for data processing,
chart element generation, and formatting that are used across
multiple visualization components.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    if df is None or df.empty:
        return False, required_columns
        
    missing = [col for col in required_columns if col not in df.columns]
    
    return len(missing) == 0, missing


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    date_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Filter a DataFrame by date range.
    
    Args:
        df: DataFrame to filter
        start_date: Start date for filtering
        end_date: End date for filtering
        date_column: Name of date column
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or date_column not in df.columns:
        return df
        
    result = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column])
    
    # Apply start date filter
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        result = result[result[date_column] >= start_date]
    
    # Apply end date filter
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        result = result[result[date_column] <= end_date]
    
    return result


def filter_by_symbols(
    df: pd.DataFrame,
    symbols: List[str],
    symbol_column: str = 'symbol'
) -> pd.DataFrame:
    """
    Filter a DataFrame by list of symbols.
    
    Args:
        df: DataFrame to filter
        symbols: List of symbols to include
        symbol_column: Name of symbol column
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or symbol_column not in df.columns or not symbols:
        return df
        
    return df[df[symbol_column].isin(symbols)]


def calculate_moving_average(
    series: pd.Series,
    window: int,
    method: str = 'sma'
) -> pd.Series:
    """
    Calculate moving average of a series.
    
    Args:
        series: Data series
        window: Window size for moving average
        method: Method ('sma', 'ema', 'wma')
    
    Returns:
        Series with moving average values
    """
    if series.empty or window <= 0:
        return pd.Series(index=series.index)
        
    if method.lower() == 'ema':
        return series.ewm(span=window, adjust=False).mean()
    elif method.lower() == 'wma':
        # Weighted moving average with linearly decreasing weights
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(
            lambda x: np.sum(weights * x) / np.sum(weights),
            raw=True
        )
    else:  # Default to SMA
        return series.rolling(window=window).mean()


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands for a series.
    
    Args:
        series: Data series
        window: Window size for moving average
        num_std: Number of standard deviations
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    if series.empty or window <= 0:
        return (
            pd.Series(index=series.index),
            pd.Series(index=series.index),
            pd.Series(index=series.index)
        )
        
    middle_band = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return middle_band, upper_band, lower_band


def calculate_rsi(
    series: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        series: Data series
        window: RSI period
    
    Returns:
        Series with RSI values
    """
    if series.empty or window <= 0:
        return pd.Series(index=series.index)
        
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_drawdown(
    equity_series: pd.Series
) -> Tuple[pd.Series, pd.Series, float, int]:
    """
    Calculate drawdown series and metrics.
    
    Args:
        equity_series: Equity curve series
    
    Returns:
        Tuple of (drawdown_series, underwater_series, max_drawdown, max_drawdown_duration)
    """
    if equity_series.empty:
        return pd.Series(), pd.Series(), 0.0, 0
        
    # Calculate rolling maximum
    rolling_max = equity_series.cummax()
    
    # Calculate drawdown series
    drawdown_series = (equity_series - rolling_max) / rolling_max
    
    # Calculate underwater series (equity / high watermark)
    underwater_series = equity_series / rolling_max
    
    # Calculate maximum drawdown
    max_drawdown = drawdown_series.min()
    
    # Calculate maximum drawdown duration
    if max_drawdown < 0:
        # Find the peak before the max drawdown
        max_dd_idx = drawdown_series.idxmin()
        
        # Find the last peak before the max drawdown
        peak_idx = rolling_max.loc[:max_dd_idx].idxmax()
        
        # Find recovery (if any)
        recovery_idx = None
        after_dd = drawdown_series.loc[max_dd_idx:]
        recovery = after_dd[after_dd >= 0]
        
        if not recovery.empty:
            recovery_idx = recovery.index[0]
            # Calculate duration from peak to recovery
            if hasattr(recovery_idx, 'days') and hasattr(peak_idx, 'days'):
                max_drawdown_duration = (recovery_idx - peak_idx).days
            else:
                max_drawdown_duration = len(equity_series.loc[peak_idx:recovery_idx])
        else:
            # No recovery, calculate from peak to last data point
            if hasattr(equity_series.index[-1], 'days') and hasattr(peak_idx, 'days'):
                max_drawdown_duration = (equity_series.index[-1] - peak_idx).days
            else:
                max_drawdown_duration = len(equity_series.loc[peak_idx:])
    else:
        max_drawdown_duration = 0
    
    return drawdown_series, underwater_series, max_drawdown, max_drawdown_duration


def calculate_returns_metrics(
    returns_series: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate common return-based performance metrics.
    
    Args:
        returns_series: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
    
    Returns:
        Dictionary of calculated metrics
    """
    if returns_series.empty:
        return {
            'total_return': None,
            'annualized_return': None,
            'volatility': None,
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'skewness': None,
            'kurtosis': None
        }
    
    # Remove NaN values
    returns = returns_series.dropna()
    
    if len(returns) < 2:
        return {
            'total_return': returns.sum() if len(returns) == 1 else None,
            'annualized_return': None,
            'volatility': None,
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'skewness': None,
            'kurtosis': None
        }
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    # Volatility (annualized)
    volatility = returns.std() * math.sqrt(periods_per_year)
    
    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Sharpe ratio
    excess_returns = returns - daily_rf
    sharpe_ratio = excess_returns.mean() / returns.std() * math.sqrt(periods_per_year) if returns.std() > 0 else None
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    sortino_ratio = (
        excess_returns.mean() / downside_returns.std() * math.sqrt(periods_per_year)
        if len(downside_returns) > 0 and downside_returns.std() > 0
        else None
    )
    
    # Higher moments
    skewness = returns.skew() if len(returns) > 2 else None
    kurtosis = returns.kurtosis() if len(returns) > 3 else None
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def calculate_value_at_risk(
    returns_series: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> Tuple[float, float]:
    """
    Calculate Value at Risk and Expected Shortfall.
    
    Args:
        returns_series: Series of returns
        confidence_level: Confidence level (0-1)
        method: VaR method ('historical', 'parametric', 'montecarlo')
    
    Returns:
        Tuple of (VaR, Expected Shortfall)
    """
    if returns_series.empty:
        return None, None
    
    # Remove NaN values
    returns = returns_series.dropna()
    
    if len(returns) < 10:  # Need sufficient data for reliable VaR
        return None, None
    
    if method == 'parametric':
        # Parametric VaR (assuming normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(returns.mean() + z_score * returns.std())
        
        # Expected Shortfall (Conditional VaR)
        z_es = stats.norm.pdf(z_score) / (1 - confidence_level)
        es = -(returns.mean() + z_es * returns.std())
        
    elif method == 'montecarlo':
        # Monte Carlo VaR
        np.random.seed(42)  # For reproducibility
        n_simulations = 10000
        simulated_returns = np.random.normal(
            loc=returns.mean(),
            scale=returns.std(),
            size=n_simulations
        )
        var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        es = -simulated_returns[simulated_returns <= -var].mean()
        
    else:  # Default to historical method
        # Historical VaR
        var_percentile = 100 * (1 - confidence_level)
        var = -np.percentile(returns, var_percentile)
        
        # Expected Shortfall
        es_returns = returns[returns <= -var]
        es = -es_returns.mean() if len(es_returns) > 0 else var
    
    return var, es


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """
    Create an empty figure with a message.
    
    Args:
        message: Message to display
    
    Returns:
        Empty plotly figure with message
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        height=400,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig


def add_axis_title(
    fig: go.Figure,
    title: str,
    axis: str = 'x',
    row: int = 1,
    col: int = 1
) -> None:
    """
    Add title to an axis in a subplot.
    
    Args:
        fig: Figure to update
        title: Title text
        axis: Axis to update ('x' or 'y')
        row: Row of subplot
        col: Column of subplot
    """
    if fig is None:
        return
        
    if axis.lower() == 'x':
        fig.update_xaxes(title_text=title, row=row, col=col)
    else:
        fig.update_yaxes(title_text=title, row=row, col=col)


def add_reference_line(
    fig: go.Figure,
    value: float,
    axis: str = 'y',
    line_dash: str = 'dash',
    line_color: str = 'gray',
    line_width: float = 1.0,
    row: int = 1,
    col: int = 1
) -> None:
    """
    Add a reference line to a subplot.
    
    Args:
        fig: Figure to update
        value: Line value
        axis: Axis for line ('x' or 'y')
        line_dash: Line dash style
        line_color: Line color
        line_width: Line width
        row: Row of subplot
        col: Column of subplot
    """
    if fig is None:
        return
        
    line_props = dict(
        line_dash=line_dash,
        line_color=line_color,
        line_width=line_width,
        row=row,
        col=col
    )
    
    if axis.lower() == 'x':
        fig.add_vline(x=value, **line_props)
    else:
        fig.add_hline(y=value, **line_props)


def add_annotation(
    fig: go.Figure,
    text: str,
    x: Union[float, str] = 0.5,
    y: Union[float, str] = 0.5,
    xref: str = "paper",
    yref: str = "paper",
    showarrow: bool = False,
    bgcolor: Optional[str] = "rgba(255, 255, 255, 0.7)",
    row: Optional[int] = None,
    col: Optional[int] = None
) -> None:
    """
    Add an annotation to a figure.
    
    Args:
        fig: Figure to update
        text: Annotation text
        x: X position
        y: Y position
        xref: X reference ('paper' or 'x')
        yref: Y reference ('paper' or 'y')
        showarrow: Whether to show arrow
        bgcolor: Background color (None for transparent)
        row: Row of subplot (None for global annotation)
        col: Column of subplot (None for global annotation)
    """
    if fig is None:
        return
        
    annotation = dict(
        text=text,
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        showarrow=showarrow,
        font=dict(size=10),
        align="center",
        borderwidth=1,
        borderpad=4
    )
    
    if bgcolor:
        annotation["bgcolor"] = bgcolor
        annotation["bordercolor"] = "gray"
    
    if row is not None and col is not None:
        annotation["row"] = row
        annotation["col"] = col
    
    fig.add_annotation(**annotation)