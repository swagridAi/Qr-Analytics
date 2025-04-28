"""
Data Preparation Utilities

This module provides common data preparation, cleaning, and feature engineering
functions used across all analytics modules. It ensures consistent preprocessing
approaches and reduces code duplication.

Features:
- DataFrame validation and cleaning
- Return calculation with multiple methodologies
- Feature engineering for time series data
- Outlier detection and handling
- Cross-validation for time series
- Resampling and alignment utilities

Usage:
    ```python
    from quant_research.analytics.common.data_prep import (
        ensure_datetime_index,
        calculate_returns,
        add_technical_features
    )
    
    # Prepare DataFrame
    df = ensure_datetime_index(df)
    
    # Calculate returns
    df['returns'] = calculate_returns(df['close'], method='log')
    
    # Add common technical features
    df = add_technical_features(df, window=20)
    ```
"""

# Standard library imports
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Local imports - now importing from data_utils
from quant_research.analytics.common.data_utils import (
    ensure_datetime_index,
    validate_ohlc,
    filter_time_range,
    detect_frequency,
    resample_data,
    align_dataframes,
    calculate_returns,
    calculate_cumulative_returns,
    calculate_drawdowns,
    calculate_momentum,
    calculate_volatility,
    detect_outliers,
    handle_outliers,
    normalize_data,
    calculate_zscore,
    add_lagged_features,
    add_difference_features,
    add_rolling_features,
    time_series_split,
    expanding_window_split,
    walk_forward_split,
    validate_missing_data
)

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data_prep")

#------------------------------------------------------------------------
# Technical Indicators
#------------------------------------------------------------------------

def add_technical_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
    window: int = 20,
    include: List[str] = None
) -> pd.DataFrame:
    """
    Add common technical indicators as features.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        volume_col: Column name for volume data
        window: Window size for indicators
        include: List of indicators to include (defaults to all)
        
    Returns:
        DataFrame with added technical features
        
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # All available indicators
    all_indicators = [
        'returns', 'ma', 'ema', 'momentum', 'volatility',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_width'
    ]
    
    # If volume is available, add volume indicators
    if volume_col in df.columns:
        all_indicators.extend(['volume_ma', 'volume_ratio', 'obv'])
    
    # Filter indicators if specified
    indicators = all_indicators if include is None else [ind for ind in include if ind in all_indicators]
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate returns if needed for indicators or explicitly requested
    if 'returns' in indicators or any(ind in indicators for ind in ['momentum', 'volatility', 'rsi']):
        result['returns'] = calculate_returns(result[price_col], method='pct')
        
        # Remove from indicators if only needed as input, not explicitly requested
        if 'returns' not in indicators:
            indicators.append('returns')
    
    # Calculate requested indicators
    if 'ma' in indicators:
        result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()
    
    if 'ema' in indicators:
        result[f'ema_{window}'] = result[price_col].ewm(span=window).mean()
    
    if 'momentum' in indicators:
        result[f'momentum_{window}'] = calculate_momentum(result[price_col], window=window)
    
    if 'volatility' in indicators:
        result[f'volatility_{window}'] = calculate_volatility(result['returns'], window=window)
    
    if 'rsi' in indicators:
        # Calculate gains and losses
        delta = result[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    if 'macd' in indicators:
        # Calculate MACD components
        fast_ema = result[price_col].ewm(span=12).mean()
        slow_ema = result[price_col].ewm(span=26).mean()
        result['macd'] = fast_ema - slow_ema
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
    
    if any(bb in indicators for bb in ['bb_upper', 'bb_lower', 'bb_width']):
        # Calculate Bollinger Bands
        ma = result[price_col].rolling(window=window).mean()
        std = result[price_col].rolling(window=window).std()
        
        if 'bb_upper' in indicators:
            result[f'bb_upper_{window}'] = ma + (2 * std)
        
        if 'bb_lower' in indicators:
            result[f'bb_lower_{window}'] = ma - (2 * std)
        
        if 'bb_width' in indicators:
            upper = ma + (2 * std)
            lower = ma - (2 * std)
            result[f'bb_width_{window}'] = (upper - lower) / ma
    
    # Volume-based indicators
    if volume_col in df.columns:
        if 'volume_ma' in indicators:
            result[f'volume_ma_{window}'] = result[volume_col].rolling(window=window).mean()
        
        if 'volume_ratio' in indicators:
            result[f'volume_ratio_{window}'] = result[volume_col] / result[volume_col].rolling(window=window).mean()
        
        if 'obv' in indicators:
            # Calculate On-Balance Volume
            result['obv'] = np.where(
                result[price_col] > result[price_col].shift(1),
                result[volume_col],
                np.where(
                    result[price_col] < result[price_col].shift(1),
                    -result[volume_col],
                    0
                )
            ).cumsum()
    
    return result


def annualize_returns(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252,
    compound: bool = True
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Annualize returns based on frequency.
    
    Args:
        returns: Return data (can be Series, DataFrame, or scalar)
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.)
        compound: Whether to use compound annualization (geometric) or simple (arithmetic)
        
    Returns:
        Annualized returns (same type as input)
    """
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        if compound:
            if isinstance(returns, pd.Series):
                # For Series, we can directly apply the formula
                return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
            else:
                # For DataFrames, apply to each column
                return returns.apply(lambda col: (1 + col).prod() ** (periods_per_year / len(col)) - 1)
        else:
            # Simple annualization (arithmetic)
            return returns.mean() * periods_per_year
    else:
        # Assume returns is a scalar representing a periodic return
        if compound:
            return (1 + returns) ** periods_per_year - 1
        else:
            return returns * periods_per_year