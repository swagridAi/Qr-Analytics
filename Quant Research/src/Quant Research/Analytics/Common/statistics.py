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

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data_prep")

#------------------------------------------------------------------------
# DataFrame Validation and Preparation
#------------------------------------------------------------------------

def ensure_datetime_index(
    df: pd.DataFrame, 
    timestamp_col: str = 'timestamp',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index, converting if necessary.
    
    Args:
        df: Input DataFrame
        timestamp_col: Column name to use for timestamp if not indexed
        inplace: Whether to modify the DataFrame in place
        
    Returns:
        DataFrame with datetime index
        
    Raises:
        ValueError: If no timestamp column is found
    """
    if not inplace:
        df = df.copy()
        
    # Check if already has datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        return df
        
    # Try to set index from timestamp column
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        return df.set_index(timestamp_col)
    
    # Try to find any datetime-like column
    datetime_cols = [
        col for col in df.columns 
        if any(time_kw in col.lower() for time_kw in ['time', 'date', 'dt', 'timestamp'])
    ]
    
    if datetime_cols:
        potential_col = datetime_cols[0]
        logger.warning(f"No '{timestamp_col}' found, using '{potential_col}' as timestamp")
        df[potential_col] = pd.to_datetime(df[potential_col])
        return df.set_index(potential_col)
    
    raise ValueError(
        f"No timestamp column found in DataFrame. Expected '{timestamp_col}' or "
        f"column with 'time', 'date', 'dt', 'timestamp' in name."
    )


def validate_ohlc(
    df: pd.DataFrame, 
    required_cols: List[str] = None,
    rename_map: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Validate and standardize OHLC data format.
    
    Args:
        df: Input DataFrame with price data
        required_cols: List of required columns (default is ['close'])
        rename_map: Dictionary mapping from original column names to standardized ones
        
    Returns:
        Standardized DataFrame with validated columns
        
    Raises:
        ValueError: If required columns are missing and can't be inferred
    """
    # Default to requiring only 'close'
    if required_cols is None:
        required_cols = ['close']
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # List of standard column names
    standard_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Common alternative column names
    alt_names = {
        'open': ['o', 'Open', 'OPEN', 'opening_price', 'price_open'],
        'high': ['h', 'High', 'HIGH', 'max_price', 'price_high', 'highest'],
        'low': ['l', 'Low', 'LOW', 'min_price', 'price_low', 'lowest'],
        'close': ['c', 'Close', 'CLOSE', 'price', 'last', 'price_close', 'closing_price'],
        'volume': ['v', 'Volume', 'VOLUME', 'vol', 'size', 'quantity']
    }
    
    # Apply custom rename map first if provided
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Try to infer column names if not in standard format
    for std_col in standard_cols:
        if std_col not in df.columns:
            found = False
            # Try alternative names
            for alt in alt_names[std_col]:
                if alt in df.columns:
                    df[std_col] = df[alt]
                    found = True
                    break
            
            if not found and std_col in required_cols:
                raise ValueError(f"Required column '{std_col}' not found in DataFrame")
    
    # Validate data types
    for col in [c for c in standard_cols if c in df.columns]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is not numeric. Attempting to convert.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for NaN values in required columns
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            pct_nan = 100 * nan_count / len(df)
            logger.warning(f"Column '{col}' has {nan_count} NaN values ({pct_nan:.2f}%)")
    
    return df


def filter_time_range(
    df: pd.DataFrame,
    start_date: Union[str, datetime, pd.Timestamp] = None,
    end_date: Union[str, datetime, pd.Timestamp] = None,
    inclusive: str = 'both'
) -> pd.DataFrame:
    """
    Filter DataFrame to a specific time range.
    
    Args:
        df: Input DataFrame with datetime index
        start_date: Start date for the filter
        end_date: End date for the filter
        inclusive: Whether to include start/end in the range ('both', 'left', 'right', 'neither')
        
    Returns:
        Filtered DataFrame
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for time filtering")
    
    # Convert string dates to timestamps
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if end_date is not None and isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    # Default to entire range if nothing specified
    if start_date is None and end_date is None:
        return df
    
    # Create the mask based on the specified range
    if start_date is not None and end_date is not None:
        if inclusive == 'both':
            mask = (df.index >= start_date) & (df.index <= end_date)
        elif inclusive == 'left':
            mask = (df.index >= start_date) & (df.index < end_date)
        elif inclusive == 'right':
            mask = (df.index > start_date) & (df.index <= end_date)
        else:  # 'neither'
            mask = (df.index > start_date) & (df.index < end_date)
    elif start_date is not None:
        mask = df.index >= start_date
    else:  # end_date is not None
        mask = df.index <= end_date
    
    # Apply the filter
    filtered_df = df[mask]
    
    # Log the filtering results
    logger.debug(
        f"Filtered from {len(df)} to {len(filtered_df)} rows "
        f"({100 * len(filtered_df) / len(df):.1f}% retained)"
    )
    
    return filtered_df


def detect_frequency(df: pd.DataFrame) -> str:
    """
    Detect the frequency of time series data.
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        Inferred frequency as pandas frequency string
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index to detect frequency")
    
    # Try pandas infer_freq first
    freq = pd.infer_freq(df.index)
    
    if freq is not None:
        return freq
    
    # Calculate time deltas and get the most common
    if len(df) > 1:
        # Get time differences
        time_diffs = df.index.to_series().diff().dropna()
        
        if not time_diffs.empty:
            # Find the most common difference
            most_common_diff = time_diffs.value_counts().idxmax()
            
            # Convert to pandas frequency string (approximate)
            seconds = most_common_diff.total_seconds()
            
            if seconds < 60:
                return f"{int(seconds)}S"
            elif seconds < 3600:
                return f"{int(seconds/60)}T"
            elif seconds < 86400:
                return f"{int(seconds/3600)}H"
            elif 86400 <= seconds < 604800:
                return f"{int(seconds/86400)}D"
            elif 604800 <= seconds < 2592000:
                return f"{int(seconds/604800)}W"
            else:
                return f"{int(seconds/2592000)}M"
    
    logger.warning("Could not infer frequency from data")
    return None


def resample_data(
    df: pd.DataFrame, 
    freq: str,
    agg_dict: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Resample time series data to a specified frequency.
    
    Args:
        df: Input DataFrame with datetime index
        freq: Target frequency (e.g., '1D', '1H', '15T')
        agg_dict: Dictionary of column:aggregation mappings
        
    Returns:
        Resampled DataFrame
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for resampling")
    
    # Default aggregation for OHLCV data
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # Filter to only include columns that exist in the DataFrame
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    # Ensure we have at least one column to aggregate
    if not agg_dict:
        raise ValueError(
            "No valid columns for aggregation. Provide an agg_dict with "
            "mappings for columns that exist in the DataFrame."
        )
    
    # Perform the resampling
    resampled = df.resample(freq).agg(agg_dict)
    
    # Log the resampling results
    logger.debug(
        f"Resampled from {len(df)} to {len(resampled)} rows "
        f"(frequency: {freq})"
    )
    
    return resampled


def align_dataframes(
    dfs: List[pd.DataFrame],
    method: str = 'outer',
    fill_method: str = None,
    freq: str = None
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to a common time index.
    
    Args:
        dfs: List of DataFrames with datetime indices
        method: Join method ('outer', 'inner', 'left', 'right')
        fill_method: Method for filling missing values (None, 'ffill', 'bfill', 'nearest')
        freq: Optional frequency to resample all DataFrames to
        
    Returns:
        List of aligned DataFrames
        
    Raises:
        TypeError: If any DataFrame doesn't have a datetime index
    """
    # Validate inputs
    for i, df in enumerate(dfs):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"DataFrame {i} must have a datetime index for alignment")
    
    # Resample to common frequency if specified
    if freq is not None:
        resampled_dfs = []
        for df in dfs:
            # Detect columns and appropriate aggregation methods
            cols = df.columns
            agg_dict = {}
            
            for col in cols:
                if any(price_kw in col.lower() for price_kw in ['open', 'first', 'start']):
                    agg_dict[col] = 'first'
                elif any(price_kw in col.lower() for price_kw in ['high', 'max']):
                    agg_dict[col] = 'max'
                elif any(price_kw in col.lower() for price_kw in ['low', 'min']):
                    agg_dict[col] = 'min'
                elif any(price_kw in col.lower() for price_kw in ['close', 'last', 'end']):
                    agg_dict[col] = 'last'
                elif any(vol_kw in col.lower() for vol_kw in ['volume', 'qty', 'amount', 'size']):
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'last'  # Default to last
                
            resampled_dfs.append(df.resample(freq).agg(agg_dict))
        
        dfs = resampled_dfs
    
    # Determine the reference index based on the join method
    if method == 'outer':
        # Union of all indices
        ref_index = dfs[0].index
        for df in dfs[1:]:
            ref_index = ref_index.union(df.index)
    elif method == 'inner':
        # Intersection of all indices
        ref_index = dfs[0].index
        for df in dfs[1:]:
            ref_index = ref_index.intersection(df.index)
    elif method == 'left':
        # Use the first DataFrame's index
        ref_index = dfs[0].index
    elif method == 'right':
        # Use the last DataFrame's index
        ref_index = dfs[-1].index
    else:
        raise ValueError(f"Invalid join method: {method}")
    
    # Reindex all DataFrames to the reference index
    aligned_dfs = []
    for df in dfs:
        aligned_df = df.reindex(ref_index)
        
        # Apply fill method if specified
        if fill_method == 'ffill':
            aligned_df = aligned_df.ffill()
        elif fill_method == 'bfill':
            aligned_df = aligned_df.bfill()
        elif fill_method == 'nearest':
            aligned_df = aligned_df.interpolate(method='nearest')
        
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs


#------------------------------------------------------------------------
# Return Calculation
#------------------------------------------------------------------------

def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'log',
    periods: int = 1,
    col_name: str = None,
    dropna: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from price data.
    
    Args:
        prices: Price data (Series or DataFrame)
        method: Return calculation method ('log', 'pct', 'diff', 'ratio')
        periods: Number of periods to shift for return calculation
        col_name: Column name if prices is a DataFrame
        dropna: Whether to drop NaN values
        
    Returns:
        Series or DataFrame with calculated returns
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Input validation
    if method not in ['log', 'pct', 'diff', 'ratio']:
        raise ValueError(f"Invalid return method: {method}. Use 'log', 'pct', 'diff', or 'ratio'")
    
    # Extract Series from DataFrame if column specified
    if isinstance(prices, pd.DataFrame) and col_name is not None:
        if col_name not in prices.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        price_data = prices[col_name]
    else:
        price_data = prices
    
    # Calculate returns based on specified method
    if method == 'log':
        returns = np.log(price_data / price_data.shift(periods))
    elif method == 'pct':
        returns = price_data.pct_change(periods=periods)
    elif method == 'diff':
        returns = price_data.diff(periods=periods)
    elif method == 'ratio':
        returns = price_data / price_data.shift(periods)
    
    # Drop NaN values if requested
    if dropna:
        if isinstance(returns, pd.DataFrame):
            returns = returns.dropna()
        else:
            returns = returns.dropna()
    
    return returns


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


def calculate_cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame],
    starting_value: float = 1.0,
    log_returns: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate cumulative returns from a return series.
    
    Args:
        returns: Return data
        starting_value: Initial investment value
        log_returns: Whether input returns are log returns
        
    Returns:
        Series or DataFrame with cumulative returns
    """
    if log_returns:
        # For log returns: exp(sum(r))
        if isinstance(returns, pd.DataFrame):
            return starting_value * np.exp(returns.cumsum())
        else:
            return starting_value * np.exp(returns.cumsum())
    else:
        # For percentage returns: prod(1+r)
        if isinstance(returns, pd.DataFrame):
            return starting_value * (1 + returns).cumprod()
        else:
            return starting_value * (1 + returns).cumprod()


def calculate_drawdowns(
    returns: Union[pd.Series, pd.DataFrame],
    log_returns: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate drawdowns from return series.
    
    Args:
        returns: Return data
        log_returns: Whether input returns are log returns
        
    Returns:
        Series or DataFrame with drawdowns
    """
    # Calculate cumulative returns
    cumulative = calculate_cumulative_returns(returns, log_returns=log_returns)
    
    # Calculate running maximum
    running_max = cumulative.cummax()
    
    # Drawdowns as percentage from peak
    drawdowns = (cumulative / running_max) - 1
    
    return drawdowns


#------------------------------------------------------------------------
# Feature Engineering
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
        result[f'momentum_{window}'] = result[price_col] / result[price_col].shift(window) - 1
    
    if 'volatility' in indicators:
        result[f'volatility_{window}'] = result['returns'].rolling(window=window).std() * np.sqrt(252)
    
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


def calculate_zscore(
    series: pd.Series, 
    window: int = 60,
    method: str = 'rolling',
    min_periods: int = None
) -> pd.Series:
    """
    Calculate z-scores for a time series.
    
    Args:
        series: Input series
        window: Window size for calculation
        method: Method for calculation ('rolling', 'ewm', 'expanding')
        min_periods: Minimum observations required, defaults to window/2
        
    Returns:
        Series with z-scores
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Set default min_periods if not specified
    if min_periods is None:
        min_periods = window // 2
    
    # Calculate means and standard deviations
    if method == 'rolling':
        means = series.rolling(window=window, min_periods=min_periods).mean()
        stds = series.rolling(window=window, min_periods=min_periods).std()
    elif method == 'ewm':
        means = series.ewm(span=window, min_periods=min_periods).mean()
        stds = series.ewm(span=window, min_periods=min_periods).std()
    elif method == 'expanding':
        means = series.expanding(min_periods=min_periods).mean()
        stds = series.expanding(min_periods=min_periods).std()
    else:
        raise ValueError(f"Invalid z-score method: {method}. Use 'rolling', 'ewm', or 'expanding'")
    
    # Replace zero standard deviations with NaN to avoid division by zero
    stds = stds.replace(0, np.nan)
    
    # Calculate z-scores
    z_scores = (series - means) / stds
    
    return z_scores


def add_lagged_features(
    df: pd.DataFrame,
    columns: List[str] = None,
    lags: List[int] = [1, 5, 10],
    drop_na: bool = False
) -> pd.DataFrame:
    """
    Add lagged versions of selected columns as features.
    
    Args:
        df: Input DataFrame
        columns: Columns to create lags for (defaults to all numeric columns)
        lags: List of lag periods to create
        drop_na: Whether to drop rows with NaN from lag creation
        
    Returns:
        DataFrame with added lag features
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Add lag features
    for col in columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)
    
    # Drop rows with NaN if requested
    if drop_na:
        result = result.dropna()
    
    return result


def add_difference_features(
    df: pd.DataFrame,
    columns: List[str] = None,
    periods: List[int] = [1, 5, 10],
    pct_change: bool = False,
    drop_na: bool = False
) -> pd.DataFrame:
    """
    Add period-over-period differences or percentage changes.
    
    Args:
        df: Input DataFrame
        columns: Columns to create differences for (defaults to all numeric columns)
        periods: List of periods for difference calculation
        pct_change: Whether to calculate percentage change instead of absolute difference
        drop_na: Whether to drop rows with NaN from lag creation
        
    Returns:
        DataFrame with added difference features
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Add difference features
    for col in columns:
        for period in periods:
            if pct_change:
                result[f"{col}_pct_{period}"] = result[col].pct_change(periods=period)
            else:
                result[f"{col}_diff_{period}"] = result[col].diff(periods=period)
    
    # Drop rows with NaN if requested
    if drop_na:
        result = result.dropna()
    
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str] = None,
    windows: List[int] = [5, 10, 20],
    functions: Dict[str, Callable] = None,
    min_periods: Optional[int] = None,
    drop_na: bool = False
) -> pd.DataFrame:
    """
    Add rolling window aggregations as features.
    
    Args:
        df: Input DataFrame
        columns: Columns to create rolling features for (defaults to all numeric columns)
        windows: List of window sizes
        functions: Dictionary mapping function names to functions (defaults to mean, std, min, max)
        min_periods: Minimum periods required in window (defaults to window // 2)
        drop_na: Whether to drop rows with NaN from window calculations
        
    Returns:
        DataFrame with added rolling features
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Default functions if not specified
    if functions is None:
        functions = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max
        }
    
    # Add rolling features
    for col in columns:
        for window in windows:
            # Set min_periods if not specified
            mp = min_periods if min_periods is not None else window // 2
            
            # Create rolling object
            rolling = result[col].rolling(window=window, min_periods=mp)
            
            # Apply each function
            for func_name, func in functions.items():
                result[f"{col}_{func_name}_{window}"] = rolling.apply(func)
    
    # Drop rows with NaN if requested
    if drop_na:
        result = result.dropna()
    
    return result


#------------------------------------------------------------------------
# Outlier Detection and Handling
#------------------------------------------------------------------------

def detect_outliers(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0,
    window: int = None
) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        series: Input time series
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        window: Window size for rolling detection (None for global)
        
    Returns:
        Boolean Series with True for outliers
        
    Raises:
        ValueError: If an invalid method is specified
    """
    if method == 'zscore':
        if window is None:
            # Global z-score
            mean = series.mean()
            std = series.std()
            z_scores = (series - mean) / std
            return z_scores.abs() > threshold
        else:
            # Rolling z-score
            z_scores = calculate_zscore(series, window=window)
            return z_scores.abs() > threshold
    
    elif method == 'iqr':
        if window is None:
            # Global IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (series < lower_bound) | (series > upper_bound)
        else:
            # Rolling IQR
            outliers = pd.Series(False, index=series.index)
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i - window:i]
                q1 = window_data.quantile(0.25)
                q3 = window_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                current_idx = series.index[i - 1]
                current_value = series.iloc[i - 1]
                outliers.loc[current_idx] = (current_value < lower_bound) or (current_value > upper_bound)
            
            return outliers
    
    elif method == 'mad':
        if window is None:
            # Global Median Absolute Deviation
            median = series.median()
            mad = (series - median).abs().median()
            return (series - median).abs() > threshold * mad
        else:
            # Rolling MAD
            outliers = pd.Series(False, index=series.index)
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i - window:i]
                median = window_data.median()
                mad = (window_data - median).abs().median()
                
                current_idx = series.index[i - 1]
                current_value = series.iloc[i - 1]
                outliers.loc[current_idx] = (current_value - median).abs() > threshold * mad
            
            return outliers
    
    else:
        raise ValueError(f"Invalid outlier detection method: {method}. Use 'zscore', 'iqr', or 'mad'")


def handle_outliers(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'zscore',
    threshold: float = 3.0,
    treatment: str = 'winsorize',
    window: int = None
) -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (defaults to all numeric columns)
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        treatment: Treatment method ('winsorize', 'mean', 'median', 'drop', 'none')
        window: Window size for rolling detection (None for global)
        
    Returns:
        DataFrame with outliers handled
        
    Raises:
        ValueError: If an invalid treatment method is specified
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Process each column
    outlier_counts = {}
    for col in columns:
        # Skip columns with all NaN
        if result[col].isna().all():
            logger.warning(f"Skipping column '{col}': all values are NaN")
            continue
        
        # Detect outliers
        is_outlier = detect_outliers(
            result[col].dropna(),
            method=method,
            threshold=threshold,
            window=window
        )
        
        # Count outliers
        outlier_count = is_outlier.sum()
        outlier_counts[col] = outlier_count
        
        if outlier_count == 0:
            continue
        
        # Get outlier indices
        outlier_indices = is_outlier[is_outlier].index
        
        # Handle outliers according to treatment method
        if treatment == 'winsorize':
            if window is None:
                # Global winsorization
                lower_bound = result[col].quantile(0.01)
                upper_bound = result[col].quantile(0.99)
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
            else:
                for idx in outlier_indices:
                    # Find the window containing this index
                    pos = result.index.get_loc(idx)
                    window_start = max(0, pos - window + 1)
                    window_end = pos + 1
                    
                    # Get the data in this window
                    window_data = result.iloc[window_start:window_end][col]
                    
                    # Calculate winsorization bounds
                    lower_bound = window_data.quantile(0.01)
                    upper_bound = window_data.quantile(0.99)
                    
                    # Apply winsorization
                    if result.loc[idx, col] < lower_bound:
                        result.loc[idx, col] = lower_bound
                    elif result.loc[idx, col] > upper_bound:
                        result.loc[idx, col] = upper_bound
        
        elif treatment == 'mean':
            if window is None:
                # Replace with global mean
                mean_value = result[col].mean()
                result.loc[outlier_indices, col] = mean_value
            else:
                for idx in outlier_indices:
                    # Find the window containing this index
                    pos = result.index.get_loc(idx)
                    window_start = max(0, pos - window + 1)
                    window_end = pos + 1
                    
                    # Calculate mean excluding the outlier
                    window_values = result.iloc[window_start:window_end][col]
                    window_values = window_values[window_values.index != idx]
                    mean_value = window_values.mean()
                    
                    # Replace outlier with mean
                    result.loc[idx, col] = mean_value
        
        elif treatment == 'median':
            if window is None:
                # Replace with global median
                median_value = result[col].median()
                result.loc[outlier_indices, col] = median_value
            else:
                for idx in outlier_indices:
                    # Find the window containing this index
                    pos = result.index.get_loc(idx)
                    window_start = max(0, pos - window + 1)
                    window_end = pos + 1
                    
                    # Calculate median excluding the outlier
                    window_values = result.iloc[window_start:window_end][col]
                    window_values = window_values[window_values.index != idx]
                    median_value = window_values.median()
                    
                    # Replace outlier with median
                    result.loc[idx, col] = median_value
        
        elif treatment == 'drop':
            # Drop rows with outliers
            result = result.drop(outlier_indices)
        
        elif treatment == 'none':
            # Just detect, don't treat
            continue
        
        else:
            raise ValueError(
                f"Invalid outlier treatment method: {treatment}. Use "
                f"'winsorize', 'mean', 'median', 'drop', or 'none'."
            )
    
    # Log outlier counts
    if any(count > 0 for count in outlier_counts.values()):
        outlier_info = ", ".join([f"{col}: {count}" for col, count in outlier_counts.items() if count > 0])
        logger.info(f"Outliers detected and {treatment}d - {outlier_info}")
    
    return result


#------------------------------------------------------------------------
# Normalization and Scaling
#------------------------------------------------------------------------

def normalize_data(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'standard',
    feature_range: Tuple[float, float] = (0, 1),
    window: int = None,
    output_scaler: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Normalize or scale data.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize (defaults to all numeric columns)
        method: Scaling method ('standard', 'minmax', 'robust', 'log', 'box-cox')
        feature_range: Range for MinMaxScaler
        window: Window size for rolling normalization (None for global)
        output_scaler: Whether to return the fitted scaler
        
    Returns:
        Normalized DataFrame or tuple of (DataFrame, scaler)
        
    Raises:
        ValueError: If an invalid method is specified
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Default to numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify that specified columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # For global scaling
    if window is None:
        # Create the appropriate scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        elif method in ['log', 'box-cox']:
            # For these methods we don't use sklearn scalers
            scaler = None
        else:
            raise ValueError(
                f"Invalid normalization method: {method}. Use "
                f"'standard', 'minmax', 'robust', 'log', or 'box-cox'."
            )
        
        # Apply scaling
        if method in ['standard', 'minmax', 'robust']:
            # Extract columns to scale
            data_to_scale = result[columns].values
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data_to_scale)
            
            # Update DataFrame
            result[columns] = scaled_data
            
        elif method == 'log':
            # Check for non-positive values
            for col in columns:
                if (result[col] <= 0).any():
                    min_val = result[col].min()
                    if min_val <= 0:
                        # Shift to make all values positive
                        shift = abs(min_val) + 1e-6
                        logger.warning(
                            f"Column '{col}' contains non-positive values. "
                            f"Shifting by {shift} for log transformation."
                        )
                        result[col] = result[col] + shift
                
                # Apply log transformation
                result[col] = np.log(result[col])
            
        elif method == 'box-cox':
            from scipy import stats
            
            # Apply Box-Cox transformation
            for col in columns:
                # Check for non-positive values
                if (result[col] <= 0).any():
                    min_val = result[col].min()
                    if min_val <= 0:
                        # Shift to make all values positive
                        shift = abs(min_val) + 1e-6
                        logger.warning(
                            f"Column '{col}' contains non-positive values. "
                            f"Shifting by {shift} for Box-Cox transformation."
                        )
                        result[col] = result[col] + shift
                
                # Apply Box-Cox transformation
                transformed_data, lambda_param = stats.boxcox(result[col].values)
                result[col] = transformed_data
    
    # For rolling window scaling
    else:
        # Only implemented for basic methods
        if method not in ['standard', 'minmax']:
            raise ValueError(
                f"Rolling normalization only supports 'standard' and 'minmax' methods, got '{method}'"
            )
        
        for col in columns:
            # For each point, normalize based on the preceding window
            for i in range(window, len(result) + 1):
                window_data = result.iloc[i - window:i][col].values.reshape(-1, 1)
                
                if method == 'standard':
                    scaler = StandardScaler().fit(window_data)
                else:  # minmax
                    scaler = MinMaxScaler(feature_range=feature_range).fit(window_data)
                
                # Transform only the current point
                current_idx = result.index[i - 1]
                current_value = result.loc[current_idx, col]
                result.loc[current_idx, col] = scaler.transform([[current_value]])[0][0]
        
        # No scaler to return for rolling window
        scaler = None
    
    if output_scaler:
        return result, scaler
    else:
        return result


#------------------------------------------------------------------------
# Time Series Cross-Validation
#------------------------------------------------------------------------

def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.8,
    valid_pct: float = 0.1,
    gap: int = 0,
    min_train_size: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create time series train/validation/test splits.
    
    Args:
        df: Input DataFrame with datetime index
        n_splits: Number of splits to create
        train_pct: Percentage of data for training in each split
        valid_pct: Percentage of data for validation in each split
        gap: Number of samples to skip between train and validation/test
        min_train_size: Minimum size of training set
        
    Returns:
        List of (train, validation, test) DataFrame tuples
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
        ValueError: If invalid split percentages are provided
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for time series split")
    
    # Validate percentages
    if train_pct + valid_pct >= 1.0:
        raise ValueError(
            f"Sum of train_pct ({train_pct}) and valid_pct ({valid_pct}) "
            f"must be less than 1.0 to allow for a test set"
        )
    
    if train_pct <= 0 or valid_pct <= 0 or (1 - train_pct - valid_pct) <= 0:
        raise ValueError("All split percentages must be positive")
    
    # Calculate default min_train_size
    if min_train_size is None:
        min_train_size = int(len(df) * train_pct / n_splits)
    
    # Create splits
    splits = []
    
    # Size of each increment
    incr_size = (len(df) - min_train_size) // (n_splits - 1) if n_splits > 1 else 0
    
    for i in range(n_splits):
        # Calculate split indices
        train_end = min_train_size + i * incr_size
        valid_start = train_end + gap
        valid_end = valid_start + int(len(df) * valid_pct)
        test_start = valid_end + gap
        test_end = min(test_start + int(len(df) * (1 - train_pct - valid_pct)), len(df))
        
        # Skip if not enough data for all splits
        if test_end <= test_start or valid_end <= valid_start:
            logger.warning(f"Skipping split {i+1}/{n_splits}: insufficient data")
            continue
        
        # Create splits
        train = df.iloc[:train_end]
        valid = df.iloc[valid_start:valid_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, valid, test))
    
    logger.info(f"Created {len(splits)} time series splits")
    
    return splits


def expanding_window_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    gap: int = 0,
    min_train_size: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding window time series splits (train set grows with each split).
    
    Args:
        df: Input DataFrame with datetime index
        n_splits: Number of splits to create
        test_size: Proportion of data for testing
        valid_size: Proportion of data for validation
        gap: Number of samples to skip between train and validation/test
        min_train_size: Minimum size of training set
        
    Returns:
        List of (train, validation, test) DataFrame tuples
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for expanding window split")
    
    # Set default min_train_size
    if min_train_size is None:
        min_train_size = int(len(df) * 0.3)
    
    # Total size of validation and test portions
    eval_size = test_size + valid_size
    
    # Calculate size of each step
    full_size = len(df)
    eval_portion = full_size * eval_size  # Size of combined validation and test data
    step_size = (full_size - eval_portion - min_train_size) / (n_splits - 1) if n_splits > 1 else 0
    
    # Create splits
    splits = []
    
    for i in range(n_splits):
        # Calculate split indices
        train_end = min_train_size + int(i * step_size)
        valid_start = train_end + gap
        valid_end = valid_start + int(valid_size * full_size)
        test_start = valid_end + gap
        test_end = min(test_start + int(test_size * full_size), full_size)
        
        # Skip if not enough data for all splits
        if test_end <= test_start or valid_end <= valid_start:
            logger.warning(f"Skipping split {i+1}/{n_splits}: insufficient data")
            continue
        
        # Create splits
        train = df.iloc[:train_end]
        valid = df.iloc[valid_start:valid_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, valid, test))
    
    logger.info(f"Created {len(splits)} expanding window splits")
    
    return splits


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_size: int = None,
    test_size: int = None,
    valid_size: int = None,
    step_size: int = None,
    gap: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward time series splits with sliding windows.
    
    Args:
        df: Input DataFrame with datetime index
        n_splits: Number of splits to create
        train_size: Number of samples in training set (default is 70% of data)
        test_size: Number of samples in test set (default is 15% of data)
        valid_size: Number of samples in validation set (default is 15% of data)
        step_size: Number of samples to slide window forward (default is test_size)
        gap: Number of samples to skip between train/validation/test
        
    Returns:
        List of (train, validation, test) DataFrame tuples
        
    Raises:
        TypeError: If DataFrame doesn't have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a datetime index for walk forward split")
    
    # Set defaults
    if train_size is None:
        train_size = int(len(df) * 0.7)
    
    if test_size is None:
        test_size = int(len(df) * 0.15)
    
    if valid_size is None:
        valid_size = int(len(df) * 0.15)
    
    if step_size is None:
        step_size = test_size
    
    # Ensure there's enough data
    if train_size + valid_size + test_size + 2 * gap > len(df):
        raise ValueError(
            f"Insufficient data for specified split sizes: "
            f"train_size={train_size}, valid_size={valid_size}, test_size={test_size}, gap={gap}"
        )
    
    # Create splits
    splits = []
    
    for i in range(n_splits):
        # Calculate start/end indices
        train_start = i * step_size
        train_end = train_start + train_size
        
        # If we've reached the end of the data, stop
        if train_end + valid_size + test_size + 2 * gap > len(df):
            break
        
        valid_start = train_end + gap
        valid_end = valid_start + valid_size
        test_start = valid_end + gap
        test_end = test_start + test_size
        
        # Create split
        train = df.iloc[train_start:train_end]
        valid = df.iloc[valid_start:valid_end]
        test = df.iloc[test_start:test_end]
        
        splits.append((train, valid, test))
    
    logger.info(f"Created {len(splits)} walk-forward splits")
    
    return splits