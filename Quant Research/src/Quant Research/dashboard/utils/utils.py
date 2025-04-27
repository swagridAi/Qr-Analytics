"""
Utility functions for the data loader module of Quant Research dashboard.

This module provides utility functions for loading, transforming, validating,
and managing data for the dashboard visualizations.
"""

import os
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# Set up logging
logger = logging.getLogger(__name__)


# ========== Data Validation Functions ==========

def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: List[str],
    numeric_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: Optional list of columns that should be numeric
        datetime_columns: Optional list of columns that should be datetime
        
    Returns:
        Tuple of (is_valid, missing_columns, type_error_columns)
    """
    if df is None or df.empty:
        return False, required_columns, []
    
    # Check for missing columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, missing, []
    
    # Check numeric columns
    type_errors = []
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns and not is_numeric_dtype(df[col]):
                type_errors.append(f"{col} (expected numeric)")
    
    # Check datetime columns
    if datetime_columns:
        for col in datetime_columns:
            if col in df.columns and not is_datetime64_any_dtype(df[col]):
                type_errors.append(f"{col} (expected datetime)")
    
    return len(missing) == 0 and len(type_errors) == 0, missing, type_errors


def clean_dataframe(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    drop_na_columns: Optional[List[str]] = None,
    fill_na_values: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Clean and preprocess a DataFrame for analysis.
    
    Args:
        df: DataFrame to clean
        numeric_columns: Columns to convert to numeric
        datetime_columns: Columns to convert to datetime
        categorical_columns: Columns to convert to categorical
        drop_na_columns: Columns where NA values should be dropped
        fill_na_values: Dictionary of {column: fill_value} for NA filling
    
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Convert numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Convert datetime columns
    if datetime_columns:
        for col in datetime_columns:
            if col in cleaned_df.columns and not is_datetime64_any_dtype(cleaned_df[col]):
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
    
    # Convert categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype('category')
    
    # Drop NA values in specified columns
    if drop_na_columns:
        valid_drop_columns = [col for col in drop_na_columns if col in cleaned_df.columns]
        if valid_drop_columns:
            cleaned_df.dropna(subset=valid_drop_columns, inplace=True)
    
    # Fill NA values in specified columns
    if fill_na_values:
        for col, value in fill_na_values.items():
            if col in cleaned_df.columns:
                cleaned_df[col].fillna(value, inplace=True)
    
    return cleaned_df


# ========== Data Loading Functions ==========

def get_data_path(subpath: Optional[str] = None) -> Path:
    """
    Get the path to the data directory or a subdirectory.
    
    Args:
        subpath: Optional subdirectory within the data directory
        
    Returns:
        Path object to the requested directory
    """
    # Determine project root from environment variable or infer from file structure
    if "QUANT_RESEARCH_ROOT" in os.environ:
        root_dir = Path(os.environ["QUANT_RESEARCH_ROOT"])
    else:
        # Assume this module is in quant_research/dashboard/utils
        current_path = Path(__file__).resolve()
        # Navigate up three levels to project root
        root_dir = current_path.parent.parent.parent.parent
    
    # Create path to data directory
    data_dir = root_dir / "data"
    
    # Return with optional subpath
    if subpath:
        return data_dir / subpath
    
    return data_dir


@lru_cache(maxsize=32)
def get_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file from the configs directory.
    
    Args:
        config_name: Name of the configuration file (without extension)
        
    Returns:
        Configuration dictionary
    """
    # Determine project root
    if "QUANT_RESEARCH_ROOT" in os.environ:
        root_dir = Path(os.environ["QUANT_RESEARCH_ROOT"])
    else:
        # Assume this module is in quant_research/dashboard/utils
        current_path = Path(__file__).resolve()
        # Navigate up three levels to project root
        root_dir = current_path.parent.parent.parent.parent
    
    # Create path to config file
    config_path = root_dir / "configs" / f"{config_name}.yaml"
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    # Load YAML config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration {config_name}: {str(e)}")
        return {}


def load_parquet_data(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List[List[Tuple]]] = None
) -> pd.DataFrame:
    """
    Load data from a Parquet file with optimized settings.
    
    Args:
        file_path: Path to the Parquet file
        columns: Optional list of columns to read
        filters: Optional PyArrow filters to apply during read
        
    Returns:
        DataFrame with the requested data
    """
    try:
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Prepend data directory if not an absolute path
        if not file_path.is_absolute():
            file_path = get_data_path() / file_path
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Parquet file not found: {file_path}")
            return pd.DataFrame()
        
        # Read parquet file
        table = pq.read_table(file_path, columns=columns, filters=filters)
        df = table.to_pandas()
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading Parquet file {file_path}: {str(e)}")
        return pd.DataFrame()


def load_duckdb_data(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    connection_string: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from DuckDB with a SQL query.
    
    Args:
        query: SQL query to execute
        params: Optional parameters for the query
        connection_string: Optional custom connection string
        
    Returns:
        DataFrame with the query results
    """
    try:
        import duckdb
        
        # Use default connection if not specified
        if not connection_string:
            db_path = get_data_path() / "database.duckdb"
            connection_string = str(db_path)
        
        # Connect to database
        con = duckdb.connect(connection_string)
        
        # Execute query with parameters if provided
        if params:
            df = con.execute(query, params).df()
        else:
            df = con.execute(query).df()
        
        con.close()
        return df
    
    except ImportError:
        logger.error("DuckDB package not installed. Please install with: pip install duckdb")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error executing DuckDB query: {str(e)}")
        return pd.DataFrame()


# ========== Data Transformation Functions ==========

def resample_time_series(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    value_column: str = 'close',
    freq: str = '1D',
    agg_func: str = 'last',
    fillna: Optional[Any] = None
) -> pd.DataFrame:
    """
    Resample a time series DataFrame to a different frequency.
    
    Args:
        df: DataFrame with time series data
        time_column: Name of the timestamp column
        value_column: Name of the value column to resample
        freq: Frequency string (e.g., '1D', '1H', '5min')
        agg_func: Aggregation function ('last', 'first', 'mean', 'sum', etc.)
        fillna: Value to fill NA values with after resampling
        
    Returns:
        Resampled DataFrame
    """
    if df is None or df.empty or time_column not in df.columns:
        return pd.DataFrame()
    
    # Ensure timestamp column is datetime
    if not is_datetime64_any_dtype(df[time_column]):
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
    
    # Set timestamp as index
    df_indexed = df.set_index(time_column)
    
    # Apply resampling
    if agg_func == 'last':
        resampled = df_indexed.resample(freq).last()
    elif agg_func == 'first':
        resampled = df_indexed.resample(freq).first()
    elif agg_func == 'mean':
        resampled = df_indexed.resample(freq).mean()
    elif agg_func == 'sum':
        resampled = df_indexed.resample(freq).sum()
    else:
        resampled = df_indexed.resample(freq).agg(agg_func)
    
    # Fill NA values if needed
    if fillna is not None:
        resampled.fillna(fillna, inplace=True)
    
    # Reset index to get timestamp as column again
    return resampled.reset_index()


def filter_time_range(
    df: pd.DataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    time_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Filter a DataFrame to a specific time range.
    
    Args:
        df: DataFrame to filter
        start_date: Start date for filter (inclusive)
        end_date: End date for filter (inclusive)
        time_column: Name of the timestamp column
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or time_column not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Ensure timestamp column is datetime
    if not is_datetime64_any_dtype(filtered_df[time_column]):
        filtered_df[time_column] = pd.to_datetime(filtered_df[time_column])
    
    # Apply start date filter
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df[time_column] >= start_date]
    
    # Apply end date filter
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df[time_column] <= end_date]
    
    return filtered_df


def calculate_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    timestamp_column: str = 'timestamp',
    method: str = 'log',
    periods: int = 1
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        df: DataFrame with price data
        price_column: Name of the price column
        timestamp_column: Name of the timestamp column
        method: Method for return calculation ('log', 'simple')
        periods: Number of periods for return calculation
        
    Returns:
        DataFrame with added return column
    """
    if df is None or df.empty or price_column not in df.columns:
        return df
    
    # Create a copy with needed columns
    result_df = df.copy()
    
    # Sort by timestamp if available
    if timestamp_column in result_df.columns:
        result_df = result_df.sort_values(timestamp_column)
    
    # Calculate returns
    if method.lower() == 'log':
        result_df['returns'] = np.log(result_df[price_column] / result_df[price_column].shift(periods))
    else:  # simple returns
        result_df['returns'] = result_df[price_column].pct_change(periods=periods)
    
    return result_df


def pivot_time_series(
    df: pd.DataFrame,
    index_column: str = 'timestamp',
    column_column: str = 'symbol',
    value_column: str = 'close'
) -> pd.DataFrame:
    """
    Pivot a time series DataFrame from long to wide format.
    
    Args:
        df: DataFrame in long format
        index_column: Column to use as index (usually timestamp)
        column_column: Column to use as columns (e.g., 'symbol')
        value_column: Column containing the values
        
    Returns:
        Pivoted DataFrame in wide format
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Check if required columns exist
    if not all(col in df.columns for col in [index_column, column_column, value_column]):
        missing = [col for col in [index_column, column_column, value_column] if col not in df.columns]
        logger.error(f"Missing columns for pivot: {missing}")
        return df
    
    try:
        # Pivot the DataFrame
        pivoted = df.pivot(index=index_column, columns=column_column, values=value_column)
        
        # Handle potential duplicate indices
        if pivoted.index.has_duplicates:
            logger.warning(f"Duplicate {index_column} values detected. Using last value for each.")
            # Group by index and take the last value
            pivoted = df.groupby([index_column, column_column])[value_column].last().unstack()
        
        return pivoted
    except Exception as e:
        logger.error(f"Error pivoting DataFrame: {str(e)}")
        return df


def normalize_series(
    series: pd.Series,
    method: str = 'min-max',
    target_range: Tuple[float, float] = (0, 1)
) -> pd.Series:
    """
    Normalize a Series to a specified range.
    
    Args:
        series: Series to normalize
        method: Normalization method ('min-max', 'z-score', 'robust')
        target_range: Target range for min-max normalization
        
    Returns:
        Normalized Series
    """
    if series is None or series.empty:
        return series
    
    if method == 'min-max':
        # Min-max normalization
        min_val, max_val = series.min(), series.max()
        if min_val == max_val:
            return pd.Series(target_range[0], index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        
        # Scale to target range if not (0, 1)
        if target_range != (0, 1):
            range_width = target_range[1] - target_range[0]
            normalized = normalized * range_width + target_range[0]
            
    elif method == 'z-score':
        # Z-score normalization
        mean, std = series.mean(), series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        
        normalized = (series - mean) / std
        
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = series.median()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return pd.Series(0, index=series.index)
        
        normalized = (series - median) / iqr
        
    else:
        # Unknown method, return original
        logger.warning(f"Unknown normalization method: {method}")
        return series
    
    return normalized


# ========== Time Series Analysis Functions ==========

def detect_outliers(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        series: Series to check for outliers
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean Series with True for outliers
    """
    if series is None or series.empty:
        return pd.Series(dtype=bool)
    
    if method == 'zscore':
        # Z-score method
        z_scores = (series - series.mean()) / series.std()
        return z_scores.abs() > threshold
        
    elif method == 'iqr':
        # IQR method
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)
        
    elif method == 'mad':
        # Median Absolute Deviation method
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:  # Avoid division by zero
            return pd.Series(False, index=series.index)
        return (series - median).abs() > threshold * mad
        
    else:
        logger.warning(f"Unknown outlier detection method: {method}")
        return pd.Series(False, index=series.index)


def detect_jumps(
    series: pd.Series,
    threshold: float = 3.0,
    window: int = 20
) -> pd.Series:
    """
    Detect price jumps in a time series.
    
    Args:
        series: Series of prices
        threshold: Threshold for jump detection (in std dev)
        window: Window size for rolling statistics
        
    Returns:
        Boolean Series with True for jumps
    """
    if series is None or len(series) < window + 1:
        return pd.Series(dtype=bool)
    
    # Calculate returns
    returns = series.pct_change()
    
    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Detect jumps where return exceeds threshold * rolling std
    jumps = returns.abs() > threshold * rolling_std
    
    return jumps


def calculate_rolling_stats(
    series: pd.Series,
    window: int = 20,
    stats: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Calculate rolling statistics for a time series.
    
    Args:
        series: Series to analyze
        window: Window size for rolling calculations
        stats: List of statistics to calculate
        
    Returns:
        DataFrame with rolling statistics
    """
    if series is None or series.empty or len(series) < window:
        return pd.DataFrame()
    
    result = pd.DataFrame(index=series.index)
    
    # Calculate requested statistics
    for stat in stats:
        if stat == 'mean':
            result['rolling_mean'] = series.rolling(window=window).mean()
        elif stat == 'std':
            result['rolling_std'] = series.rolling(window=window).std()
        elif stat == 'min':
            result['rolling_min'] = series.rolling(window=window).min()
        elif stat == 'max':
            result['rolling_max'] = series.rolling(window=window).max()
        elif stat == 'median':
            result['rolling_median'] = series.rolling(window=window).median()
        elif stat == 'var':
            result['rolling_var'] = series.rolling(window=window).var()
        elif stat == 'skew':
            result['rolling_skew'] = series.rolling(window=window).skew()
        elif stat == 'kurt':
            result['rolling_kurt'] = series.rolling(window=window).kurt()
        elif stat == 'sum':
            result['rolling_sum'] = series.rolling(window=window).sum()
        else:
            logger.warning(f"Unknown rolling statistic: {stat}")
    
    return result


# ========== Special Financial Functions ==========

def calculate_drawdowns(
    equity_series: pd.Series
) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdown series and metrics from an equity curve.
    
    Args:
        equity_series: Series of equity values
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_drawdown_duration)
    """
    if equity_series is None or equity_series.empty:
        return pd.Series(), 0.0, 0
    
    # Calculate running maximum
    running_max = equity_series.cummax()
    
    # Calculate drawdown series (as percentage)
    drawdown_series = (equity_series / running_max - 1) * 100
    
    # Find maximum drawdown
    max_drawdown = drawdown_series.min()
    
    # Calculate maximum drawdown duration
    if max_drawdown < 0:
        # Find the start of the max drawdown (index of the peak)
        dd_start = np.argmax(running_max.values[:np.argmin(drawdown_series.values)])
        
        # Find the end of the max drawdown (index when equity returns to previous peak)
        # or the last index if recovery hasn't happened
        dd_end = len(equity_series)
        for i in range(np.argmin(drawdown_series.values), len(equity_series)):
            if equity_series.iloc[i] >= running_max.iloc[dd_start]:
                dd_end = i
                break
        
        max_drawdown_duration = dd_end - dd_start
    else:
        max_drawdown_duration = 0
    
    return drawdown_series, max_drawdown, max_drawdown_duration


def calculate_var(
    returns_series: pd.Series,
    confidence_level: float = 0.95,
    period: int = 1
) -> float:
    """
    Calculate Value at Risk (VaR) for a returns series.
    
    Args:
        returns_series: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        period: Time period for the VaR calculation
        
    Returns:
        VaR value (as a positive percentage)
    """
    if returns_series is None or returns_series.empty:
        return 0.0
    
    # Calculate the VaR percentile
    var_percentile = 100 * (1 - confidence_level)
    
    # Historical VaR calculation
    var = np.percentile(returns_series, var_percentile)
    
    # Scale VaR to the requested period (assuming returns are i.i.d.)
    if period > 1:
        var = var * np.sqrt(period)
    
    # Return as a positive value
    return abs(var)


def calculate_sharpe_ratio(
    returns_series: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a returns series.
    
    Args:
        returns_series: Series of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if returns_series is None or returns_series.empty:
        return 0.0
    
    # Calculate excess returns over risk-free rate
    excess_returns = returns_series - (risk_free_rate / periods_per_year)
    
    # Calculate annualized mean and standard deviation
    mean_excess = excess_returns.mean() * periods_per_year
    std_dev = returns_series.std() * np.sqrt(periods_per_year)
    
    # Avoid division by zero
    if std_dev == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    sharpe = mean_excess / std_dev
    
    return sharpe


def calculate_sortino_ratio(
    returns_series: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino ratio for a returns series.
    
    Args:
        returns_series: Series of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    if returns_series is None or returns_series.empty:
        return 0.0
    
    # Calculate excess returns over risk-free rate
    excess_returns = returns_series - (risk_free_rate / periods_per_year)
    
    # Calculate annualized mean
    mean_excess = excess_returns.mean() * periods_per_year
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = returns_series[returns_series < 0]
    
    if negative_returns.empty:
        return np.inf  # No negative returns, infinite Sortino ratio
    
    downside_dev = negative_returns.std() * np.sqrt(periods_per_year)
    
    # Avoid division by zero
    if downside_dev == 0:
        return 0.0
    
    # Calculate Sortino ratio
    sortino = mean_excess / downside_dev
    
    return sortino


def adjust_for_corporate_actions(
    prices: pd.Series,
    dividends: Optional[pd.Series] = None,
    splits: Optional[pd.Series] = None
) -> pd.Series:
    """
    Adjust price series for dividends and stock splits.
    
    Args:
        prices: Series of prices indexed by date
        dividends: Optional series of dividends indexed by ex-dividend date
        splits: Optional series of split ratios indexed by split date
        
    Returns:
        Adjusted price series
    """
    if prices is None or prices.empty:
        return prices
    
    # Create a copy of the original prices
    adjusted_prices = prices.copy()
    
    # Apply split adjustments if available
    if splits is not None and not splits.empty:
        # Combine indices to ensure we have all dates
        all_dates = sorted(set(prices.index) | set(splits.index))
        
        # Create a new Series with all dates
        full_prices = pd.Series(index=all_dates)
        full_prices.loc[prices.index] = prices
        
        # Forward fill to ensure we have prices on split dates
        full_prices = full_prices.ffill()
        
        # Calculate cumulative split factor (from newest to oldest)
        cum_split_factor = 1.0
        
        # Adjust prices for splits (working backwards in time)
        for date in sorted(splits.index, reverse=True):
            split_ratio = splits.loc[date]
            
            # Update cumulative factor
            cum_split_factor *= split_ratio
            
            # Adjust prices before this date
            mask = full_prices.index < date
            full_prices.loc[mask] = full_prices.loc[mask] / split_ratio
        
        # Keep only the original dates
        adjusted_prices = full_prices.loc[prices.index]
    
    # Apply dividend adjustments if available
    if dividends is not None and not dividends.empty:
        # Combine indices to ensure we have all dates
        all_dates = sorted(set(adjusted_prices.index) | set(dividends.index))
        
        # Create a new Series with all dates
        full_prices = pd.Series(index=all_dates)
        full_prices.loc[adjusted_prices.index] = adjusted_prices
        
        # Forward fill to ensure we have prices on dividend dates
        full_prices = full_prices.ffill()
        
        # Calculate adjustment factor for each dividend
        for date in sorted(dividends.index, reverse=True):
            if date in full_prices.index:
                dividend = dividends.loc[date]
                price_on_exdate = full_prices.loc[date]
                
                # Calculate adjustment factor
                if price_on_exdate > 0:
                    factor = (price_on_exdate - dividend) / price_on_exdate
                    
                    # Adjust all prices before this date
                    mask = full_prices.index < date
                    full_prices.loc[mask] = full_prices.loc[mask] * factor
        
        # Keep only the original dates
        adjusted_prices = full_prices.loc[prices.index]
    
    return adjusted_prices