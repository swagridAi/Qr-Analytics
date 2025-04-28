"""
Core DataFrame Operations

This module provides fundamental data handling and validation operations for time series data.
It focuses on ensuring data is in the correct format and structure for analysis.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data.core_operations")

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
    required_cols: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None
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


def detect_frequency(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the frequency of time series data.
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        Inferred frequency as pandas frequency string or None if can't be detected
        
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


def validate_missing_data(
    df: pd.DataFrame,
    threshold: float = 0.1,
    columns: Optional[List[str]] = None,
    drop_columns: bool = False,
    fill_method: Optional[str] = None,
    raise_exceptions: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Validate and handle missing data in a DataFrame.
    
    Args:
        df: DataFrame to validate
        threshold: Maximum allowed proportion of missing data
        columns: Columns to check (None for all columns)
        drop_columns: Whether to drop columns with too many missing values
        fill_method: Method to fill missing values ('ffill', 'bfill', 'mean', etc.)
        raise_exceptions: Whether to raise exceptions on validation errors
        
    Returns:
        Tuple of (validated DataFrame, dictionary of missing data proportions)
        
    Raises:
        Exception: If validation fails and raise_exceptions is True
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Determine columns to check
    check_columns = columns or df_copy.columns
    
    # Calculate missing data proportions
    missing_props = {}
    for col in check_columns:
        if col in df_copy.columns:
            missing_props[col] = df_copy[col].isna().mean()
    
    # Find columns with too many missing values
    high_missing_cols = {col: prop for col, prop in missing_props.items() if prop > threshold}
    
    # Handle columns with too many missing values
    if high_missing_cols:
        if drop_columns:
            df_copy = df_copy.drop(columns=list(high_missing_cols.keys()))
            
            # Update missing_props after dropping columns
            missing_props = {col: prop for col, prop in missing_props.items() if col not in high_missing_cols}
        elif fill_method:
            # Fill missing values
            if fill_method == 'ffill':
                df_copy = df_copy.ffill()
            elif fill_method == 'bfill':
                df_copy = df_copy.bfill()
            elif fill_method == 'mean':
                for col in high_missing_cols:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif fill_method == 'median':
                for col in high_missing_cols:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif fill_method == 'mode':
                for col in high_missing_cols:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif fill_method == 'zero':
                for col in high_missing_cols:
                    df_copy[col] = df_copy[col].fillna(0)
            else:
                logger.warning(f"Unknown fill method: {fill_method}")
        
        # Raise exception if requested
        if raise_exceptions:
            errors = [f"Column '{col}' has {prop*100:.1f}% missing values (threshold: {threshold*100:.1f}%)" 
                     for col, prop in high_missing_cols.items()]
            
            raise Exception(
                f"Too many missing values: {'; '.join(errors)}"
            )
    
    return df_copy, missing_props