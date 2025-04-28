"""
Feature Engineering Utilities

This module provides functions for generating features from time series data,
including lagged values, differences, and rolling window statistics.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data.feature_engineering")

def add_lagged_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
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
    columns: Optional[List[str]] = None,
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
    columns: Optional[List[str]] = None,
    windows: List[int] = [5, 10, 20],
    functions: Optional[Dict[str, Callable]] = None,
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