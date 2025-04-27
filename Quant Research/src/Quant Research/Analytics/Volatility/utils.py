"""
Volatility Analysis Utilities

This module provides common utility functions used across the volatility lab.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

# Import core models from project structure
from quant_research.core.models import Signal
from quant_research.core.storage import save_to_parquet

logger = logging.getLogger(__name__)

def calculate_returns(df: pd.DataFrame, 
                     column: str = 'close', 
                     return_type: str = 'log') -> pd.Series:
    """
    Calculate returns from price data.
    
    Args:
        df: DataFrame with price data
        column: Column name to use for price data
        return_type: Type of returns ('log' or 'pct')
        
    Returns:
        Series with returns
    """
    if return_type == 'log':
        return np.log(df[column] / df[column].shift(1))
    elif return_type == 'pct':
        return df[column].pct_change()
    else:
        raise ValueError(f"Unknown return type: {return_type}")

def detect_jumps(returns: pd.Series, 
                window: int = 21, 
                threshold: float = 3.0) -> pd.Series:
    """
    Detect jump components in returns.
    
    Args:
        returns: Series with return data
        window: Rolling window size for local volatility
        threshold: Z-score threshold for jump detection
        
    Returns:
        Series with boolean jump indicators
    """
    # Calculate rolling statistics
    rolling_mean = returns.rolling(window=window, min_periods=5).mean()
    rolling_std = returns.rolling(window=window, min_periods=5).std()
    
    # Calculate z-scores
    z_scores = (returns - rolling_mean) / rolling_std
    
    # Identify jumps
    jumps = np.abs(z_scores) > threshold
    
    return jumps

def filter_jumps(returns: pd.Series, 
                window: int = 21, 
                threshold: float = 3.0) -> pd.Series:
    """
    Filter out jump components from returns.
    
    Args:
        returns: Series with return data
        window: Rolling window size for local volatility
        threshold: Z-score threshold for jump detection
        
    Returns:
        Series with filtered returns (jumps replaced with local mean)
    """
    # Detect jumps
    jumps = detect_jumps(returns, window, threshold)
    
    # Get local mean
    local_mean = returns.rolling(window=window, min_periods=5).mean()
    
    # Replace jumps with local mean
    filtered_returns = returns.copy()
    filtered_returns[jumps] = local_mean[jumps]
    
    return filtered_returns

def calculate_confidence_interval(vol: pd.Series, 
                                 window: int = 21,
                                 alpha: float = 0.05,
                                 dist: str = 'normal') -> Dict[str, pd.Series]:
    """
    Calculate confidence intervals for volatility.
    
    Args:
        vol: Series with volatility estimates
        window: Window size used in estimation
        alpha: Significance level
        dist: Distribution assumption ('normal' or 't')
        
    Returns:
        Dictionary with lower and upper CI series
    """
    from scipy.stats import norm, t
    
    # Degrees of freedom
    dof = window - 1
    
    if dist == 'normal':
        # For normal distribution
        z = norm.ppf(1 - alpha/2)
        lower = vol / np.exp(z * np.sqrt(1/(2*dof)))
        upper = vol * np.exp(z * np.sqrt(1/(2*dof)))
    
    elif dist == 't':
        # For t distribution
        t_val = t.ppf(1 - alpha/2, dof)
        lower = vol / np.exp(t_val * np.sqrt(1/(2*dof)))
        upper = vol * np.exp(t_val * np.sqrt(1/(2*dof)))
    
    else:
        raise ValueError(f"Unknown distribution: {dist}. Use 'normal' or 't'.")
    
    return {'lower': lower, 'upper': upper}

def prepare_dataframe(df: pd.DataFrame, target_column: str = 'close') -> pd.DataFrame:
    """
    Prepare DataFrame for volatility analysis.
    
    Args:
        df: DataFrame with price data
        target_column: Column to validate
        
    Returns:
        Prepared DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure index is datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have a timestamp column or datetime index")
    
    # Validate required columns
    if target_column not in df.columns:
        raise ValueError(f"Required column '{target_column}' not found in DataFrame")
    
    return df

def annualize_volatility(vol: pd.Series, 
                        df: pd.DataFrame, 
                        trading_days: int = 252) -> pd.Series:
    """
    Annualize volatility.
    
    Args:
        vol: Series with volatility estimates
        df: Original DataFrame with index
        trading_days: Number of trading days in a year
        
    Returns:
        Series with annualized volatility
    """
    # Get time interval between consecutive observations
    if isinstance(df.index, pd.DatetimeIndex):
        # For timestamp index, get median interval in days
        intervals = df.index.to_series().diff().dropna()
        if len(intervals) > 0:
            median_interval = intervals.median().total_seconds() / (24 * 3600)
            scaling_factor = np.sqrt(trading_days / median_interval)
        else:
            scaling_factor = np.sqrt(trading_days)
    else:
        # Default scaling for daily data
        scaling_factor = np.sqrt(trading_days)
    
    return vol * scaling_factor

def format_signals(volatility: pd.DataFrame, 
                  df: pd.DataFrame, 
                  params: Dict[str, Any],
                  signal_type: str,
                  model_name: str) -> pd.DataFrame:
    """
    Format volatility estimates into standard signal format.
    
    Args:
        volatility: DataFrame with volatility estimates
        df: Original data DataFrame
        params: Model parameters
        signal_type: Type of signal
        model_name: Name of the model
        
    Returns:
        DataFrame with formatted signals
    """
    # Create signal DataFrame
    result = pd.DataFrame({
        'timestamp': volatility.index,
        'symbol': params.get('symbol', df.get('symbol', 'UNKNOWN').iloc[0]),
        'signal_type': signal_type,
        'model': model_name,
        'value': volatility['volatility'],
        'lower_bound': volatility['volatility_lower'],
        'upper_bound': volatility['volatility_upper'],
        'window': params.get('window', 21)
    })
    
    return result

def save_signals(signals: pd.DataFrame, 
                params: Dict[str, Any], 
                as_objects: bool = False) -> Union[pd.DataFrame, List[Signal]]:
    """
    Save signals and optionally convert to Signal objects.
    
    Args:
        signals: DataFrame with signals
        params: Model parameters
        as_objects: Whether to return Signal objects
        
    Returns:
        DataFrame or list of Signal objects
    """
    # Convert to Signal objects if needed
    if as_objects:
        result = []
        for _, row in signals.iterrows():
            result.append(
                Signal(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    signal_type=row['signal_type'],
                    model=row['model'],
                    value=row['value'],
                    metadata={
                        'lower_bound': row['lower_bound'],
                        'upper_bound': row['upper_bound'],
                        'window': row['window'],
                        'params': params
                    }
                )
            )
    else:
        result = signals
    
    # Optionally save to parquet
    output_file = params.get('output_file')
    if output_file:
        save_to_parquet(signals, output_file)
        logger.info(f"Saved signals to {output_file}")
    
    return result