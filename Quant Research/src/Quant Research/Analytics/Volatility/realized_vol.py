"""
Realized Volatility Estimators

This module implements various realized volatility estimators, including range-based
and high-frequency return-based methods. These estimators are designed to capture
realized market volatility based on historical price data.

The main function to use is generate_signal which accepts price data and returns
volatility measurements in the standard signal format.

Example:
    >>> from quant_research.analytics.volatility import realized_vol
    >>> signals = realized_vol.generate_signal(price_df, estimator='yang_zhang', window=21)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

# Import utilities and models
from quant_research.analytics.volatility.utils import (
    calculate_returns, prepare_dataframe, calculate_confidence_interval,
    filter_jumps, annualize_volatility, format_signals, save_signals
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants and Default Parameters
# -----------------------------------------------------------------------------

# Default parameters
DEFAULT_PARAMS = {
    'estimator': 'standard',    # Volatility estimator method
    'window': 21,              # Rolling window size (days)
    'min_periods': 5,          # Minimum periods for calculation
    'return_type': 'log',      # Either 'log' or 'pct' for return calculation
    'annualize': True,         # Whether to annualize volatility
    'trading_days': 252,       # Number of trading days in a year
    'alpha': 0.05,             # Significance level for confidence intervals
    'use_high_low': True,      # Use high/low prices (if available)
    'use_jump_filter': False,  # Apply jump filtering
    'jump_threshold': 3.0,     # Z-score threshold for jumps
    'bias_correction': True,   # Apply small-sample bias correction
    'target_column': 'close',  # Column to use for pricing
}

# Available estimator methods
ESTIMATOR_METHODS = {
    'standard': 'Standard close-to-close volatility',
    'parkinson': 'Parkinson high-low range estimator',
    'garman_klass': 'Garman-Klass estimator with open/high/low/close',
    'rogers_satchell': 'Rogers-Satchell estimator for non-zero mean',
    'yang_zhang': 'Yang-Zhang estimator combining overnight and intraday vol',
    'hodges_tompkins': 'Hodges-Tompkins bias-corrected estimator',
    'realized_range': 'Realized range using intraday high-low',
    'realized_variance': 'Realized variance from high-frequency returns',
    'realized_kernel': 'Realized kernel with noise correction',
}

# -----------------------------------------------------------------------------
# Standard Volatility Estimators
# -----------------------------------------------------------------------------

def standard_deviation(returns: pd.Series, 
                      window: int = 21, 
                      min_periods: int = 5,
                      center: bool = False,
                      bias_correction: bool = True) -> pd.Series:
    """
    Calculate rolling standard deviation of returns.
    
    Args:
        returns: Series with return data
        window: Rolling window size
        min_periods: Minimum number of observations
        center: Whether to center the window
        bias_correction: Apply small-sample bias correction
        
    Returns:
        Series with rolling standard deviation
    """
    # Standard deviation with specified window
    std = returns.rolling(
        window=window, 
        min_periods=min_periods, 
        center=center
    ).std()
    
    # Apply bias correction for small samples if required
    if bias_correction:
        # Correction factor for small samples
        n = window
        correction = np.sqrt((n-1) / 2) * np.exp(
            np.lgamma((n-1)/2) - np.lgamma(n/2)
        )
        std = std * correction
    
    return std


def hodges_tompkins_vol(returns: pd.Series, 
                       window: int = 21, 
                       min_periods: int = 5) -> pd.Series:
    """
    Calculate Hodges-Tompkins volatility estimator with bias correction.
    
    Args:
        returns: Series with return data
        window: Rolling window size
        min_periods: Minimum number of observations
        
    Returns:
        Series with bias-corrected volatility estimator
    """
    # Simple rolling standard deviation
    simple_std = returns.rolling(window=window, min_periods=min_periods).std()
    
    # Apply Hodges-Tompkins correction
    n = window
    t = np.arange(1, n+1)
    w = np.sum(t) / np.sum(t**2)  # Weights based on linear time scale
    
    # Calculate weighted mean
    weighted_mean = returns.rolling(window=window, min_periods=min_periods).apply(
        lambda x: np.sum(w * x * np.arange(1, len(x)+1)) / np.sum(w * np.arange(1, len(x)+1)),
        raw=True
    )
    
    # Calculate weighted variance
    def weighted_var(x):
        if len(x) < min_periods:
            return np.nan
        t = np.arange(1, len(x)+1)
        w = t / np.sum(t)
        mean = np.sum(w * x)
        return np.sum(w * (x - mean)**2) * (n / (n-1))
    
    weighted_var_series = returns.rolling(window=window, min_periods=min_periods).apply(
        weighted_var,
        raw=True
    )
    
    # Take square root for volatility
    result = np.sqrt(weighted_var_series)
    
    return result


# -----------------------------------------------------------------------------
# Range-Based Volatility Estimators
# -----------------------------------------------------------------------------

def parkinson_vol(df: pd.DataFrame, 
                 window: int = 21, 
                 min_periods: int = 5) -> pd.Series:
    """
    Calculate Parkinson's volatility estimator based on high-low range.
    
    Args:
        df: DataFrame with high and low prices
        window: Rolling window size
        min_periods: Minimum number of observations
        
    Returns:
        Series with Parkinson volatility estimator
    """
    # Check if high and low columns exist
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns for Parkinson estimator")
    
    # Calculate squared log range
    log_hl = np.log(df['high'] / df['low'])
    squared_log_range = log_hl**2
    
    # Parkinson estimator (using high-low range)
    # Scaling factor 1/(4*ln(2)) ≈ 0.361
    scaling_factor = 1.0 / (4.0 * np.log(2.0))
    estimator = squared_log_range * scaling_factor
    
    # Rolling sum for the window
    result = np.sqrt(
        estimator.rolling(window=window, min_periods=min_periods).sum() / window
    )
    
    return result


def garman_klass_vol(df: pd.DataFrame, 
                    window: int = 21, 
                    min_periods: int = 5) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    
    Uses open, high, low, close prices.
    
    Args:
        df: DataFrame with OHLC prices
        window: Rolling window size
        min_periods: Minimum number of observations
        
    Returns:
        Series with Garman-Klass volatility estimator
    """
    # Check if required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column for Garman-Klass estimator")
    
    # Calculate components
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    
    # Garman-Klass estimator
    # 0.5 * hl_part - (2*ln(2)-1) * co_part
    hl_part = log_hl**2
    co_part = log_co**2
    
    daily_vol = 0.5 * hl_part - (2 * np.log(2) - 1) * co_part
    
    # Rolling sum for the window
    result = np.sqrt(
        daily_vol.rolling(window=window, min_periods=min_periods).sum() / window
    )
    
    return result


def rogers_satchell_vol(df: pd.DataFrame, 
                       window: int = 21, 
                       min_periods: int = 5) -> pd.Series:
    """
    Calculate Rogers-Satchell volatility estimator.
    
    This estimator is drift-independent (works for non-zero mean).
    
    Args:
        df: DataFrame with OHLC prices
        window: Rolling window size
        min_periods: Minimum number of observations
        
    Returns:
        Series with Rogers-Satchell volatility estimator
    """
    # Check if required columns exist
    required_cols = ['high', 'low', 'close', 'open']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column for Rogers-Satchell estimator")
    
    # Calculate components
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])
    
    # Rogers-Satchell estimator
    daily_vol = log_hc * log_ho + log_lc * log_lo
    
    # Rolling sum for the window
    result = np.sqrt(
        daily_vol.rolling(window=window, min_periods=min_periods).sum() / window
    )
    
    return result


def yang_zhang_vol(df: pd.DataFrame, 
                  window: int = 21, 
                  min_periods: int = 5,
                  k: float = 0.34) -> pd.Series:
    """
    Calculate Yang-Zhang volatility estimator.
    
    This estimator combines overnight and intraday volatility.
    
    Args:
        df: DataFrame with OHLC prices
        window: Rolling window size
        min_periods: Minimum number of observations
        k: Weighting parameter (typically 0.34)
        
    Returns:
        Series with Yang-Zhang volatility estimator
    """
    # Check if required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column for Yang-Zhang estimator")
    
    # Close-to-close volatility component
    close_returns = np.log(df['close'] / df['close'].shift(1))
    close_vol = close_returns.rolling(window=window, min_periods=min_periods).var()
    
    # Open-to-open volatility component
    open_returns = np.log(df['open'] / df['open'].shift(1))
    open_vol = open_returns.rolling(window=window, min_periods=min_periods).var()
    
    # Overnight volatility: close to next day's open
    overnight_returns = np.log(df['open'] / df['close'].shift(1))
    overnight_vol = overnight_returns.rolling(window=window, min_periods=min_periods).var()
    
    # Open to close (intraday) volatility
    intraday_returns = np.log(df['close'] / df['open'])
    intraday_vol = intraday_returns.rolling(window=window, min_periods=min_periods).var()
    
    # Rogers-Satchell volatility component
    rs_vol = rogers_satchell_vol(df, window, min_periods)
    
    # Yang-Zhang estimator
    # YZ = overnight_vol + k*intraday_vol + (1-k)*rs_vol
    yang_zhang = overnight_vol + k * intraday_vol + (1 - k) * rs_vol**2
    
    return np.sqrt(yang_zhang)


# -----------------------------------------------------------------------------
# High-Frequency Volatility Estimators
# -----------------------------------------------------------------------------

def realized_range_vol(df: pd.DataFrame, 
                      window: int = 21, 
                      min_periods: int = 5,
                      sampling_freq: str = '1H') -> pd.Series:
    """
    Calculate realized range volatility using intraday high-low.
    
    This estimator works best with high-frequency data.
    
    Args:
        df: DataFrame with high and low prices (may include intraday data)
        window: Rolling window size (in days)
        min_periods: Minimum number of observations
        sampling_freq: Frequency for subsampling intraday data
        
    Returns:
        Series with realized range volatility
    """
    # Check if high and low columns exist
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns for realized range")
    
    # Check if we have intraday data
    has_intraday = False
    if isinstance(df.index, pd.DatetimeIndex):
        # Check if we have multiple observations per day
        date_only = df.index.normalize()
        if len(date_only.unique()) < len(df):
            has_intraday = True
    
    if has_intraday:
        # Resample to desired frequency and calculate range
        high_resampled = df['high'].resample(sampling_freq).max()
        low_resampled = df['low'].resample(sampling_freq).min()
        
        # Calculate squared log range
        log_range = np.log(high_resampled / low_resampled)
        sq_log_range = log_range**2
        
        # Scale by factor and aggregate to daily
        factor = 1.0 / (4.0 * np.log(2.0))
        daily_range_var = sq_log_range * factor
        daily_range_vol = np.sqrt(daily_range_var.resample('D').sum())
        
        # Apply rolling window at daily level
        result = daily_range_vol.rolling(window=window, min_periods=min_periods).mean()
    else:
        # If no intraday data, fall back to Parkinson estimator
        logger.warning("No intraday data detected. Falling back to Parkinson estimator.")
        result = parkinson_vol(df, window, min_periods)
    
    return result


def realized_variance_vol(df: pd.DataFrame, 
                        window: int = 21, 
                        min_periods: int = 5,
                        return_col: str = 'close',
                        return_type: str = 'log',
                        sampling_freq: str = None) -> pd.Series:
    """
    Calculate realized volatility from high-frequency returns.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size (in days)
        min_periods: Minimum number of observations
        return_col: Column to use for returns
        return_type: Type of returns to use
        sampling_freq: Optional frequency for resampling
        
    Returns:
        Series with realized volatility
    """
    # Check if required column exists
    if return_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{return_col}' column for realized variance")
    
    # Calculate returns
    if sampling_freq and isinstance(df.index, pd.DatetimeIndex):
        # Resample first then calculate returns
        price_resampled = df[return_col].resample(sampling_freq).last()
        if return_type == 'log':
            returns = np.log(price_resampled / price_resampled.shift(1))
        else:
            returns = price_resampled.pct_change()
    else:
        # Calculate returns directly
        returns = calculate_returns(df, return_col, return_type)
    
    # Check if we have intraday data
    has_intraday = False
    if isinstance(returns.index, pd.DatetimeIndex):
        date_only = returns.index.normalize()
        if len(date_only.unique()) < len(returns):
            has_intraday = True
    
    if has_intraday:
        # Square returns
        squared_returns = returns**2
        
        # Aggregate to daily realized variance
        daily_rv = squared_returns.groupby(squared_returns.index.date).sum()
        daily_rv.index = pd.DatetimeIndex(daily_rv.index)
        
        # Take square root for volatility
        daily_rvol = np.sqrt(daily_rv)
        
        # Apply rolling window
        result = daily_rvol.rolling(window=window, min_periods=min_periods).mean()
    else:
        # If no intraday data, fall back to standard deviation
        logger.warning("No intraday data detected. Falling back to standard deviation.")
        result = standard_deviation(returns, window, min_periods)
    
    return result


def realized_kernel_vol(df: pd.DataFrame, 
                       window: int = 21, 
                       min_periods: int = 5,
                       return_col: str = 'close',
                       return_type: str = 'log',
                       sampling_freq: str = None,
                       kernel_type: str = 'bartlett',
                       bandwidth: int = None) -> pd.Series:
    """
    Calculate realized kernel volatility with noise correction.
    
    Args:
        df: DataFrame with price data
        window: Rolling window size (in days)
        min_periods: Minimum number of observations
        return_col: Column to use for returns
        return_type: Type of returns to use
        sampling_freq: Optional frequency for resampling
        kernel_type: Type of kernel ('bartlett', 'flat_top', 'epanechnikov')
        bandwidth: Kernel bandwidth (auto if None)
        
    Returns:
        Series with realized kernel volatility
    """
    # Check if required column exists
    if return_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{return_col}' column for realized kernel")
    
    # Calculate returns
    if sampling_freq and isinstance(df.index, pd.DatetimeIndex):
        # Resample first then calculate returns
        price_resampled = df[return_col].resample(sampling_freq).last()
        if return_type == 'log':
            returns = np.log(price_resampled / price_resampled.shift(1))
        else:
            returns = price_resampled.pct_change()
    else:
        # Calculate returns directly
        returns = calculate_returns(df, return_col, return_type)
    
    # Check if we have intraday data
    has_intraday = False
    if isinstance(returns.index, pd.DatetimeIndex):
        date_only = returns.index.normalize()
        if len(date_only.unique()) < len(returns):
            has_intraday = True
    
    if not has_intraday:
        # If no intraday data, fall back to standard deviation
        logger.warning("No intraday data detected. Falling back to standard deviation.")
        return standard_deviation(returns, window, min_periods)
    
    # Define kernel functions
    def bartlett_kernel(x, h):
        return np.maximum(0, 1 - np.abs(x) / h)
    
    def flat_top_kernel(x, h):
        return np.where(np.abs(x) <= h/2, 1, 
                       np.where(np.abs(x) <= h, 2 - 2*np.abs(x)/h, 0))
    
    def epanechnikov_kernel(x, h):
        return np.maximum(0, 1 - (x/h)**2)
    
    kernel_funcs = {
        'bartlett': bartlett_kernel,
        'flat_top': flat_top_kernel,
        'epanechnikov': epanechnikov_kernel
    }
    
    if kernel_type not in kernel_funcs:
        raise ValueError(f"Kernel type '{kernel_type}' not supported. Choose from: {list(kernel_funcs.keys())}")
    
    kernel_func = kernel_funcs[kernel_type]
    
    # Function to compute realized kernel for one day
    def compute_daily_rk(day_returns):
        n = len(day_returns)
        if n < min_periods:
            return np.nan
        
        # Automatic bandwidth selection if not provided
        h = bandwidth if bandwidth is not None else int(4 * (n/100)**(2/5))
        h = min(h, n-1)  # Ensure bandwidth is valid
        
        # Compute autocovariances
        gamma0 = np.sum(day_returns**2)  # Realized variance
        gammas = np.zeros(h+1)
        gammas[0] = gamma0
        
        for k in range(1, h+1):
            gammas[k] = np.sum(day_returns[k:] * day_returns[:-k])
        
        # Apply kernel weights
        weights = np.array([kernel_func(k/h, 1) for k in range(1, h+1)])
        
        # Realized kernel estimator
        rk = gamma0 + 2 * np.sum(weights * gammas[1:])
        
        return np.sqrt(rk)
    
    # Split returns by day and compute RK for each day
    daily_rk = []
    day_groups = returns.groupby(returns.index.date)
    
    for day, group in day_groups:
        if len(group) >= min_periods:
            daily_rk.append((day, compute_daily_rk(group.dropna().values)))
    
    # Create daily series
    daily_rk_series = pd.Series([rk for _, rk in daily_rk], index=[pd.Timestamp(day) for day, _ in daily_rk])
    
    # Apply rolling window
    result = daily_rk_series.rolling(window=window, min_periods=min_periods).mean()
    
    return result


# -----------------------------------------------------------------------------
# Main Calculation Function
# -----------------------------------------------------------------------------

def calculate_realized_vol(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate realized volatility with specified estimator.
    
    Args:
        df: DataFrame with price data
        params: Dictionary with parameters
        
    Returns:
        DataFrame with volatility estimates and confidence intervals
    """
    # Extract parameters
    estimator = params.get('estimator', DEFAULT_PARAMS['estimator'])
    window = params.get('window', DEFAULT_PARAMS['window'])
    min_periods = params.get('min_periods', DEFAULT_PARAMS['min_periods'])
    return_type = params.get('return_type', DEFAULT_PARAMS['return_type'])
    target_column = params.get('target_column', DEFAULT_PARAMS['target_column'])
    annualize = params.get('annualize', DEFAULT_PARAMS['annualize'])
    trading_days = params.get('trading_days', DEFAULT_PARAMS['trading_days'])
    alpha = params.get('alpha', DEFAULT_PARAMS['alpha'])
    use_high_low = params.get('use_high_low', DEFAULT_PARAMS['use_high_low'])
    use_jump_filter = params.get('use_jump_filter', DEFAULT_PARAMS['use_jump_filter'])
    jump_threshold = params.get('jump_threshold', DEFAULT_PARAMS['jump_threshold'])
    
    # Calculate returns if needed
    if estimator in ['standard', 'hodges_tompkins']:
        returns = calculate_returns(df, target_column, return_type)
        
        if use_jump_filter:
            returns = filter_jumps(returns, window, jump_threshold)
    
    # Choose estimator
    if estimator == 'standard':
        vol = standard_deviation(returns, window, min_periods)
    
    elif estimator == 'parkinson' and use_high_low:
        vol = parkinson_vol(df, window, min_periods)
    
    elif estimator == 'garman_klass' and use_high_low:
        vol = garman_klass_vol(df, window, min_periods)
    
    elif estimator == 'rogers_satchell' and use_high_low:
        vol = rogers_satchell_vol(df, window, min_periods)
    
    elif estimator == 'yang_zhang' and use_high_low:
        vol = yang_zhang_vol(df, window, min_periods)
    
    elif estimator == 'hodges_tompkins':
        vol = hodges_tompkins_vol(returns, window, min_periods)
    
    elif estimator == 'realized_range' and use_high_low:
        sampling_freq = params.get('sampling_freq', '1H')
        vol = realized_range_vol(df, window, min_periods, sampling_freq)
    
    elif estimator == 'realized_variance':
        sampling_freq = params.get('sampling_freq', None)
        vol = realized_variance_vol(df, window, min_periods, target_column, return_type, sampling_freq)
    
    elif estimator == 'realized_kernel':
        sampling_freq = params.get('sampling_freq', None)
        kernel_type = params.get('kernel_type', 'bartlett')
        bandwidth = params.get('bandwidth', None)
        vol = realized_kernel_vol(
            df, window, min_periods, target_column, return_type, 
            sampling_freq, kernel_type, bandwidth
        )
    
    else:
        # Fall back to standard if estimator not found or high/low not available
        if use_high_low and estimator in ['parkinson', 'garman_klass', 'rogers_satchell', 
                                        'yang_zhang', 'realized_range']:
            logger.warning(f"High/low data required but not available. Falling back to standard estimator.")
        else:
            logger.warning(f"Unknown estimator: {estimator}. Falling back to standard estimator.")
        
        returns = calculate_returns(df, target_column, return_type)
        
        if use_jump_filter:
            returns = filter_jumps(returns, window, jump_threshold)
        
        vol = standard_deviation(returns, window, min_periods)
    
    # Annualize if needed
    if annualize:
        vol = annualize_volatility(vol, df, trading_days)
    
    # Calculate confidence intervals
    ci_dist = params.get('ci_dist', 'normal')
    ci = calculate_confidence_interval(vol, window, alpha, ci_dist)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'volatility': vol,
        'volatility_lower': ci['lower'],
        'volatility_upper': ci['upper']
    })
    
    return result


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------

def generate_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """
    Generate realized volatility signals from price data.
    
    This is the main function that should be called by the analytics pipeline.
    
    Args:
        df: DataFrame with price data
        **params: Dictionary with model parameters
            - estimator: Volatility estimator method
            - window: Rolling window size
            - min_periods: Minimum periods for calculation
            - return_type: Return calculation method
            - annualize: Whether to annualize volatility
            - trading_days: Number of trading days in a year
            - alpha: Significance level for CI
            - target_column: Column to use for pricing
            - use_high_low: Use high/low prices if available
            - use_jump_filter: Apply jump filtering
            - jump_threshold: Z-score threshold for jumps
            - output_file: Path to save the signals (optional)
        
    Returns:
        DataFrame with volatility signals or list of Signal objects
    """
    logger.info(f"Generating realized volatility signals with parameters: {params}")
    
    # Prepare DataFrame
    target_column = params.get('target_column', DEFAULT_PARAMS['target_column'])
    df = prepare_dataframe(df, target_column)
    
    # Merge parameters with defaults
    model_params = DEFAULT_PARAMS.copy()
    model_params.update(params)
    
    # Generate volatility estimates
    volatility = calculate_realized_vol(df, model_params)
    
    # Format as signals
    signals = format_signals(volatility, df, model_params, 'realized_volatility', model_params['estimator'])
    
    # Save and/or return as objects
    return save_signals(signals, model_params, model_params.get('as_objects', False))


# -----------------------------------------------------------------------------
# Convenience Functions for Specific Estimators
# -----------------------------------------------------------------------------

def generate_standard_vol_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate standard volatility signal."""
    params['estimator'] = 'standard'
    return generate_signal(df, **params)


def generate_parkinson_vol_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate Parkinson volatility signal."""
    params['estimator'] = 'parkinson'
    return generate_signal(df, **params)


def generate_garman_klass_vol_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate Garman-Klass volatility signal."""
    params['estimator'] = 'garman_klass'
    return generate_signal(df, **params)


def generate_rogers_satchell_vol_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate Rogers-Satchell volatility signal."""
    params['estimator'] = 'rogers_satchell'
    return generate_signal(df, **params)


def generate_yang_zhang_vol_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate Yang-Zhang volatility signal."""
    params['estimator'] = 'yang_zhang'
    return generate_signal(df, **params)


def generate_realized_variance_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate realized variance volatility signal."""
    params['estimator'] = 'realized_variance'
    return generate_signal(df, **params)


def generate_realized_kernel_signal(df: pd.DataFrame, **params) -> Union[pd.DataFrame, List[Signal]]:
    """Generate realized kernel volatility signal."""
    params['estimator'] = 'realized_kernel'
    return generate_signal(df, **params)


# -----------------------------------------------------------------------------
# Command-line Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate realized volatility signals")
    parser.add_argument("--input", required=True, help="Input price data file (CSV or Parquet)")
    parser.add_argument("--output", required=True, help="Output signals file (Parquet)")
    parser.add_argument("--estimator", default="standard", choices=list(ESTIMATOR_METHODS.keys()), 
                        help="Volatility estimator method")
    parser.add_argument("--window", type=int, default=21, help="Rolling window size")
    parser.add_argument("--annualize", action="store_true", help="Annualize volatility")
    
    args = parser.parse_args()
    
    # Load data
    if args.input.endswith('.csv'):
        data = pd.read_csv(args.input, parse_dates=['timestamp']).set_index('timestamp')
    else:
        data = pd.read_parquet(args.input)
    
    # Generate signals
    signals = generate_signal(
        data, 
        estimator=args.estimator,
        window=args.window,
        annualize=args.annualize,
        output_file=args.output
    )
    
    print(f"Generated {len(signals)} realized volatility signals")