"""
Financial Calculations

This module provides functions for calculating common financial metrics from
time series data, including returns, volatility, drawdowns, and momentum.
"""

# Standard library imports
import logging
from typing import Optional, Union, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data.financial_metrics")

def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'log',
    periods: int = 1,
    col_name: Optional[str] = None,
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


def calculate_momentum(
    prices: Union[pd.Series, pd.DataFrame],
    window: int = 20,
    method: str = 'ratio',
    col_name: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate price momentum.
    
    Args:
        prices: Price data (Series or DataFrame)
        window: Lookback window for momentum calculation
        method: Calculation method ('ratio', 'diff', 'log')
        col_name: Column name if prices is a DataFrame
        
    Returns:
        Series or DataFrame with momentum values
    """
    # Extract Series from DataFrame if column specified
    if isinstance(prices, pd.DataFrame) and col_name is not None:
        if col_name not in prices.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        price_data = prices[col_name]
    else:
        price_data = prices
    
    # Calculate momentum based on method
    if method == 'ratio':
        momentum = price_data / price_data.shift(window) - 1
    elif method == 'diff':
        momentum = price_data - price_data.shift(window)
    elif method == 'log':
        momentum = np.log(price_data / price_data.shift(window))
    else:
        raise ValueError(f"Invalid momentum method: {method}. Use 'ratio', 'diff', or 'log'")
    
    return momentum


def calculate_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
    method: str = 'std',
    col_name: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate return volatility.
    
    Args:
        returns: Return data (Series or DataFrame)
        window: Lookback window for volatility calculation
        annualize: Whether to annualize the result
        trading_days: Number of trading days in a year
        method: Calculation method ('std', 'garch', 'parkinson', 'garman_klass')
        col_name: Column name if returns is a DataFrame
        
    Returns:
        Series or DataFrame with volatility values
        
    Raises:
        ValueError: If an invalid method is specified or required columns are missing
    """
    # Extract Series from DataFrame if column specified
    if isinstance(returns, pd.DataFrame) and col_name is not None:
        if col_name not in returns.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        return_data = returns[col_name]
    else:
        return_data = returns
    
    # Calculate volatility based on method
    if method == 'std':
        vol = return_data.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            vol = vol * np.sqrt(trading_days)
            
    elif method == 'parkinson':
        # For Parkinson estimator, we need high/low data
        if isinstance(returns, pd.DataFrame):
            if not all(col in returns.columns for col in ['high', 'low']):
                raise ValueError("Parkinson volatility requires 'high' and 'low' columns")
            vol = parkinson_volatility(
                returns,
                window=window,
                annualize=annualize,
                trading_days=trading_days
            )
        else:
            raise ValueError("Parkinson volatility requires a DataFrame with 'high' and 'low' columns")
            
    elif method == 'garman_klass':
        # For Garman-Klass estimator, we need OHLC data
        if isinstance(returns, pd.DataFrame):
            if not all(col in returns.columns for col in ['open', 'high', 'low', 'close']):
                raise ValueError("Garman-Klass volatility requires 'open', 'high', 'low', and 'close' columns")
            vol = garman_klass_volatility(
                returns,
                window=window,
                annualize=annualize,
                trading_days=trading_days
            )
        else:
            raise ValueError("Garman-Klass volatility requires a DataFrame with OHLC columns")
            
    elif method == 'garch':
        try:
            from arch import arch_model
            
            # Handle both Series and DataFrame
            data = return_data.dropna().values
            
            # Fit GARCH(1,1) model
            model = arch_model(data, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Extract conditional volatility
            conditional_vol = pd.Series(
                model_fit.conditional_volatility,
                index=return_data.dropna().index
            )
            
            # Reindex to match original data
            vol = pd.Series(index=return_data.index, dtype=float)
            vol.loc[conditional_vol.index] = conditional_vol
            
            # Annualize if requested
            if annualize:
                vol = vol * np.sqrt(trading_days)
                
        except ImportError:
            raise ImportError("GARCH volatility requires the 'arch' package. Install it with 'pip install arch'")
    else:
        raise ValueError(f"Volatility method '{method}' not implemented. Use 'std', 'parkinson', 'garman_klass', or 'garch'.")
    
    return vol


def parkinson_volatility(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator using high-low range.
    
    Parkinson volatility uses the high-low range to estimate volatility,
    and is more efficient than standard deviation for estimating volatility
    when the true underlying process is a geometric Brownian motion.
    
    Args:
        df: DataFrame with high and low prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        window: Rolling window size
        annualize: Whether to annualize the result
        trading_days: Number of trading days per year
        
    Returns:
        Series with Parkinson volatility estimates
    """
    # Calculate log high/low range
    log_hl = np.log(df[high_col] / df[low_col])
    squared_log_range = log_hl**2
    
    # Parkinson estimator with scaling factor 1/(4*ln(2))
    scaling_factor = 1.0 / (4.0 * np.log(2.0))
    vol = np.sqrt(
        (squared_log_range * scaling_factor).rolling(window=window).mean()
    )
    
    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def garman_klass_volatility(
    df: pd.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator using OHLC data.
    
    Garman-Klass volatility is an estimator that incorporates open-high-low-close
    data to provide a more efficient estimate than close-to-close volatility.
    
    Args:
        df: DataFrame with OHLC data
        open_col: Column name for open prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
        window: Rolling window size
        annualize: Whether to annualize the result
        trading_days: Number of trading days per year
        
    Returns:
        Series with Garman-Klass volatility estimates
    """
    # Calculate log ranges
    log_hl = np.log(df[high_col] / df[low_col])
    log_co = np.log(df[close_col] / df[open_col])
    
    # Garman-Klass components
    hl_part = 0.5 * log_hl**2
    co_part = (2 * np.log(2) - 1) * log_co**2
    
    # Full estimator
    estimator = hl_part - co_part
    
    # Calculate rolling volatility
    vol = np.sqrt(estimator.rolling(window=window).mean())
    
    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def calculate_adx(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    window: int = 14
) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) indicator.
    
    The ADX measures trend strength regardless of direction.
    
    Args:
        df: DataFrame with high, low, and close prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
        window: Window size for calculations
        
    Returns:
        Series with ADX values
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate price changes
    result['high_change'] = result[high_col] - result[high_col].shift(1)
    result['low_change'] = result[low_col].shift(1) - result[low_col]
    
    # Calculate Directional Movement
    result['plus_dm'] = np.where(
        (result['high_change'] > result['low_change']) & (result['high_change'] > 0),
        result['high_change'],
        0
    )
    
    result['minus_dm'] = np.where(
        (result['low_change'] > result['high_change']) & (result['low_change'] > 0),
        result['low_change'],
        0
    )
    
    # Calculate True Range
    result['tr'] = np.maximum(
        np.maximum(
            result[high_col] - result[low_col],
            np.abs(result[high_col] - result[close_col].shift(1))
        ),
        np.abs(result[low_col] - result[close_col].shift(1))
    )
    
    # Smooth with EMA
    result['smoothed_tr'] = result['tr'].rolling(window=window).mean()
    result['smoothed_plus_dm'] = result['plus_dm'].rolling(window=window).mean()
    result['smoothed_minus_dm'] = result['minus_dm'].rolling(window=window).mean()
    
    # Calculate Directional Indicators
    result['plus_di'] = 100 * result['smoothed_plus_dm'] / result['smoothed_tr']
    result['minus_di'] = 100 * result['smoothed_minus_dm'] / result['smoothed_tr']
    
    # Calculate Directional Index
    result['dx'] = 100 * np.abs(result['plus_di'] - result['minus_di']) / (result['plus_di'] + result['minus_di'])
    
    # Calculate ADX (smoothed DX)
    adx = result['dx'].rolling(window=window).mean()
    
    # Clean up intermediate columns if needed
    return adx