"""
Financial Calculations

This module provides functions for calculating common financial metrics from
time series data, including returns, volatility, drawdowns, and momentum.
"""

# Standard library imports
import logging
from typing import Optional, Union

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
        method: Calculation method ('std', 'garch', 'parkinson', etc.)
        col_name: Column name if returns is a DataFrame
        
    Returns:
        Series or DataFrame with volatility values
    """
    # Extract Series from DataFrame if column specified
    if isinstance(returns, pd.DataFrame) and col_name is not None:
        if col_name not in returns.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        return_data = returns[col_name]
    else:
        return_data = returns
    
    # Calculate volatility
    if method == 'std':
        vol = return_data.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            vol = vol * np.sqrt(trading_days)
    else:
        # For more complex methods, we could implement or call other modules
        raise ValueError(f"Volatility method '{method}' not implemented yet. Use 'std'.")
    
    return vol