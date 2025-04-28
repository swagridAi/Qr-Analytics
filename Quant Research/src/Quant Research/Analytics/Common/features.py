"""
Feature Engineering Module

This module provides specialized feature engineering functions for financial time series data.
Unlike the core data_utils.py which provides fundamental operations, this module focuses on:

1. Creating domain-specific features for financial analysis
2. Building comprehensive feature sets for different analysis types
3. Implementing more advanced feature transformations
4. Providing higher-level feature engineering workflows

Features are organized into categories:
- Price-based features (patterns, trends, support/resistance)
- Technical indicators (specialized beyond basic moving averages)
- Volatility features (regime-specific, forward-looking)
- Market microstructure features
- Cross-sectional features (relative performance, industry-specific)
- Alternative data integration features

Usage:
    ```python
    from quant_research.analytics.common.features import (
        # Feature categories
        add_technical_features,
        add_trend_features,
        add_volatility_features,
        add_microstructure_features,
        
        # Feature sets
        create_momentum_features,
        create_mean_reversion_features,
        create_ml_feature_set
    )
    
    # Add technical indicators to a DataFrame
    df = add_technical_features(df, include=['rsi', 'macd', 'bbands'])
    
    # Create a comprehensive feature set for ML
    ml_features = create_ml_feature_set(
        df, 
        feature_types=['technical', 'volatility', 'trend'],
        lookback_windows=[5, 10, 20, 60]
    )
    ```
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Local imports - now importing from data_utils
from quant_research.analytics.common.data_utils import (
    # Financial calculations
    calculate_returns,
    calculate_momentum,
    calculate_volatility,
    calculate_zscore,
    
    # Feature engineering utilities
    add_lagged_features,
    add_difference_features,
    add_rolling_features,
    
    # Data normalization 
    normalize_data
)

# Configure logger
logger = logging.getLogger("quant_research.analytics.common.features")

#------------------------------------------------------------------------
# Technical Indicators
#------------------------------------------------------------------------

def add_technical_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume',
    window: int = 20,
    include: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add common technical indicators as features.
    
    This function creates technical analysis indicators based on price and volume data.
    It can create a comprehensive set of indicators or a focused subset.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        volume_col: Column name for volume data (Optional)
        window: Base window size for indicators
        include: List of indicators to include (defaults to all)
        
    Returns:
        DataFrame with added technical features
        
    Raises:
        ValueError: If price_col doesn't exist in the DataFrame
        
    Available indicators:
        - 'returns': Price returns
        - 'ma': Moving Average
        - 'ema': Exponential Moving Average
        - 'momentum': Price momentum 
        - 'volatility': Return volatility
        - 'rsi': Relative Strength Index
        - 'macd': Moving Average Convergence Divergence
        - 'bbands': Bollinger Bands (upper, lower, width)
        - 'volume_ma': Volume Moving Average
        - 'volume_ratio': Volume to Moving Average ratio
        - 'obv': On-Balance Volume
    
    Examples:
        ```python
        # Add all technical indicators
        df = add_technical_features(df)
        
        # Add only selected indicators
        df = add_technical_features(df, include=['rsi', 'macd', 'bbands'])
        ```
    """
    # Validate required columns
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # All available indicators
    all_indicators = [
        'returns', 'ma', 'ema', 'momentum', 'volatility',
        'rsi', 'macd', 'bbands', 'stoch'
    ]
    
    # If volume is available, add volume indicators
    if volume_col in df.columns:
        all_indicators.extend(['volume_ma', 'volume_ratio', 'obv'])
    
    # Filter indicators if specified, or use all
    if include is None:
        indicators = all_indicators
    else:
        # Group related indicators
        expanded_include = []
        for ind in include:
            if ind == 'bbands':
                expanded_include.extend(['bb_upper', 'bb_lower', 'bb_width'])
            elif ind == 'macd':
                expanded_include.extend(['macd', 'macd_signal', 'macd_hist'])
            elif ind == 'stoch':
                expanded_include.extend(['stoch_k', 'stoch_d'])
            else:
                expanded_include.append(ind)
        
        indicators = [ind for ind in expanded_include if ind in all_indicators]
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate returns if needed for indicators or explicitly requested
    if 'returns' in indicators or any(ind in indicators for ind in ['momentum', 'volatility', 'rsi']):
        result['returns'] = calculate_returns(result[price_col], method='pct')
    
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
    
    if any(ind in indicators for ind in ['macd', 'macd_signal', 'macd_hist']):
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
    
    if any(k in indicators for k in ['stoch_k', 'stoch_d']):
        # Calculate Stochastic Oscillator
        if all(col in df.columns for col in ['high', 'low']):
            # %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
            low_min = result['low'].rolling(window=window).min()
            high_max = result['high'].rolling(window=window).max()
            
            # Handle division by zero
            denominator = high_max - low_min
            denominator = denominator.replace(0, np.nan)
            
            result['stoch_k'] = 100 * ((result[price_col] - low_min) / denominator)
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
    
    # Volume-based indicators (if volume data available)
    if volume_col in df.columns:
        if 'volume_ma' in indicators:
            result[f'volume_ma_{window}'] = result[volume_col].rolling(window=window).mean()
        
        if 'volume_ratio' in indicators:
            vol_ma = result[volume_col].rolling(window=window).mean()
            result[f'volume_ratio_{window}'] = result[volume_col] / vol_ma
        
        if 'obv' in indicators:
            # Calculate On-Balance Volume
            obv = pd.Series(0, index=result.index)
            
            for i in range(1, len(result)):
                if result[price_col].iloc[i] > result[price_col].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + result[volume_col].iloc[i]
                elif result[price_col].iloc[i] < result[price_col].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - result[volume_col].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            result['obv'] = obv
    
    # Log generated indicators
    added_indicators = [col for col in result.columns if col not in df.columns]
    logger.info(f"Added {len(added_indicators)} technical indicators")
    
    return result


def add_trend_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    windows: Optional[List[int]] = None,
    include_crosses: bool = True
) -> pd.DataFrame:
    """
    Add trend-following features and moving average crosses.
    
    This function creates features that help identify and analyze trends in price data,
    including moving averages, price to moving average ratios, and moving average crossovers.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        windows: List of moving average window sizes (default [20, 50, 200])
        include_crosses: Whether to include moving average crossover indicators
        
    Returns:
        DataFrame with added trend features
    
    Examples:
        ```python
        # Standard trend features with default windows
        df = add_trend_features(df)
        
        # Custom windows without crossovers
        df = add_trend_features(
            df, 
            windows=[10, 30, 90, 200],
            include_crosses=False
        )
        ```
    """
    # Default windows if not provided
    if windows is None:
        windows = [20, 50, 200]
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate moving averages for each window
    for window in windows:
        result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()
        
        # Add price to moving average ratio
        result[f'price_to_ma_{window}'] = result[price_col] / result[f'ma_{window}']
        
        # Calculate distance from moving average in standard deviations
        # (aka Bollinger Band position)
        ma_std = result[price_col].rolling(window=window).std()
        result[f'ma_{window}_distance'] = (result[price_col] - result[f'ma_{window}']) / ma_std
    
    # Add moving average crossovers if requested
    if include_crosses and len(windows) >= 2:
        # Sort windows to ensure consistent naming
        sorted_windows = sorted(windows)
        
        # Create crossover indicators for adjacent window pairs
        for i in range(len(sorted_windows) - 1):
            fast_window = sorted_windows[i]
            slow_window = sorted_windows[i + 1]
            
            # Moving average ratio
            result[f'ma_{fast_window}_to_ma_{slow_window}'] = (
                result[f'ma_{fast_window}'] / result[f'ma_{slow_window}']
            )
            
            # Crossover signal (-1 when fast crosses below slow, +1 when above)
            fast_above_slow = result[f'ma_{fast_window}'] > result[f'ma_{slow_window}']
            cross = fast_above_slow.astype(int).diff().fillna(0)
            result[f'ma_{fast_window}_{slow_window}_cross'] = cross
            
            # Golden/Death cross binary indicators (commonly used in technical analysis)
            if fast_window <= 50 and slow_window >= 200:
                result['golden_cross'] = (
                    (result[f'ma_{fast_window}'] > result[f'ma_{slow_window}']) &
                    (result[f'ma_{fast_window}'].shift(1) <= result[f'ma_{slow_window}'].shift(1))
                ).astype(int)
                
                result['death_cross'] = (
                    (result[f'ma_{fast_window}'] < result[f'ma_{slow_window}']) &
                    (result[f'ma_{fast_window}'].shift(1) >= result[f'ma_{slow_window}'].shift(1))
                ).astype(int)
    
    # Add directional movement indicators
    result[f'adx_{windows[0]}'] = calculate_adx(
        df, high_col='high', low_col='low', close_col='close', window=windows[0]
    ) if all(col in df.columns for col in ['high', 'low', 'close']) else None
    
    # Log generated indicators
    added_indicators = [col for col in result.columns if col not in df.columns]
    logger.info(f"Added {len(added_indicators)} trend indicators")
    
    return result


#------------------------------------------------------------------------
# Volatility Features
#------------------------------------------------------------------------

def add_volatility_features(
    df: pd.DataFrame,
    returns_col: str = 'returns',
    price_col: str = 'close',
    windows: Optional[List[int]] = None,
    use_log_scale: bool = False,
    annualize: bool = True,
    trading_days: int = 252,
    include_garch: bool = False
) -> pd.DataFrame:
    """
    Add various volatility features to assess market conditions.
    
    This function creates a comprehensive set of volatility indicators
    that can be used to identify regimes, risk levels, and forecast volatility.
    
    Args:
        df: Input DataFrame with return data
        returns_col: Column name for returns
        price_col: Column name for price data
        windows: List of window sizes for calculations
        use_log_scale: Whether to apply log transformation to volatility
        annualize: Whether to annualize volatility
        trading_days: Number of trading days per year
        include_garch: Whether to include GARCH volatility estimates
        
    Returns:
        DataFrame with added volatility features
    
    Examples:
        ```python
        # Basic volatility features
        df = add_volatility_features(df)
        
        # Advanced volatility modeling including GARCH
        df = add_volatility_features(
            df,
            windows=[5, 21, 63, 252],  # 1wk, 1mo, 3mo, 1yr
            include_garch=True
        )
        ```
    """
    # Default windows if not provided
    if windows is None:
        windows = [5, 20, 60]
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Check if returns column exists, calculate if it doesn't
    if returns_col not in result.columns and price_col in result.columns:
        result[returns_col] = calculate_returns(result[price_col], method='pct')
        logger.info("Calculated returns for volatility features")
    
    # Add standard rolling volatility for each window
    for window in windows:
        vol_col = f'volatility_{window}'
        result[vol_col] = calculate_volatility(
            result[returns_col], 
            window=window, 
            annualize=annualize, 
            trading_days=trading_days
        )
        
        # Apply log transformation if requested
        if use_log_scale:
            result[vol_col] = np.log(result[vol_col])
        
        # Volatility of volatility (meta-volatility)
        result[f'vol_of_vol_{window}'] = calculate_volatility(
            calculate_returns(result[vol_col], method='pct'),
            window=window,
            annualize=False
        )
        
        # Volatility trend (is volatility increasing or decreasing)
        result[f'vol_trend_{window}'] = (
            result[vol_col] / result[vol_col].rolling(window=window//2).mean() - 1
        )
        
        # Z-score of volatility (normalized volatility)
        result[f'vol_zscore_{window}'] = calculate_zscore(
            result[vol_col], 
            window=min(window*3, len(result)//2)
        )
    
    # Add range-based volatility estimators if OHLC data is available
    if all(col in result.columns for col in ['high', 'low']):
        # Parkinson estimator (uses high-low range)
        result['parkinson_vol'] = parkinson_volatility(
            result,
            window=windows[0],
            annualize=annualize,
            trading_days=trading_days
        )
        
        # Add Garman-Klass if open data available
        if 'open' in result.columns:
            result['garman_klass_vol'] = garman_klass_volatility(
                result,
                window=windows[0],
                annualize=annualize,
                trading_days=trading_days
            )
    
    # Add GARCH volatility if requested
    if include_garch:
        try:
            from arch import arch_model
            
            # Fit a simple GARCH(1,1) model
            returns = result[returns_col].dropna().values
            
            if len(returns) > 100:  # Need sufficient data for GARCH
                model = arch_model(returns, vol='Garch', p=1, q=1)
                model_fit = model.fit(disp='off')
                
                # Get conditional volatility
                garch_vol = model_fit.conditional_volatility
                
                # Align with original DataFrame
                vol_series = pd.Series(
                    garch_vol, 
                    index=result.index[-len(garch_vol):]
                )
                
                # Annualize if requested
                if annualize:
                    vol_series = vol_series * np.sqrt(trading_days)
                
                result['garch_vol'] = vol_series
                
                logger.info("Added GARCH volatility estimates")
            else:
                logger.warning("Insufficient data for GARCH modeling")
        except ImportError:
            logger.warning("arch package not available for GARCH modeling")
    
    # Log generated indicators
    added_indicators = [col for col in result.columns if col not in df.columns]
    logger.info(f"Added {len(added_indicators)} volatility indicators")
    
    return result


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
    
    # Clean up intermediate columns
    return adx


#------------------------------------------------------------------------
# ZScore Features
#------------------------------------------------------------------------

def add_zscore_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Add z-score features for multiple columns and windows.
    
    Z-scores are particularly useful for mean-reversion strategies and
    identifying extreme values relative to recent history.
    
    Args:
        df: Input DataFrame
        columns: Columns to calculate z-scores for
        windows: List of window sizes (default [20, 60, 120])
        methods: List of calculation methods (default ['rolling', 'ewm'])
        prefix: Prefix for column names
        
    Returns:
        DataFrame with added z-score features
    
    Examples:
        ```python
        # Standard z-scores for price and volume
        df = add_zscore_features(df, columns=['close', 'volume'])
        
        # Specialized z-scores for technical indicators
        df = add_zscore_features(
            df, 
            columns=['rsi_14', 'macd', 'bbands_width'],
            windows=[10, 30, 60],
            methods=['rolling', 'ewm', 'expanding']
        )
        ```
    """
    # Default values if not provided
    if windows is None:
        windows = [20, 60, 120]
        
    if methods is None:
        methods = ['rolling']
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        for window in windows:
            for method in methods:
                # Calculate z-score
                zscore = calculate_zscore(result[col], window=window, method=method)
                
                # Create column name
                if prefix:
                    col_name = f"{prefix}_{col}_zscore_{method}_{window}"
                else:
                    col_name = f"{col}_zscore_{method}_{window}"
                
                # Add to result
                result[col_name] = zscore
    
    # Log generated indicators
    added_indicators = [col for col in result.columns if col not in df.columns]
    logger.info(f"Added {len(added_indicators)} z-score features")
    
    return result


def add_percentile_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: Optional[List[int]] = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Add percentile rank features.
    
    Percentile ranks provide a normalized measure of where the current value
    stands relative to its historical distribution.
    
    Args:
        df: Input DataFrame
        columns: Columns to calculate percentiles for
        windows: List of window sizes (default [20, 60, 120])
        prefix: Prefix for column names
        
    Returns:
        DataFrame with added percentile features
    
    Examples:
        ```python
        # Add percentile ranks for volume
        df = add_percentile_features(df, columns=['volume'])
        ```
    """
    # Default windows if not provided
    if windows is None:
        windows = [20, 60, 120]
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        for window in windows:
            # Calculate rolling rank (percentile)
            def rolling_percentile(x):
                return pd.Series(x).rank(pct=True).iloc[-1]
            
            percentile = result[col].rolling(window).apply(rolling_percentile, raw=False)
            
            # Create column name
            if prefix:
                col_name = f"{prefix}_{col}_percentile_{window}"
            else:
                col_name = f"{col}_percentile_{window}"
            
            # Add to result
            result[col_name] = percentile
    
    # Log generated indicators
    added_indicators = [col for col in result.columns if col not in df.columns]
    logger.info(f"Added {len(added_indicators)} percentile features")
    
    return result


#------------------------------------------------------------------------
# Market Microstructure Features
#------------------------------------------------------------------------

def add_microstructure_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume',
    window: int = 20
) -> pd.DataFrame:
    """
    Add market microstructure features for high-frequency analysis.
    
    These features help analyze market behavior at a more granular level,
    useful for high-frequency trading and market quality assessment.
    
    Args:
        df: Input DataFrame with price and volume data
        price_col: Column name for price data
        volume_col: Column name for volume data
        window: Window size for calculations
        
    Returns:
        DataFrame with added microstructure features
    
    Examples:
        ```python
        # Add microstructure features to tick data
        tick_df = add_microstructure_features(tick_df)
        ```
    """
    # Verify required columns
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate price changes
    result['price_change'] = result[price_col].diff()
    
    # Calculate absolute price changes
    result['abs_price_change'] = result['price_change'].abs()
    
    # Calculate rolling volatility
    result['rolling_volatility'] = result['price_change'].rolling(window=window).std()
    
    # Calculate signed realized volatility
    result['signed_price_change'] = np.sign(result['price_change']) * result['abs_price_change']
    
    # Add volume-weighted features if volume is available
    if volume_col in result.columns:
        # Volume-weighted price
        result['volume_weighted_price'] = (result[price_col] * result[volume_col]).rolling(window=window).sum() / result[volume_col].rolling(window=window).sum()
        
        # Volume volatility
        result['volume_volatility'] = result[volume_col].rolling(window=window).std() / result[volume_col].rolling(window=window).mean()
        
        # Price-volume correlation
        def rolling_corr(x):
            if len(x) < 2:
                return np.nan
            price_data = x[price_col]
            volume_data = x[volume_col]
            return np.corrcoef(price_data, volume_data)[0, 1] if len(price_data) > 1 else np.nan
        
        result['price_volume_corr'] = result.rolling(window=window).apply(rolling_corr, raw=False)
    
    # Add price acceleration (change in price changes)
    result['price_acceleration'] = result['price_change'].diff()
    
    # Log generated indicators
    added_indicators = [col for col in result.columns if col not in df.columns]
    logger.info(f"Added {len(added_indicators)} microstructure features")
    
    return result


#------------------------------------------------------------------------
# Feature Selection and Engineering
#------------------------------------------------------------------------

def create_feature_set(
    df: pd.DataFrame,
    feature_types: Optional[List[str]] = None,
    price_col: str = 'close',
    returns_col: Optional[str] = None,
    windows: Optional[List[int]] = None,
    add_lagged: bool = True,
    lag_periods: Optional[List[int]] = None,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create a comprehensive feature set based on specified types.
    
    This function orchestrates the creation of a complete feature set
    by combining multiple feature categories and enhancements.
    
    Args:
        df: Input DataFrame
        feature_types: Types of features to include (default ['technical', 'volatility', 'zscore'])
        price_col: Column name for price data
        returns_col: Column name for return data
        windows: Window sizes for calculations
        add_lagged: Whether to add lagged features
        lag_periods: Periods for lagged features
        drop_na: Whether to drop rows with NaN values
        
    Returns:
        DataFrame with comprehensive feature set
    
    Available feature types:
        - 'technical': Technical indicators
        - 'volatility': Volatility measures
        - 'trend': Trend-following indicators
        - 'zscore': Normalization and extreme value indicators
        - 'microstructure': Market microstructure metrics
        - 'all': All available feature types
    
    Examples:
        ```python
        # Create a comprehensive feature set
        features_df = create_feature_set(df, feature_types=['all'])
        
        # Create focused momentum feature set
        momentum_df = create_feature_set(
            df, 
            feature_types=['technical', 'trend'],
            windows=[10, 20, 50]
        )
        ```
    """
    # Default values if not provided
    if feature_types is None:
        feature_types = ['technical', 'volatility', 'zscore']
    
    if windows is None:
        windows = [20, 60, 120]
    
    if lag_periods is None:
        lag_periods = [1, 5, 10]
    
    # Expand 'all' to include all feature types
    if 'all' in feature_types:
        feature_types = ['technical', 'volatility', 'trend', 'zscore', 'microstructure']
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate returns if needed and not present
    if (('technical' in feature_types or 'volatility' in feature_types) and 
        returns_col is None and 'returns' not in result.columns):
        result['returns'] = calculate_returns(result[price_col], method='log')
        returns_col = 'returns'
    
    # Track created features
    created_features = []
    
    # Add features based on specified types
    if 'technical' in feature_types:
        # Add basic technical indicators
        tech_result = add_technical_features(
            result, 
            price_col=price_col,
            window=windows[0]
        )
        
        # Identify new columns
        new_cols = [col for col in tech_result.columns if col not in result.columns]
        created_features.extend(new_cols)
        
        # Add to result
        result = tech_result
    
    if 'volatility' in feature_types:
        # Add volatility features
        vol_result = add_volatility_features(
            result,
            returns_col=returns_col if returns_col else 'returns',
            price_col=price_col,
            windows=windows
        )
        
        # Identify new columns
        new_cols = [col for col in vol_result.columns if col not in result.columns]
        created_features.extend(new_cols)
        
        # Add to result
        result = vol_result
    
    if 'trend' in feature_types:
        # Add trend features
        trend_result = add_trend_features(
            result,
            price_col=price_col,
            windows=windows
        )
        
        # Identify new columns
        new_cols = [col for col in trend_result.columns if col not in result.columns]
        created_features.extend(new_cols)
        
        # Add to result
        result = trend_result
    
    if 'zscore' in feature_types:
        # Identify columns for z-score calculation
        zscore_cols = []
        
        # Use technical indicators and price for z-scores
        if 'technical' in feature_types:
            zscore_cols.extend([col for col in result.columns if any(
                indicator in col for indicator in ['rsi', 'macd', 'stoch', 'bb_width']
            )])
        
        # Always include price
        zscore_cols.append(price_col)
        
        # Add z-score features if we have columns to process
        if zscore_cols:
            zscore_result = add_zscore_features(
                result,
                columns=zscore_cols,
                windows=windows,
                methods=['rolling', 'ewm'] if len(windows) <= 2 else ['rolling']
            )
            
            # Identify new columns
            new_cols = [col for col in zscore_result.columns if col not in result.columns]
            created_features.extend(new_cols)
            
            # Add to result
            result = zscore_result
    
    if 'microstructure' in feature_types:
        # Add microstructure features
        micro_result = add_microstructure_features(
            result,
            price_col=price_col,
            volume_col='volume' if 'volume' in result.columns else None,
            window=windows[0]
        )
        
        # Identify new columns
        new_cols = [col for col in micro_result.columns if col not in result.columns]
        created_features.extend(new_cols)
        
        # Add to result
        result = micro_result
    
    # Add lagged features if requested
    if add_lagged and created_features:
        lagged_result = add_lagged_features(
            result,
            columns=created_features,
            lags=lag_periods,
            drop_na=False
        )
        
        # Add to result
        result = lagged_result
    
    # Drop NA values if requested
    if drop_na:
        result = result.dropna()
    
    logger.info(f"Created comprehensive feature set with {len(created_features)} base features")
    
    return result


def create_momentum_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create a feature set optimized for momentum strategies.
    
    This creates a focused set of features particularly useful for
    momentum and trend-following strategies.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        windows: Window sizes for calculations
        
    Returns:
        DataFrame with momentum-focused features
    
    Examples:
        ```python
        # Create momentum features with default settings
        momentum_df = create_momentum_features(df)
        
        # Custom momentum features for different timeframes
        momentum_df = create_momentum_features(
            df, 
            windows=[5, 10, 20, 60, 120]  # Multiple timeframes
        )
        ```
    """
    # Default windows if not provided
    if windows is None:
        windows = [20, 60, 120, 250]
    
    # Create focused feature set
    return create_feature_set(
        df,
        feature_types=['technical', 'trend'],
        price_col=price_col,
        windows=windows,
        add_lagged=True,
        lag_periods=[1, 5]
    )


def create_mean_reversion_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create a feature set optimized for mean-reversion strategies.
    
    This creates a focused set of features particularly useful for
    mean-reversion and contrarian strategies.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        windows: Window sizes for calculations
        
    Returns:
        DataFrame with mean-reversion-focused features
    
    Examples:
        ```python
        # Create mean-reversion features with default settings
        reversion_df = create_mean_reversion_features(df)
        ```
    """
    # Default windows if not provided
    if windows is None:
        windows = [5, 10, 20, 60]
    
    # Create focused feature set
    return create_feature_set(
        df,
        feature_types=['technical', 'zscore', 'volatility'],
        price_col=price_col,
        windows=windows,
        add_lagged=True,
        lag_periods=[1, 2, 3]
    )


def create_ml_feature_set(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume',
    returns_col: Optional[str] = None,
    feature_types: Optional[List[str]] = None,
    lookback_windows: Optional[List[int]] = None,
    normalize_features: bool = True,
    drop_na: bool = True,
    remove_correlated: bool = False,
    correlation_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Create a comprehensive feature set optimized for machine learning models.
    
    This function creates a robust feature set for ML, handling the special
    requirements of ML models including normalization and correlation removal.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        volume_col: Column name for volume data
        returns_col: Column name for return data
        feature_types: Types of features to include
        lookback_windows: Window sizes for calculations
        normalize_features: Whether to normalize features
        drop_na: Whether to drop rows with NaN values
        remove_correlated: Whether to remove highly correlated features
        correlation_threshold: Threshold for correlation removal
        
    Returns:
        DataFrame with ML-ready features
    
    Examples:
        ```python
        # Create ML-ready feature set
        ml_df = create_ml_feature_set(
            df,
            normalize_features=True,
            remove_correlated=True
        )
        ```
    """
    # Default values if not provided
    if feature_types is None:
        feature_types = ['technical', 'volatility', 'trend', 'zscore']
    
    if lookback_windows is None:
        lookback_windows = [5, 10, 20, 60]
    
    # Create comprehensive feature set
    features_df = create_feature_set(
        df,
        feature_types=feature_types,
        price_col=price_col,
        returns_col=returns_col,
        windows=lookback_windows,
        add_lagged=True,
        drop_na=drop_na
    )
    
    # Remove highly correlated features if requested
    if remove_correlated:
        # Calculate correlation matrix for numeric columns
        numeric_cols = features_df.select_dtypes(include=np.number).columns
        corr_matrix = features_df[numeric_cols].corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        
        # Remove correlated features
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            features_df = features_df.drop(columns=to_drop)
    
    # Normalize features if requested
    if normalize_features:
        # Only normalize numeric columns
        numeric_cols = features_df.select_dtypes(include=np.number).columns
        
        # Exclude price and volume columns (often useful to keep unnormalized)
        cols_to_normalize = [col for col in numeric_cols 
                            if col != price_col and col != volume_col]
        
        # Normalize
        features_df = normalize_data(
            features_df, 
            columns=cols_to_normalize,
            method='standard'
        )
    
    return features_df