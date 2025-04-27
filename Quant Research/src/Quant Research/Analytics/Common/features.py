"""
Feature Engineering Module

This module provides standardized functions for creating features from financial time series
data. It centralizes feature engineering logic used across different analytics modules.

Features are organized into categories:
- Basic time series features (returns, momentum, volatility)
- Technical indicators (moving averages, MACD, RSI, etc.)
- Statistical features (z-scores, normalization)
- Volatility features (various volatility estimators)
- Regime features (regime detection preprocessing)
- Cross-sectional features (relative performance measures)
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logger
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------
# Basic Time Series Features
#------------------------------------------------------------------------

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


def add_trend_features(
    df: pd.DataFrame,
    price_col: str = 'close',
    windows: List[int] = [20, 50, 200],
    include_crosses: bool = True
) -> pd.DataFrame:
    """
    Add trend-following features and moving average crosses.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        windows: List of moving average window sizes
        include_crosses: Whether to include moving average crossover indicators
        
    Returns:
        DataFrame with added trend features
    """
    result = df.copy()
    
    # Calculate moving averages for each window
    for window in windows:
        result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()
        
        # Add price to moving average ratio
        result[f'price_to_ma_{window}'] = result[price_col] / result[f'ma_{window}']
    
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
    
    return result


#------------------------------------------------------------------------
# Statistical Features
#------------------------------------------------------------------------

def calculate_zscore(
    series: pd.Series, 
    window: int = 60,
    method: str = 'rolling',
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate z-score for a time series.
    
    Args:
        series: Time series to calculate z-score for
        window: Rolling window size
        method: Method for calculation ('rolling', 'ewm', 'expanding')
        min_periods: Minimum number of observations required
        
    Returns:
        Series of z-scores
        
    Raises:
        ValueError: If an invalid method is specified
    """
    if min_periods is None:
        min_periods = window // 2
        
    # Calculate means and standard deviations
    if method == 'rolling':
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        
        # Replace zero standard deviations with NaN to avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        z_score = (series - rolling_mean) / rolling_std
    
    elif method == 'ewm':
        ewma_mean = series.ewm(span=window, min_periods=min_periods).mean()
        ewma_std = series.ewm(span=window, min_periods=min_periods).std()
        
        # Replace zero standard deviations with NaN to avoid division by zero
        ewma_std = ewma_std.replace(0, np.nan)
        
        z_score = (series - ewma_mean) / ewma_std
    
    elif method == 'expanding':
        expanding_mean = series.expanding(min_periods=min_periods).mean()
        expanding_std = series.expanding(min_periods=min_periods).std()
        
        # Replace zero standard deviations with NaN to avoid division by zero
        expanding_std = expanding_std.replace(0, np.nan)
        
        z_score = (series - expanding_mean) / expanding_std
    
    elif method == 'regime_adjusted':
        # First calculate standard rolling z-score
        z_score = calculate_zscore(series, window, 'rolling', min_periods)
        
        # Calculate volatility of volatility (meta-volatility)
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        vol_window = window * 3  # Longer window for regime detection
        vol_of_vol = rolling_std.rolling(window=vol_window).std() / rolling_std.rolling(window=vol_window).mean()
        
        # Calculate scaling factor (higher volatility periods get downscaled)
        scaling = 1.0 / np.maximum(1.0, vol_of_vol / vol_of_vol.rolling(window=vol_window).median())
        
        # Apply scaling to original z-score
        z_score = z_score * scaling
        
    else:
        raise ValueError(f"Invalid z-score method: {method}. Use 'rolling', 'ewm', 'expanding', or 'regime_adjusted'")
    
    return z_score


def add_zscore_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [20, 60, 120],
    methods: List[str] = ['rolling'],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Add z-score features for multiple columns and windows.
    
    Args:
        df: Input DataFrame
        columns: Columns to calculate z-scores for
        windows: List of window sizes
        methods: List of calculation methods
        prefix: Prefix for column names
        
    Returns:
        DataFrame with added z-score features
    """
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
    
    return result


def add_percentile_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [20, 60, 120],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Add percentile rank features.
    
    Args:
        df: Input DataFrame
        columns: Columns to calculate percentiles for
        windows: List of window sizes
        prefix: Prefix for column names
        
    Returns:
        DataFrame with added percentile features
    """
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
    
    return result


#------------------------------------------------------------------------
# Volatility Features
#------------------------------------------------------------------------

def add_volatility_features(
    df: pd.DataFrame,
    returns_col: str = 'returns',
    windows: List[int] = [5, 20, 60],
    use_log_scale: bool = False,
    annualize: bool = True,
    trading_days: int = 252,
    estimators: List[str] = ['standard']
) -> pd.DataFrame:
    """
    Add various volatility features.
    
    Args:
        df: Input DataFrame with return data
        returns_col: Column name for returns
        windows: List of window sizes
        use_log_scale: Whether to apply log transformation to volatility
        annualize: Whether to annualize volatility
        trading_days: Number of trading days per year
        estimators: List of volatility estimators to use
        
    Returns:
        DataFrame with added volatility features
    """
    result = df.copy()
    
    # Check if returns column exists
    if returns_col not in result.columns:
        if 'close' in result.columns:
            # Calculate returns if not present
            result[returns_col] = result['close'].pct_change()
        else:
            raise ValueError(f"Returns column '{returns_col}' not found in DataFrame")
    
    # Add volatility for each window
    for window in windows:
        for estimator in estimators:
            if estimator == 'standard':
                # Standard rolling volatility
                vol = result[returns_col].rolling(window=window).std()
                
                # Annualize if requested
                if annualize:
                    vol = vol * np.sqrt(trading_days)
                
                # Apply log transformation if requested
                if use_log_scale:
                    vol = np.log(vol)
                
                # Add to result
                result[f'volatility_{window}'] = vol
                
                # Volatility of volatility (meta-volatility)
                vol_of_vol = vol.rolling(window=window).std()
                result[f'volatility_of_volatility_{window}'] = vol_of_vol
                
                # Volatility trend (is volatility increasing or decreasing)
                vol_trend = vol / vol.shift(window // 4) - 1
                result[f'volatility_trend_{window}'] = vol_trend
                
            # Additional estimators can be added here
            elif estimator == 'parkinson':
                # Need high/low data for Parkinson estimator
                if all(col in result.columns for col in ['high', 'low']):
                    log_hl = np.log(result['high'] / result['low'])
                    squared_log_range = log_hl**2
                    
                    # Parkinson estimator with scaling factor 1/(4*ln(2))
                    scaling_factor = 1.0 / (4.0 * np.log(2.0))
                    vol = np.sqrt(
                        (squared_log_range * scaling_factor).rolling(window=window).mean()
                    )
                    
                    # Annualize if requested
                    if annualize:
                        vol = vol * np.sqrt(trading_days)
                    
                    result[f'volatility_parkinson_{window}'] = vol
                else:
                    logger.warning("High/Low data required for Parkinson estimator, skipping")
                    
            elif estimator == 'garman_klass':
                # Need OHLC data for Garman-Klass estimator
                if all(col in result.columns for col in ['open', 'high', 'low', 'close']):
                    log_hl = np.log(result['high'] / result['low'])
                    log_co = np.log(result['close'] / result['open'])
                    
                    # Garman-Klass components
                    hl_part = log_hl**2
                    co_part = log_co**2
                    
                    # Full estimator with coefficient
                    estimator = 0.5 * hl_part - (2 * np.log(2) - 1) * co_part
                    
                    vol = np.sqrt(estimator.rolling(window=window).mean())
                    
                    # Annualize if requested
                    if annualize:
                        vol = vol * np.sqrt(trading_days)
                    
                    result[f'volatility_garman_klass_{window}'] = vol
                else:
                    logger.warning("OHLC data required for Garman-Klass estimator, skipping")
    
    return result


#------------------------------------------------------------------------
# Regime Features
#------------------------------------------------------------------------

def prepare_features_for_regime_detection(
    df: pd.DataFrame,
    features: List[str] = ["returns", "volatility"],
    window: int = 20,
    add_derived: bool = True,
    standardize: bool = True
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Prepare and normalize features for regime detection algorithms.
    
    Args:
        df: Input dataframe with price/return data
        features: List of column names to use as features
        window: Window size for calculating derived features
        add_derived: Whether to add derived features if not already present
        standardize: Whether to standardize features
        
    Returns:
        Tuple of processed feature dataframe and optional fitted scaler
        
    Raises:
        ValueError: If no valid features are found
    """
    feature_df = df.copy()
    scaler = None
    
    # Add derived features if not present and requested
    if add_derived:
        if "returns" not in feature_df.columns and "close" in feature_df.columns:
            feature_df["returns"] = calculate_returns(feature_df["close"], method='pct')
            
        if "volatility" not in feature_df.columns and "returns" in feature_df.columns:
            feature_df["volatility"] = calculate_volatility(feature_df["returns"], window=window)
            
        if "volume_change" not in feature_df.columns and "volume" in feature_df.columns:
            feature_df["volume_change"] = feature_df["volume"].pct_change()
    
    # Select only requested features
    feature_subset = [f for f in features if f in feature_df.columns]
    
    if not feature_subset:
        raise ValueError(f"None of the requested features {features} found in dataframe")
    
    # Create feature matrix
    X = feature_df[feature_subset].copy()
    
    # Drop NaN values (from rolling calculations)
    X = X.dropna()
    
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        return X_scaled, scaler
    else:
        return X, None


def add_change_point_features(
    df: pd.DataFrame,
    columns: List[str],
    window: int = 50,
    step: int = 10,
    method: str = 'sliding_window',
    minimum_size: int = 10
) -> pd.DataFrame:
    """
    Add change point detection features.
    
    Args:
        df: Input DataFrame
        columns: Columns to analyze for change points
        window: Window size for analysis
        step: Step size for sliding window
        method: Detection method
        minimum_size: Minimum segment size
        
    Returns:
        DataFrame with added change point features
    """
    # This is a placeholder for more complex implementation
    # In a real implementation, we would use change point detection algorithms
    # like ruptures package or custom implementations
    
    result = df.copy()
    
    # Add placeholder for actual implementation
    result['change_score'] = 0.0
    
    for col in columns:
        if col not in result.columns:
            continue
            
        # For now, just use a simple change detector based on volatility changes
        # A more sophisticated implementation would use proper change point detection
        rolling_std = result[col].rolling(window=window//2).std()
        prev_rolling_std = rolling_std.shift(window//2)
        
        # Calculate ratio of current to previous volatility
        volatility_ratio = rolling_std / prev_rolling_std
        
        # Mark points with significant volatility changes
        result[f'{col}_change_score'] = volatility_ratio.abs()
    
    return result


#------------------------------------------------------------------------
# Cross-Sectional Features
#------------------------------------------------------------------------

def add_cross_sectional_features(
    df: pd.DataFrame,
    columns: List[str],
    window: int = 20,
    ranking_methods: List[str] = ['percentile']
) -> pd.DataFrame:
    """
    Add cross-sectional ranking features.
    
    Args:
        df: Input DataFrame with multiple assets as columns
        columns: Columns to rank
        window: Window for rolling calculations
        ranking_methods: Methods for ranking
        
    Returns:
        DataFrame with added cross-sectional features
    """
    result = df.copy()
    
    # This is a simplified implementation
    # For a real implementation, we would need a multi-asset DataFrame
    # with proper alignment
    
    if 'percentile' in ranking_methods:
        for col in columns:
            if col not in result.columns:
                continue
                
            # Calculate rolling mean for normalization
            rolling_mean = result[col].rolling(window=window).mean()
            
            # Calculate rolling standard deviation for normalization
            rolling_std = result[col].rolling(window=window).std()
            
            # Z-score normalization
            result[f'{col}_cross_sectional_zscore'] = (result[col] - rolling_mean) / rolling_std
    
    return result


#------------------------------------------------------------------------
# Feature Selection and Engineering
#------------------------------------------------------------------------

def create_feature_set(
    df: pd.DataFrame,
    feature_types: List[str] = ['technical', 'volatility', 'zscore'],
    price_col: str = 'close',
    returns_col: str = 'returns',
    windows: List[int] = [20, 60],
    add_lagged: bool = True,
    lag_periods: List[int] = [1, 5],
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create a comprehensive feature set based on specified types.
    
    Args:
        df: Input DataFrame
        feature_types: Types of features to include
        price_col: Column name for price data
        returns_col: Column name for return data
        windows: Window sizes for calculations
        add_lagged: Whether to add lagged features
        lag_periods: Periods for lagged features
        drop_na: Whether to drop rows with NaN values
        
    Returns:
        DataFrame with comprehensive feature set
    """
    result = df.copy()
    
    # Calculate returns if needed and not present
    if 'returns' not in result.columns and 'close' in result.columns:
        result['returns'] = calculate_returns(result['close'], method='pct')
    
    # Track created features
    created_features = []
    
    # Add features based on specified types
    if 'technical' in feature_types:
        # Add basic technical indicators
        tech_indicators = add_technical_features(
            result, 
            price_col=price_col,
            window=windows[0]
        )
        result = pd.concat([result, tech_indicators], axis=1)
        
        # Track created feature columns
        created_features.extend([col for col in tech_indicators.columns if col not in result.columns])
    
    if 'volatility' in feature_types:
        # Add volatility features for each window
        vol_features = add_volatility_features(
            result,
            returns_col=returns_col,
            windows=windows
        )
        result = pd.concat([result, vol_features], axis=1)
        
        # Track created feature columns
        created_features.extend([col for col in vol_features.columns if col not in result.columns])
    
    if 'zscore' in feature_types:
        # Add z-score features
        zscore_cols = []
        if price_col in result.columns:
            zscore_cols.append(price_col)
        if returns_col in result.columns:
            zscore_cols.append(returns_col)
        
        if zscore_cols:
            zscore_features = add_zscore_features(
                result,
                columns=zscore_cols,
                windows=windows
            )
            result = pd.concat([result, zscore_features], axis=1)
            
            # Track created feature columns
            created_features.extend([col for col in zscore_features.columns if col not in result.columns])
    
    # Add lagged features if requested
    if add_lagged and created_features:
        for lag in lag_periods:
            for feature in created_features:
                result[f"{feature}_lag_{lag}"] = result[feature].shift(lag)
    
    # Drop NA values if requested
    if drop_na:
        result = result.dropna()
    
    return result