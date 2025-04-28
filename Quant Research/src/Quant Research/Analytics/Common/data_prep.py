"""
Data Preparation Workflows

This module provides high-level data preparation workflows that build upon the
core utilities in data_utils.py. It focuses on complete end-to-end preparation 
pipelines for different types of financial data analysis.

Key workflows include:
- Preparing OHLCV data for technical analysis
- Preparing return data for statistical analysis
- Preparing data for machine learning models
- Common data preparation patterns for backtesting

Each workflow combines multiple lower-level operations from data_utils.py
into coherent, reusable preparation sequences.

Usage:
    ```python
    from quant_research.analytics.common.data_prep import (
        prepare_ohlcv_data,
        prepare_returns_data,
        prepare_for_machine_learning,
        prepare_multi_asset_data
    )
    
    # Prepare OHLCV data for analysis
    clean_data = prepare_ohlcv_data(
        df, 
        resample_freq='1D',
        handle_outliers=True
    )
    
    # Prepare data for ML model with automated feature engineering
    X_train, y_train = prepare_for_machine_learning(
        df,
        target_col='returns',
        lookahead=5,
        add_features=True
    )
    ```
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports - using refactored data_utils
from quant_research.analytics.common.data_utils import (
    # Core DataFrame operations
    ensure_datetime_index,
    validate_ohlc,
    detect_frequency,
    validate_missing_data,
    
    # Time series processing
    filter_time_range,
    resample_data,
    align_dataframes,
    
    # Financial calculations
    calculate_returns,
    calculate_cumulative_returns,
    calculate_drawdowns,
    calculate_volatility,
    
    # Data cleaning
    handle_outliers,
    normalize_data,
    
    # Feature engineering
    add_lagged_features,
    add_difference_features,
    add_rolling_features,
    
    # Cross-validation
    time_series_split,
    expanding_window_split,
    walk_forward_split
)

# Configure module logger
logger = logging.getLogger("quant_research.analytics.common.data_prep")

#------------------------------------------------------------------------
# OHLCV Data Preparation
#------------------------------------------------------------------------

def prepare_ohlcv_data(
    df: pd.DataFrame,
    resample_freq: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    handle_outliers: bool = False,
    fill_missing: bool = True,
    add_returns: bool = True,
    return_method: str = 'log',
    handle_missing_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Prepare OHLCV data for analysis with optional resampling and cleaning.
    
    This function performs a complete preparation workflow for financial price data:
    1. Validates and standardizes OHLCV columns
    2. Ensures proper datetime index
    3. Filters to desired date range
    4. Handles missing values
    5. Resamples to desired frequency (if requested)
    6. Handles outliers (if requested)
    7. Adds return calculations (if requested)
    
    Args:
        df: Input DataFrame with price data
        resample_freq: Target frequency for resampling (e.g., '1D', '1H', '15min')
        start_date: Start date for filtering
        end_date: End date for filtering
        handle_outliers: Whether to detect and handle outliers
        fill_missing: Whether to fill missing values
        add_returns: Whether to add return calculations
        return_method: Method for return calculation ('log', 'pct')
        handle_missing_threshold: Threshold for handling missing data
    
    Returns:
        Prepared DataFrame ready for analysis
    
    Examples:
        ```python
        # Basic preparation with defaults
        clean_df = prepare_ohlcv_data(raw_df)
        
        # Prepare daily data from intraday with outlier handling
        daily_df = prepare_ohlcv_data(
            raw_df, 
            resample_freq='1D',
            handle_outliers=True,
            start_date='2020-01-01'
        )
        ```
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # 1. Validate and standardize OHLCV columns
    try:
        result = validate_ohlc(result)
        logger.info("OHLCV validation completed successfully")
    except ValueError as e:
        logger.warning(f"OHLCV validation warning: {e}")
    
    # 2. Ensure datetime index
    result = ensure_datetime_index(result)
    
    # 3. Filter to desired date range
    if start_date is not None or end_date is not None:
        result = filter_time_range(result, start_date, end_date)
        logger.info(f"Filtered data to {len(result)} rows")
    
    # 4. Handle missing values
    if fill_missing:
        result, missing_props = validate_missing_data(
            result,
            threshold=handle_missing_threshold,
            fill_method='ffill'  # Forward fill as default for OHLCV
        )
        missing_cols = [col for col, prop in missing_props.items() if prop > 0]
        if missing_cols:
            logger.info(f"Filled missing values in columns: {', '.join(missing_cols)}")
    
    # 5. Resample if requested
    if resample_freq is not None:
        result = resample_data(result, freq=resample_freq)
        logger.info(f"Resampled data to frequency: {resample_freq}")
    
    # 6. Handle outliers if requested
    if handle_outliers:
        # For OHLCV data, we're careful about outlier handling to preserve price spikes
        # that might be real market events, so we use conservative settings
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
        
        # Handle outliers in price columns with winsorization to preserve structure but limit extremes
        if price_cols:
            result = handle_outliers(
                result,
                columns=price_cols,
                method='iqr',  # IQR is less sensitive than z-score
                threshold=3.0,  # Conservative threshold
                treatment='winsorize'  # Don't remove, just cap extremes
            )
        
        # For volume, we can be more aggressive as extreme volume spikes might be data errors
        if 'volume' in result.columns:
            result = handle_outliers(
                result,
                columns=['volume'],
                method='iqr',
                threshold=5.0,  # More permissive for volume
                treatment='winsorize'
            )
    
    # 7. Add returns if requested
    if add_returns and 'close' in result.columns:
        result['returns'] = calculate_returns(
            result['close'],
            method=return_method
        )
        logger.info(f"Added {return_method} returns")
    
    return result


def prepare_returns_data(
    df: pd.DataFrame,
    price_col: str = 'close',
    return_method: str = 'log',
    periods: List[int] = [1, 5, 20],
    add_volatility: bool = True,
    add_cumulative: bool = True,
    include_drawdowns: bool = False,
    annualize: bool = False,
    trading_days: int = 252,
    handle_outliers: bool = True
) -> pd.DataFrame:
    """
    Prepare return data for statistical analysis.
    
    This function creates a comprehensive return dataset from price data,
    including different horizons, volatility metrics, and return statistics.
    
    Args:
        df: Input DataFrame with price data
        price_col: Column name for price data
        return_method: Method for return calculation ('log', 'pct')
        periods: List of periods for return calculations
        add_volatility: Whether to add volatility metrics
        add_cumulative: Whether to add cumulative returns
        include_drawdowns: Whether to include drawdown calculations
        annualize: Whether to annualize volatility
        trading_days: Number of trading days per year for annualization
        handle_outliers: Whether to handle outliers in returns
    
    Returns:
        DataFrame with calculated returns and metrics
    
    Examples:
        ```python
        # Basic return preparation
        returns_df = prepare_returns_data(price_df)
        
        # Comprehensive return analysis
        returns_df = prepare_returns_data(
            price_df,
            periods=[1, 5, 10, 21, 63],  # Daily, weekly, biweekly, monthly, quarterly
            add_volatility=True,
            add_cumulative=True,
            include_drawdowns=True
        )
        ```
    """
    # Check if price column exists
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Ensure datetime index
    result = ensure_datetime_index(result)
    
    # Calculate returns for different periods
    for period in periods:
        period_suffix = "" if period == 1 else f"_{period}"
        col_name = f"return{period_suffix}"
        
        result[col_name] = calculate_returns(
            result[price_col],
            method=return_method,
            periods=period
        )
    
    # Handle return outliers if requested
    if handle_outliers:
        return_cols = [col for col in result.columns if col.startswith('return')]
        result = handle_outliers(
            result,
            columns=return_cols,
            method='zscore',
            threshold=4.0,  # Conservative threshold for returns
            treatment='winsorize'
        )
    
    # Add volatility if requested
    if add_volatility:
        # Get base return column (period=1)
        base_return_col = "return" if 1 in periods else f"return_{periods[0]}"
        
        # Calculate with different windows
        vol_windows = [min(window, len(result)//2) for window in [20, 60, 120]]
        
        for window in vol_windows:
            result[f'volatility_{window}d'] = calculate_volatility(
                result[base_return_col],
                window=window,
                annualize=annualize,
                trading_days=trading_days
            )
    
    # Add cumulative returns if requested
    if add_cumulative:
        # Get base return column (period=1)
        base_return_col = "return" if 1 in periods else f"return_{periods[0]}"
        
        # Default to log_returns if method is 'log'
        is_log_returns = return_method == 'log'
        
        result['cumulative_return'] = calculate_cumulative_returns(
            result[base_return_col],
            log_returns=is_log_returns
        )
    
    # Add drawdowns if requested
    if include_drawdowns and 'cumulative_return' in result.columns:
        result['drawdown'] = calculate_drawdowns(
            result[base_return_col],
            log_returns=is_log_returns
        )
    
    return result


#------------------------------------------------------------------------
# Machine Learning Data Preparation
#------------------------------------------------------------------------

def prepare_for_machine_learning(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    lookahead: int = 1,
    add_features: bool = True,
    normalize: bool = True,
    test_size: float = 0.2,
    valid_size: float = 0.1,
    cv_method: str = 'expanding',
    n_splits: int = 5,
    scaling_method: str = 'standard',
    classification_threshold: Optional[float] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare data for machine learning models with feature engineering and proper CV splits.
    
    This function handles the complete workflow for preparing time series data for ML:
    1. Feature selection and engineering
    2. Target creation (with optional classification transformation)
    3. Proper time series train/valid/test splitting
    4. Feature scaling
    5. Returns data ready for model training
    
    Args:
        df: Input DataFrame with time series data
        target_col: Column to use as prediction target
        feature_cols: Columns to use as features (None for auto-selection)
        lookahead: Target lookahead period for prediction
        add_features: Whether to add engineered features
        normalize: Whether to normalize features
        test_size: Proportion of data for testing
        valid_size: Proportion of data for validation
        cv_method: Cross-validation method ('simple', 'expanding', 'walk_forward')
        n_splits: Number of CV splits
        scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
        classification_threshold: If provided, convert target to binary classification
    
    Returns:
        Dictionary with train/valid/test data splits and scaler
    
    Examples:
        ```python
        # Prepare data for regression task
        ml_data = prepare_for_machine_learning(
            df, 
            target_col='returns',
            lookahead=5,
            add_features=True
        )
        
        # Access the prepared data
        X_train, y_train = ml_data['train']
        X_valid, y_valid = ml_data['valid']
        X_test, y_test = ml_data['test']
        scaler = ml_data['scaler']
        ```
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Ensure datetime index
    result = ensure_datetime_index(result)
    
    # Verify target column exists
    if target_col not in result.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Select feature columns if not specified
    if feature_cols is None:
        # Auto-select numeric columns, excluding the target
        feature_cols = [
            col for col in result.select_dtypes(include=np.number).columns
            if col != target_col
        ]
        logger.info(f"Auto-selected {len(feature_cols)} feature columns")
    
    # Add engineered features if requested
    if add_features:
        # Add lagged features
        result = add_lagged_features(
            result,
            columns=feature_cols,
            lags=[1, 2, 3, 5]
        )
        
        # Add rolling statistical features
        result = add_rolling_features(
            result,
            columns=feature_cols,
            windows=[5, 10, 20]
        )
        
        # Add difference features
        result = add_difference_features(
            result,
            columns=feature_cols,
            periods=[1, 5],
            pct_change=True
        )
        
        # Update feature_cols to include new features
        feature_cols = [
            col for col in result.columns 
            if col != target_col and col in result.select_dtypes(include=np.number).columns
        ]
        logger.info(f"Added engineered features. Total features: {len(feature_cols)}")
    
    # Create future target for prediction (with lookahead)
    if lookahead > 0:
        # Shift the target backwards to align future values with current features
        result[f'target_{lookahead}'] = result[target_col].shift(-lookahead)
        prediction_target = f'target_{lookahead}'
    else:
        prediction_target = target_col
    
    # Convert to classification problem if threshold is provided
    if classification_threshold is not None:
        result[f'{prediction_target}_class'] = (
            result[prediction_target] > classification_threshold
        ).astype(int)
        prediction_target = f'{prediction_target}_class'
        logger.info(f"Converted to classification problem with threshold {classification_threshold}")
    
    # Drop rows with NaN in target
    result = result.dropna(subset=[prediction_target])
    
    # Create time series CV splits
    if cv_method == 'expanding':
        splits = expanding_window_split(
            result,
            n_splits=n_splits,
            test_size=test_size,
            valid_size=valid_size
        )
    elif cv_method == 'walk_forward':
        # Calculate sizes in terms of number of rows
        train_size = int(len(result) * (1 - test_size - valid_size))
        test_size_rows = int(len(result) * test_size)
        valid_size_rows = int(len(result) * valid_size)
        
        splits = walk_forward_split(
            result,
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size_rows,
            valid_size=valid_size_rows
        )
    else:  # Simple split
        # Use the last split from expanding window for a simple split
        splits = expanding_window_split(
            result,
            n_splits=1,
            test_size=test_size,
            valid_size=valid_size
        )
    
    # Use the first split for initial train/valid/test
    # (all splits are available for cross-validation)
    train_df, valid_df, test_df = splits[0]
    
    # Normalize features if requested
    scaler = None
    if normalize:
        # Fit scaler on training data only
        train_df, scaler = normalize_data(
            train_df,
            columns=feature_cols,
            method=scaling_method,
            output_scaler=True
        )
        
        # Apply same scaling to validation and test
        if scaler is not None:
            # For MinMaxScaler and StandardScaler from sklearn
            valid_features = valid_df[feature_cols].values
            valid_df[feature_cols] = scaler.transform(valid_features)
            
            test_features = test_df[feature_cols].values
            test_df[feature_cols] = scaler.transform(test_features)
    
    # Create X, y pairs
    X_train = train_df[feature_cols]
    y_train = train_df[prediction_target]
    
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[prediction_target]
    
    X_test = test_df[feature_cols]
    y_test = test_df[prediction_target]
    
    # Create result dictionary
    ml_data = {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test': (X_test, y_test),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target': prediction_target,
        'splits': splits  # All CV splits for cross-validation
    }
    
    return ml_data


#------------------------------------------------------------------------
# Multi-Asset Data Preparation
#------------------------------------------------------------------------

def prepare_multi_asset_data(
    dfs: Dict[str, pd.DataFrame],
    resample_freq: Optional[str] = None,
    align_method: str = 'outer',
    fill_method: str = 'ffill',
    calculate_returns_kwargs: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Prepare and align data from multiple assets into a single DataFrame.
    
    This function processes multiple asset DataFrames and combines them 
    for cross-sectional analysis.
    
    Args:
        dfs: Dictionary mapping asset names to asset DataFrames
        resample_freq: Frequency to resample all assets to
        align_method: Method for aligning indexes ('outer', 'inner')
        fill_method: Method for filling missing values after alignment
        calculate_returns_kwargs: Arguments for return calculation (if None, returns not calculated)
    
    Returns:
        Combined DataFrame with multi-level columns (asset, field)
    
    Examples:
        ```python
        # Prepare data for multiple assets
        combined_df = prepare_multi_asset_data(
            {'AAPL': aapl_df, 'MSFT': msft_df, 'GOOG': goog_df},
            resample_freq='1D',
            calculate_returns_kwargs={'method': 'log'}
        )
        ```
    """
    # Validate input
    if not dfs:
        raise ValueError("No asset DataFrames provided")
    
    # Process each asset
    processed_dfs = {}
    
    for asset_name, asset_df in dfs.items():
        # Make a copy
        df = asset_df.copy()
        
        # Ensure datetime index
        df = ensure_datetime_index(df)
        
        # Resample if requested
        if resample_freq is not None:
            df = resample_data(df, freq=resample_freq)
        
        # Calculate returns if requested
        if calculate_returns_kwargs is not None and 'close' in df.columns:
            # Default parameters
            return_params = {
                'method': 'log',
                'periods': 1,
                'col_name': 'close'
            }
            
            # Update with provided parameters
            return_params.update(calculate_returns_kwargs)
            
            # Calculate returns
            df['returns'] = calculate_returns(
                df, 
                **return_params
            )
        
        processed_dfs[asset_name] = df
    
    # Align all dataframes
    aligned_dfs = align_dataframes(
        list(processed_dfs.values()),
        method=align_method,
        fill_method=fill_method
    )
    
    # Create a dictionary mapping asset names to aligned dataframes
    aligned_dict = dict(zip(processed_dfs.keys(), aligned_dfs))
    
    # Create multi-level columns DataFrame
    asset_dfs = []
    
    for asset_name, df in aligned_dict.items():
        # Create multi-level column index
        df.columns = pd.MultiIndex.from_product(
            [[asset_name], df.columns],
            names=['asset', 'field']
        )
        
        asset_dfs.append(df)
    
    # Concatenate along columns
    combined_df = pd.concat(asset_dfs, axis=1)
    
    logger.info(f"Combined data for {len(dfs)} assets with shape {combined_df.shape}")
    
    return combined_df


def annualize_returns(
    returns: Union[float, pd.Series, pd.DataFrame],
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