"""
Utility functions for backtesting strategies.

This module provides common utilities for data processing, position management,
performance calculation, and other shared functionality across different strategies.
"""

import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings

from quant_research.core.models import Signal, Trade

logger = logging.getLogger(__name__)

# ===============================================================================
# Data Validation and Processing
# ===============================================================================

def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: List[str],
    strategy_name: str = "unknown"
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        strategy_name: Name of strategy (for logging)
        
    Returns:
        (is_valid, missing_columns): Tuple of validation result and list of missing columns
    
    Example:
        >>> is_valid, missing = validate_dataframe(prices_df, ['timestamp', 'asset_id', 'close'])
        >>> if not is_valid:
        >>>     logger.error(f"Missing columns: {missing}")
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame passed to {strategy_name} strategy")
        return False, []
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    if not is_valid:
        logger.warning(f"DataFrame missing required columns for {strategy_name}: {missing_columns}")
    
    return is_valid, missing_columns


def ensure_columns(
    df: pd.DataFrame, 
    required_columns: Dict[str, Any],
    strategy_name: str = "unknown"
) -> pd.DataFrame:
    """
    Ensure a DataFrame has all required columns, filling in defaults if missing.
    
    Args:
        df: DataFrame to process
        required_columns: Dict mapping column names to default values
        strategy_name: Name of strategy (for logging)
        
    Returns:
        DataFrame with all required columns
    
    Example:
        >>> prices_df = ensure_columns(prices_df, {
        >>>     'bid': None,  # Will be filled from close if missing
        >>>     'ask': None,  # Will be filled from close if missing
        >>>     'volume': 0   # Will use 0 as default
        >>> })
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame passed to {strategy_name} strategy")
        return df
    
    result = df.copy()
    
    for col, default_value in required_columns.items():
        if col not in result.columns:
            if default_value is None and 'close' in result.columns:
                # Special case for bid/ask
                if col == 'bid':
                    result[col] = result['close'] * 0.9999
                    logger.info(f"Created '{col}' column from 'close' in {strategy_name}")
                elif col == 'ask':
                    result[col] = result['close'] * 1.0001
                    logger.info(f"Created '{col}' column from 'close' in {strategy_name}")
                else:
                    result[col] = result['close']
                    logger.info(f"Created '{col}' column from 'close' in {strategy_name}")
            else:
                result[col] = default_value
                logger.info(f"Added missing '{col}' column with default value in {strategy_name}")
    
    return result


def resample_dataframe(
    df: pd.DataFrame,
    timeframe: str,
    price_columns: List[str] = ['open', 'high', 'low', 'close'],
    volume_columns: List[str] = ['volume'],
    group_columns: List[str] = ['asset_id', 'exchange_id']
) -> pd.DataFrame:
    """
    Resample time series data to a different timeframe.
    
    Args:
        df: DataFrame with time series data, must have 'timestamp' column
        timeframe: Target timeframe (e.g., '1h', '1d', '1w')
        price_columns: Columns to apply OHLC aggregation
        volume_columns: Columns to sum
        group_columns: Columns to group by before resampling
        
    Returns:
        Resampled DataFrame
    
    Example:
        >>> daily_data = resample_dataframe(hourly_data, '1d')
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column for resampling")
    
    # Ensure timestamp is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set up default aggregations
    aggs = {}
    
    # Add price column aggregations (OHLC)
    for col in price_columns:
        if col in df.columns:
            if col == 'open':
                aggs[col] = 'first'
            elif col == 'high':
                aggs[col] = 'max'
            elif col == 'low':
                aggs[col] = 'min'
            elif col == 'close':
                aggs[col] = 'last'
            else:
                # For other price-like columns, use last value
                aggs[col] = 'last'
    
    # Add volume column aggregations (sum)
    for col in volume_columns:
        if col in df.columns:
            aggs[col] = 'sum'
    
    # For any other numeric columns not specified, use mean
    for col in df.columns:
        if col not in aggs and col != 'timestamp' and col not in group_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                aggs[col] = 'mean'
    
    # If we have grouping columns, process by group
    resampled_dfs = []
    
    if group_columns:
        for _, group in df.groupby(group_columns):
            # Set timestamp as index for resampling
            group_indexed = group.set_index('timestamp')
            
            # Resample
            resampled = group_indexed.resample(timeframe).agg(aggs)
            
            # Reset index and add back group columns
            resampled = resampled.reset_index()
            
            # Add to list
            resampled_dfs.append(resampled)
        
        # Combine all resampled groups
        if resampled_dfs:
            result = pd.concat(resampled_dfs, ignore_index=True)
        else:
            result = pd.DataFrame(columns=df.columns)
    else:
        # No groups, just resample the entire DataFrame
        df_indexed = df.set_index('timestamp')
        resampled = df_indexed.resample(timeframe).agg(aggs)
        result = resampled.reset_index()
    
    return result


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training and testing sets based on time.
    
    Args:
        df: DataFrame with time series data, must have 'timestamp' column
        test_ratio: Proportion of data to use for testing (0-1)
        
    Returns:
        (train_df, test_df): Tuple of training and testing DataFrames
    
    Example:
        >>> train_data, test_data = split_train_test(data, test_ratio=0.3)
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column for splitting")
    
    # Ensure timestamp is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_ratio))
    
    # Split data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Split data into train ({len(train_df)} rows) and test ({len(test_df)} rows) sets")
    
    return train_df, test_df


# ===============================================================================
# Position Management
# ===============================================================================

def normalize_positions(
    positions_df: pd.DataFrame, 
    max_leverage: float = 1.0,
    group_columns: List[str] = ['timestamp'],
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Normalize positions to respect maximum leverage constraints.
    
    Args:
        positions_df: DataFrame with positions
        max_leverage: Maximum allowed leverage (sum of absolute position sizes)
        group_columns: Columns to group by when calculating leverage
        position_column: Column containing position sizes
        weight_column: Column to store target weight
        
    Returns:
        DataFrame with normalized positions
    
    Example:
        >>> positions_df = normalize_positions(positions_df, max_leverage=1.5)
    """
    if positions_df is None or positions_df.empty:
        return positions_df
    
    if position_column not in positions_df.columns:
        logger.warning(f"Column '{position_column}' not found in positions DataFrame")
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Calculate leverage by group
    leverage = result.groupby(group_columns)[position_column].sum().abs()
    
    # Calculate scaling factors, clipping at 1.0 (only scale down, not up)
    scale_factors = max_leverage / leverage
    scale_factors = scale_factors.clip(upper=1.0)
    
    # Apply scaling to positions
    if not scale_factors.empty:
        # Reset index to get scaling factors as columns
        scale_factors_df = scale_factors.reset_index()
        scale_factors_df.columns = list(scale_factors_df.columns[:-1]) + ['scale_factor']
        
        # Merge scaling factors with positions
        result = pd.merge(
            result,
            scale_factors_df,
            on=group_columns,
            how='left'
        )
        
        # Apply scaling
        result[position_column] = result[position_column] * result['scale_factor'].fillna(1.0)
        
        # Update weights if weight column exists or should be created
        if weight_column in result.columns or weight_column not in positions_df.columns:
            result[weight_column] = result[position_column]
        
        # Clean up
        result.drop('scale_factor', axis=1, inplace=True)
    
    return result


def cap_position_sizes(
    positions_df: pd.DataFrame,
    max_position: Union[float, Dict[str, float]],
    asset_column: str = 'asset_id',
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Cap position sizes to maximum limits, with per-asset customization.
    
    Args:
        positions_df: DataFrame with positions
        max_position: Maximum position size, either a float or dict mapping asset to limit
        asset_column: Column containing asset identifiers
        position_column: Column containing position sizes
        weight_column: Column to store target weight
        
    Returns:
        DataFrame with capped positions
    
    Example:
        >>> positions_df = cap_position_sizes(positions_df, {'BTC': 1.0, 'ETH': 10.0})
    """
    if positions_df is None or positions_df.empty:
        return positions_df
    
    if position_column not in positions_df.columns:
        logger.warning(f"Column '{position_column}' not found in positions DataFrame")
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Apply position caps
    if isinstance(max_position, dict):
        # Apply different caps for different assets
        for asset, cap in max_position.items():
            asset_mask = result[asset_column] == asset
            
            # Get current positions for this asset
            asset_positions = result.loc[asset_mask, position_column]
            
            # Calculate scaling factors for positions exceeding cap
            scaling = np.minimum(1.0, cap / np.abs(asset_positions))
            
            # Apply scaling
            result.loc[asset_mask, position_column] = asset_positions * scaling
    else:
        # Apply uniform cap to all assets
        scaling = np.minimum(1.0, max_position / np.abs(result[position_column]))
        result[position_column] = result[position_column] * scaling
    
    # Update weights if weight column exists or should be created
    if weight_column in result.columns or weight_column not in positions_df.columns:
        result[weight_column] = result[position_column]
    
    return result


def blend_positions(
    positions_list: List[pd.DataFrame],
    weights: List[float] = None,
    on: List[str] = ['timestamp', 'asset_id'],
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Blend multiple position DataFrames with optional weights.
    
    Args:
        positions_list: List of position DataFrames to blend
        weights: List of weights for each DataFrame (if None, equal weights)
        on: Columns to join on
        position_column: Column containing position sizes
        weight_column: Column to store target weight
        
    Returns:
        Blended positions DataFrame
    
    Example:
        >>> blended = blend_positions([momentum_positions, mean_rev_positions], [0.7, 0.3])
    """
    if not positions_list:
        return pd.DataFrame()
    
    # Handle empty DataFrames
    positions_list = [df for df in positions_list if df is not None and not df.empty]
    if not positions_list:
        return pd.DataFrame()
    
    # Set equal weights if not provided
    if weights is None:
        weights = [1.0 / len(positions_list)] * len(positions_list)
    elif len(weights) != len(positions_list):
        raise ValueError("Number of weights must match number of position DataFrames")
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Create a list of positions with their respective weights
    weighted_positions = []
    for i, df in enumerate(positions_list):
        if position_column not in df.columns:
            logger.warning(f"Column '{position_column}' not found in position DataFrame {i}")
            continue
        
        weighted_df = df.copy()
        weighted_df[position_column] = weighted_df[position_column] * weights[i]
        
        if weight_column in weighted_df.columns:
            weighted_df[weight_column] = weighted_df[weight_column] * weights[i]
        
        weighted_positions.append(weighted_df)
    
    if not weighted_positions:
        return pd.DataFrame()
    
    # Get unique values for join columns across all DataFrames
    all_keys = set()
    for df in weighted_positions:
        if all(col in df.columns for col in on):
            keys = df[on].drop_duplicates().itertuples(index=False, name=None)
            all_keys.update(keys)
    
    # Create a base DataFrame with all keys
    all_keys_df = pd.DataFrame(list(all_keys), columns=on)
    
    # Merge each weighted position DataFrame with the base
    merged = all_keys_df.copy()
    for df in weighted_positions:
        if not all(col in df.columns for col in on):
            continue
        
        # Select only necessary columns to avoid duplicate column names
        select_columns = on + [col for col in df.columns if col not in on]
        df_subset = df[select_columns]
        
        # Merge
        merged = pd.merge(merged, df_subset, on=on, how='left')
    
    # Aggregate position and weight columns
    agg_columns = {}
    for col in merged.columns:
        if col.startswith(f"{position_column}_"):
            agg_columns[col] = position_column
        elif col.startswith(f"{weight_column}_"):
            agg_columns[col] = weight_column
        elif col == position_column:
            agg_columns[position_column] = position_column
        elif col == weight_column:
            agg_columns[weight_column] = weight_column
    
    # Create final blended DataFrame
    blended = merged.copy()
    
    # Sum position columns
    position_cols = [col for col in merged.columns if col.startswith(f"{position_column}_") or col == position_column]
    if position_cols:
        blended[position_column] = merged[position_cols].sum(axis=1, skipna=True)
    
    # Sum weight columns
    weight_cols = [col for col in merged.columns if col.startswith(f"{weight_column}_") or col == weight_column]
    if weight_cols:
        blended[weight_column] = merged[weight_cols].sum(axis=1, skipna=True)
    
    # Select only necessary columns for final output
    final_columns = on + [position_column, weight_column]
    final_columns = [col for col in final_columns if col in blended.columns]
    
    return blended[final_columns]


def smooth_positions(
    positions_df: pd.DataFrame,
    lookback: int = 3,
    group_columns: List[str] = ['asset_id'],
    time_column: str = 'timestamp',
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Smooth position changes over time to reduce turnover.
    
    Args:
        positions_df: DataFrame with positions
        lookback: Number of periods to use for smoothing
        group_columns: Columns to group by when smoothing
        time_column: Column containing timestamp
        position_column: Column containing position sizes
        weight_column: Column to store target weight
        
    Returns:
        DataFrame with smoothed positions
    
    Example:
        >>> positions_df = smooth_positions(positions_df, lookback=5)
    """
    if positions_df is None or positions_df.empty:
        return positions_df
    
    if position_column not in positions_df.columns:
        logger.warning(f"Column '{position_column}' not found in positions DataFrame")
        return positions_df
    
    if time_column not in positions_df.columns:
        logger.warning(f"Column '{time_column}' not found in positions DataFrame")
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[time_column]):
        result[time_column] = pd.to_datetime(result[time_column])
    
    # Sort by time
    result = result.sort_values([time_column] + group_columns)
    
    # Apply smoothing by group
    group_columns_with_time = group_columns + [time_column]
    for _, group in result.groupby(group_columns):
        # Sort by time
        group = group.sort_values(time_column)
        
        # Calculate smoothed positions using rolling window
        group[f'{position_column}_smooth'] = group[position_column].rolling(
            window=lookback, min_periods=1
        ).mean()
        
        # Update positions in result
        result.loc[group.index, f'{position_column}_smooth'] = group[f'{position_column}_smooth']
    
    # Replace original positions with smoothed
    result[position_column] = result[f'{position_column}_smooth']
    result.drop(f'{position_column}_smooth', axis=1, inplace=True)
    
    # Update weights if weight column exists or should be created
    if weight_column in result.columns or weight_column not in positions_df.columns:
        result[weight_column] = result[position_column]
    
    return result


# ===============================================================================
# Performance Metrics
# ===============================================================================

def calculate_returns(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    position_column: str = 'position',
    price_column: str = 'close',
    join_columns: List[str] = ['timestamp', 'asset_id']
) -> pd.DataFrame:
    """
    Calculate returns for a series of positions.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        position_column: Column containing position sizes
        price_column: Column containing price data
        join_columns: Columns to join positions and prices
        
    Returns:
        DataFrame with returns
    
    Example:
        >>> returns_df = calculate_returns(positions_df, prices_df)
    """
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()
    
    if prices_df is None or prices_df.empty:
        return pd.DataFrame()
    
    # Verify required columns
    missing_pos_cols = [col for col in join_columns + [position_column] if col not in positions_df.columns]
    missing_price_cols = [col for col in join_columns + [price_column] if col not in prices_df.columns]
    
    if missing_pos_cols:
        logger.warning(f"Missing columns in positions DataFrame: {missing_pos_cols}")
        return pd.DataFrame()
    
    if missing_price_cols:
        logger.warning(f"Missing columns in prices DataFrame: {missing_price_cols}")
        return pd.DataFrame()
    
    # Ensure timestamps are datetime
    positions = positions_df.copy()
    prices = prices_df.copy()
    
    if 'timestamp' in join_columns:
        if not pd.api.types.is_datetime64_any_dtype(positions['timestamp']):
            positions['timestamp'] = pd.to_datetime(positions['timestamp'])
        
        if not pd.api.types.is_datetime64_any_dtype(prices['timestamp']):
            prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    
    # Calculate price changes
    price_changes = prices.copy()
    price_changes['price_change'] = prices.groupby(['asset_id'])[price_column].pct_change()
    
    # Merge positions with price changes
    merged = pd.merge(
        positions[join_columns + [position_column]],
        price_changes[join_columns + ['price_change']],
        on=join_columns,
        how='left'
    )
    
    # Calculate returns
    merged['return'] = merged[position_column] * merged['price_change']
    
    # Calculate cumulative returns
    merged['cumulative_return'] = (1 + merged['return']).cumprod() - 1
    
    return merged


def calculate_metrics(returns_df: pd.DataFrame, return_column: str = 'return') -> Dict[str, float]:
    """
    Calculate performance metrics from returns.
    
    Args:
        returns_df: DataFrame with returns
        return_column: Column containing return values
        
    Returns:
        Dictionary of performance metrics
    
    Example:
        >>> metrics = calculate_metrics(returns_df)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """
    if returns_df is None or returns_df.empty:
        return {}
    
    if return_column not in returns_df.columns:
        logger.warning(f"Column '{return_column}' not found in returns DataFrame")
        return {}
    
    returns = returns_df[return_column].dropna()
    
    if len(returns) < 2:
        logger.warning("Not enough return data to calculate metrics")
        return {}
    
    # Calculate basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe & Sortino
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Win/loss metrics
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
    
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    profit_factor = (returns[returns > 0].sum() / -returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
    
    # Collect metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'winning_days': winning_days,
        'losing_days': losing_days,
        'number_of_trades': len(returns)
    }
    
    return metrics


def calculate_turnover(
    positions_df: pd.DataFrame,
    position_column: str = 'position',
    time_column: str = 'timestamp',
    group_columns: List[str] = ['asset_id']
) -> float:
    """
    Calculate turnover for a series of positions.
    
    Args:
        positions_df: DataFrame with positions
        position_column: Column containing position sizes
        time_column: Column containing timestamp
        group_columns: Columns to group by when calculating changes
        
    Returns:
        Average daily turnover
    
    Example:
        >>> turnover = calculate_turnover(positions_df)
        >>> print(f"Average Daily Turnover: {turnover:.2%}")
    """
    if positions_df is None or positions_df.empty:
        return 0.0
    
    if position_column not in positions_df.columns:
        logger.warning(f"Column '{position_column}' not found in positions DataFrame")
        return 0.0
    
    if time_column not in positions_df.columns:
        logger.warning(f"Column '{time_column}' not found in positions DataFrame")
        return 0.0
    
    # Ensure timestamp is datetime
    positions = positions_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(positions[time_column]):
        positions[time_column] = pd.to_datetime(positions[time_column])
    
    # Sort by time
    positions = positions.sort_values([time_column] + group_columns)
    
    # Calculate position changes
    turnover_by_group = []
    
    for _, group in positions.groupby(group_columns):
        # Sort by time
        group = group.sort_values(time_column)
        
        # Calculate position changes
        group['position_change'] = group[position_column].diff().abs()
        
        # Add to list
        turnover_by_group.append(group)
    
    # Combine results
    if not turnover_by_group:
        return 0.0
    
    all_changes = pd.concat(turnover_by_group, ignore_index=True)
    
    # Group by time and sum changes
    daily_turnover = all_changes.groupby(time_column)['position_change'].sum()
    
    # Calculate average daily turnover
    avg_turnover = daily_turnover.mean() if len(daily_turnover) > 0 else 0.0
    
    return avg_turnover


def detect_survivorship_bias(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    asset_column: str = 'asset_id',
    time_column: str = 'timestamp'
) -> Dict[str, Any]:
    """
    Detect potential survivorship bias in backtest data.
    
    Args:
        signals_df: DataFrame with trading signals
        prices_df: DataFrame with price data
        asset_column: Column containing asset identifiers
        time_column: Column containing timestamp
        
    Returns:
        Dictionary with survivorship bias indicators
    
    Example:
        >>> bias = detect_survivorship_bias(signals_df, prices_df)
        >>> if bias['potential_bias']:
        >>>     print(f"Warning: {bias['missing_assets']} assets missing from recent data")
    """
    if signals_df is None or signals_df.empty or prices_df is None or prices_df.empty:
        return {
            'potential_bias': False,
            'missing_assets': 0,
            'missing_asset_ids': [],
            'analysis': "Insufficient data to analyze survivorship bias"
        }
    
    # Verify required columns
    for df, df_name in [(signals_df, 'signals'), (prices_df, 'prices')]:
        if asset_column not in df.columns:
            logger.warning(f"Column '{asset_column}' not found in {df_name} DataFrame")
            return {
                'potential_bias': False,
                'missing_assets': 0,
                'missing_asset_ids': [],
                'analysis': f"Column '{asset_column}' not found in {df_name} DataFrame"
            }
        
        if time_column not in df.columns:
            logger.warning(f"Column '{time_column}' not found in {df_name} DataFrame")
            return {
                'potential_bias': False,
                'missing_assets': 0,
                'missing_asset_ids': [],
                'analysis': f"Column '{time_column}' not found in {df_name} DataFrame"
            }
    
    # Ensure timestamps are datetime
    signals = signals_df.copy()
    prices = prices_df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(signals[time_column]):
        signals[time_column] = pd.to_datetime(signals[time_column])
    
    if not pd.api.types.is_datetime64_any_dtype(prices[time_column]):
        prices[time_column] = pd.to_datetime(prices[time_column])
    
    # Get unique timestamps and sort
    signal_times = sorted(signals[time_column].unique())
    price_times = sorted(prices[time_column].unique())
    
    if len(signal_times) < 2 or len(price_times) < 2:
        return {
            'potential_bias': False,
            'missing_assets': 0,
            'missing_asset_ids': [],
            'analysis': "Insufficient time points to analyze survivorship bias"
        }
    
    # Analyze first and last quartile of data
    first_quartile_idx = len(signal_times) // 4
    last_quartile_idx = (len(signal_times) * 3) // 4
    
    first_quartile_times = signal_times[:first_quartile_idx]
    last_quartile_times = signal_times[last_quartile_idx:]
    
    # Get assets in first and last quartile
    first_quartile_assets = set(signals[signals[time_column].isin(first_quartile_times)][asset_column].unique())
    last_quartile_assets = set(signals[signals[time_column].isin(last_quartile_times)][asset_column].unique())
    
    # Find missing assets
    missing_assets = first_quartile_assets - last_quartile_assets
    
    # Calculate metrics
    missing_count = len(missing_assets)
    missing_pct = missing_count / len(first_quartile_assets) if first_quartile_assets else 0
    
    # Analyze results
    potential_bias = missing_pct > 0.05  # Flag if more than 5% of assets disappeared
    
    analysis = (
        f"{missing_count} assets ({missing_pct:.1%}) present in the first quartile "
        f"are missing from the last quartile."
    )
    
    if potential_bias:
        analysis += (
            " This suggests potential survivorship bias. "
            "Consider including delisted/failed assets in backtest."
        )
    
    return {
        'potential_bias': potential_bias,
        'missing_assets': missing_count,
        'missing_asset_ids': list(missing_assets),
        'missing_percentage': missing_pct,
        'analysis': analysis
    }


def detect_lookahead_bias(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    time_column: str = 'timestamp'
) -> Dict[str, Any]:
    """
    Detect potential lookahead bias in signals.
    
    Args:
        signals_df: DataFrame with trading signals
        prices_df: DataFrame with price data
        time_column: Column containing timestamp
        
    Returns:
        Dictionary with lookahead bias indicators
    
    Example:
        >>> bias = detect_lookahead_bias(signals_df, prices_df)
        >>> if bias['potential_bias']:
        >>>     print(f"Warning: {bias['analysis']}")
    """
    if signals_df is None or signals_df.empty or prices_df is None or prices_df.empty:
        return {
            'potential_bias': False,
            'analysis': "Insufficient data to analyze lookahead bias"
        }
    
    # Verify required columns
    for df, df_name in [(signals_df, 'signals'), (prices_df, 'prices')]:
        if time_column not in df.columns:
            logger.warning(f"Column '{time_column}' not found in {df_name} DataFrame")
            return {
                'potential_bias': False,
                'analysis': f"Column '{time_column}' not found in {df_name} DataFrame"
            }
    
    # Ensure timestamps are datetime
    signals = signals_df.copy()
    prices = prices_df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(signals[time_column]):
        signals[time_column] = pd.to_datetime(signals[time_column])
    
    if not pd.api.types.is_datetime64_any_dtype(prices[time_column]):
        prices[time_column] = pd.to_datetime(prices[time_column])
    
    # Get unique timestamps and sort
    signal_times = sorted(signals[time_column].unique())
    price_times = sorted(prices[time_column].unique())
    
    if len(signal_times) < 2 or len(price_times) < 2:
        return {
            'potential_bias': False,
            'analysis': "Insufficient time points to analyze lookahead bias"
        }
    
    # Check for signals referencing future prices
    future_signals = 0
    
    for signal_time in signal_times:
        # Find closest price time before signal time
        prev_prices = [pt for pt in price_times if pt <= signal_time]
        
        if not prev_prices:
            # Signal appears before any price data - potential issue
            future_signals += 1
            continue
        
        # Get most recent price time
        latest_price_time = max(prev_prices)
        
        # Check time difference
        time_diff = signal_time - latest_price_time
        
        # Flag suspicious time differences
        if time_diff.total_seconds() < 1:  # Less than 1 second difference
            future_signals += 1
    
    # Calculate metrics
    future_pct = future_signals / len(signal_times) if signal_times else 0
    
    # Analyze results
    potential_bias = future_pct > 0.01  # Flag if more than 1% of signals are suspicious
    
    analysis = f"{future_signals} signals ({future_pct:.1%}) may be using future price information."
    
    if potential_bias:
        analysis += (
            " This suggests potential lookahead bias. "
            "Ensure signals are generated using only past data."
        )
    
    return {
        'potential_bias': potential_bias,
        'suspicious_signals': future_signals,
        'suspicious_percentage': future_pct,
        'analysis': analysis
    }


# ===============================================================================
# Miscellaneous Utilities
# ===============================================================================

def generate_sample_data(
    n_assets: int = 5,
    n_days: int = 252,
    start_date: str = '2022-01-01',
    volatility: float = 0.02,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sample price and signal data for testing.
    
    Args:
        n_assets: Number of assets to generate
        n_days: Number of trading days
        start_date: Starting date
        volatility: Daily price volatility
        seed: Random seed for reproducibility
        
    Returns:
        (prices_df, signals_df): Tuple of price and signal DataFrames
    
    Example:
        >>> prices_df, signals_df = generate_sample_data(n_assets=3, n_days=100)
        >>> # Use for testing strategy implementations
    """
    np.random.seed(seed)
    
    # Create date range
    start_date = pd.to_datetime(start_date)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Create asset IDs
    asset_ids = [f'ASSET_{i}' for i in range(1, n_assets + 1)]
    
    # Generate price data
    prices_data = []
    
    for asset_id in asset_ids:
        # Generate random walk
        returns = np.random.normal(0.0005, volatility, n_days)
        prices = 100 * np.cumprod(1 + returns)
        
        for i, date in enumerate(dates):
            price = prices[i]
            prices_data.append({
                'timestamp': date,
                'asset_id': asset_id,
                'open': price * (1 - volatility/2),
                'high': price * (1 + volatility),
                'low': price * (1 - volatility),
                'close': price,
                'volume': np.random.randint(100000, 1000000)
            })
    
    prices_df = pd.DataFrame(prices_data)
    
    # Generate signal data (with some randomness)
    signals_data = []
    
    for asset_id in asset_ids:
        asset_prices = prices_df[prices_df['asset_id'] == asset_id]
        
        for i in range(1, n_days):
            # Only generate signal occasionally
            if np.random.random() < 0.3:
                date = dates[i]
                
                # Calculate some indicator
                price_change = asset_prices.iloc[i]['close'] / asset_prices.iloc[i-1]['close'] - 1
                
                # Determine signal direction and strength
                if price_change > 0.01:
                    direction = 1
                    strength = min(0.5 + price_change * 20, 1.0)
                elif price_change < -0.01:
                    direction = -1
                    strength = min(0.5 + abs(price_change) * 20, 1.0)
                else:
                    direction = 0
                    strength = 0.1
                
                # Add some variety to the signals
                strategy = np.random.choice(['momentum', 'mean_reversion', 'regime_hmm'])
                
                signals_data.append({
                    'timestamp': date,
                    'asset_id': asset_id,
                    'strategy': strategy,
                    'direction': direction,
                    'strength': strength,
                    'z_score': np.random.normal(0, 1) if strategy == 'mean_reversion' else None,
                    'regime': np.random.choice(['trending', 'mean_reverting', 'high_volatility']) if strategy == 'regime_hmm' else None,
                    'probability': np.random.random() if strategy == 'regime_hmm' else None
                })
    
    signals_df = pd.DataFrame(signals_data)
    
    return prices_df, signals_df


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 10000,
    time_column: str = 'timestamp'
) -> List[pd.DataFrame]:
    """
    Split a large DataFrame into smaller chunks for processing.
    
    Args:
        df: Large DataFrame to split
        chunk_size: Maximum rows per chunk
        time_column: Column to ensure consistent timestamps in chunks
        
    Returns:
        List of DataFrame chunks
    
    Example:
        >>> chunks = chunk_dataframe(large_prices_df, chunk_size=50000)
        >>> for chunk in chunks:
        >>>     process_chunk(chunk)
    """
    if df is None or df.empty:
        return []
    
    # If smaller than chunk size, return as is
    if len(df) <= chunk_size:
        return [df]
    
    # If time column exists, use it to ensure consistent chunks
    if time_column in df.columns:
        # Sort by time
        df = df.sort_values(time_column)
        
        # Get unique timestamps
        unique_times = df[time_column].unique()
        
        # Calculate chunks based on number of unique timestamps
        n_chunks = max(1, int(np.ceil(len(unique_times) / chunk_size)))
        time_splits = np.array_split(unique_times, n_chunks)
        
        # Create chunks based on time splits
        chunks = []
        for time_split in time_splits:
            chunk = df[df[time_column].isin(time_split)].copy()
            chunks.append(chunk)
    else:
        # Simple row-based chunking
        n_chunks = max(1, int(np.ceil(len(df) / chunk_size)))
        chunks = np.array_split(df, n_chunks)
    
    return chunks


def parallel_apply(
    func: Callable,
    df_list: List[pd.DataFrame],
    n_jobs: int = -1,
    verbosity: int = 1
) -> List:
    """
    Apply a function to multiple DataFrames in parallel.
    
    Args:
        func: Function to apply to each DataFrame
        df_list: List of DataFrames to process
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbosity: Logging verbosity (0-3)
        
    Returns:
        List of function results
    
    Example:
        >>> def process_chunk(chunk):
        >>>     # Do something with the chunk
        >>>     return result
        >>> 
        >>> chunks = chunk_dataframe(large_df)
        >>> results = parallel_apply(process_chunk, chunks)
    """
    try:
        from joblib import Parallel, delayed
        
        # Process in parallel
        results = Parallel(n_jobs=n_jobs, verbose=verbosity)(
            delayed(func)(df) for df in df_list
        )
        
        return results
    
    except ImportError:
        # Fallback to serial processing
        logger.warning("joblib not available, processing in serial mode")
        return [func(df) for df in df_list]


def save_to_json(obj: Any, filepath: str) -> bool:
    """
    Save an object to JSON file with error handling.
    
    Args:
        obj: Object to save (must be JSON serializable)
        filepath: Path to save the file
        
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> metrics = calculate_metrics(returns_df)
        >>> save_to_json(metrics, 'results/strategy_metrics.json')
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Custom JSON encoder for NumPy types and special objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif pd.isna(obj):
                    return None
                return super().default(obj)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(obj, f, cls=CustomEncoder, indent=2)
        
        logger.info(f"Successfully saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {str(e)}")
        return False


def load_from_json(filepath: str) -> Any:
    """
    Load an object from JSON file with error handling.
    
    Args:
        filepath: Path to load the file from
        
    Returns:
        Loaded object or None if error
    
    Example:
        >>> metrics = load_from_json('results/strategy_metrics.json')
        >>> if metrics:
        >>>     print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            obj = json.load(f)
        
        logger.info(f"Successfully loaded from {filepath}")
        return obj
    
    except Exception as e:
        logger.error(f"Error loading from {filepath}: {str(e)}")
        return None