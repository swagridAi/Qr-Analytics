"""
Position sizing functions for backtesting.

This module provides implementations of various position sizing techniques including:
- Kelly criterion: Optimal sizing based on win rate and win/loss ratio
- Volatility targeting: Scaling positions to maintain consistent risk level
- Fixed fraction sizing: Allocating a fixed percentage of portfolio to positions
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def apply_kelly_sizing(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    lookback: int = 20,
    fraction: float = 0.5,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    min_periods: int = 10
) -> pd.DataFrame:
    """
    Apply Kelly Criterion position sizing based on historical returns.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price history
        lookback: Number of periods to look back for return calculation
        fraction: Fraction of Kelly to apply (0.5 = "half Kelly")
        position_column: Column containing position size
        weight_column: Column to store target weight
        min_periods: Minimum periods required for calculation
        
    Returns:
        DataFrame with Kelly-adjusted positions
    
    Example:
        >>> positions_df = apply_kelly_sizing(positions_df, prices_df, lookback=30, fraction=0.3)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply Kelly sizing: '{position_column}' column missing")
        return positions_df
    
    # Prepare price data for return calculation
    if 'timestamp' not in prices_df.columns or 'asset_id' not in prices_df.columns or 'close' not in prices_df.columns:
        logger.warning("Price data missing required columns for Kelly calculation")
        return positions_df
    
    # Calculate historical returns for each asset
    returns_by_asset = {}
    
    for asset_id in result['asset_id'].unique():
        # Get price history for this asset
        asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
        
        if len(asset_prices) < min_periods:
            logger.debug(f"Not enough price history for {asset_id}, skipping Kelly calculation")
            continue
        
        # Calculate returns
        asset_prices['return'] = asset_prices['close'].pct_change()
        
        # Store for later use
        returns_by_asset[asset_id] = asset_prices
    
    # Apply Kelly sizing to each position
    for i, row in result.iterrows():
        asset_id = row['asset_id']
        
        # Skip if no return data for this asset
        if asset_id not in returns_by_asset:
            continue
        
        # Get current position direction
        position_direction = np.sign(row[position_column])
        
        if position_direction == 0:
            continue
        
        # Get recent returns
        asset_returns = returns_by_asset[asset_id]
        
        # Get timestamp for this position
        position_timestamp = row['timestamp']
        
        # Filter returns up to current position timestamp
        past_returns = asset_returns[asset_returns['timestamp'] < position_timestamp].tail(lookback)
        
        if len(past_returns) < min_periods:
            logger.debug(f"Not enough past returns for {asset_id} at {position_timestamp}, skipping Kelly")
            continue
        
        # Calculate Kelly fraction
        win_rate = (past_returns['return'] * position_direction > 0).mean()
        avg_win = past_returns.loc[past_returns['return'] * position_direction > 0, 'return'].mean()
        avg_loss = past_returns.loc[past_returns['return'] * position_direction < 0, 'return'].mean()
        
        # Handle edge cases
        if win_rate == 0 or avg_win is None or avg_loss is None or avg_loss == 0:
            logger.debug(f"Invalid Kelly parameters for {asset_id}, skipping")
            continue
        
        # Classic Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        loss_rate = 1 - win_rate
        win_loss_ratio = abs(avg_win / avg_loss)
        
        kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply user's fraction of Kelly
        kelly_fraction = kelly_fraction * fraction
        
        # Ensure reasonable bounds
        kelly_fraction = np.clip(kelly_fraction, 0, 1)
        
        # Scale position by Kelly fraction
        original_size = abs(row[position_column])
        kelly_size = original_size * kelly_fraction
        result.loc[i, position_column] = kelly_size * position_direction
        
        # Update weight if weight column exists
        if weight_column in result.columns:
            original_weight = abs(row[weight_column])
            result.loc[i, weight_column] = original_weight * kelly_fraction * position_direction
    
    return result


def apply_vol_targeting(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    target_volatility: float,
    lookback: int = 20,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    annualization_factor: int = 252,
    min_periods: int = 10
) -> pd.DataFrame:
    """
    Apply volatility targeting to scale positions to a target risk level.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price history
        target_volatility: Target annualized volatility (e.g., 0.15 for 15%)
        lookback: Number of periods for volatility calculation
        position_column: Column containing position size
        weight_column: Column to store target weight
        annualization_factor: Factor to annualize volatility (e.g., 252 for daily data)
        min_periods: Minimum periods required for volatility calculation
        
    Returns:
        DataFrame with volatility-targeted positions
    
    Example:
        >>> positions_df = apply_vol_targeting(positions_df, prices_df, target_volatility=0.10)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply volatility targeting: '{position_column}' column missing")
        return positions_df
    
    # Prepare price data for volatility calculation
    if 'timestamp' not in prices_df.columns or 'asset_id' not in prices_df.columns or 'close' not in prices_df.columns:
        logger.warning("Price data missing required columns for volatility calculation")
        return positions_df
    
    # Calculate historical volatility for each asset
    vol_by_asset = {}
    
    for asset_id in result['asset_id'].unique():
        # Get price history for this asset
        asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
        
        if len(asset_prices) < min_periods:
            logger.debug(f"Not enough price history for {asset_id}, skipping volatility calculation")
            continue
        
        # Calculate returns
        asset_prices['return'] = asset_prices['close'].pct_change()
        
        # Calculate rolling volatility
        asset_prices['volatility'] = asset_prices['return'].rolling(
            window=lookback, min_periods=min_periods
        ).std() * np.sqrt(annualization_factor)
        
        # Store for later use
        vol_by_asset[asset_id] = asset_prices
    
    # Apply volatility targeting to each position
    for i, row in result.iterrows():
        asset_id = row['asset_id']
        
        # Skip if no volatility data for this asset
        if asset_id not in vol_by_asset:
            continue
        
        # Get current position direction
        position_direction = np.sign(row[position_column])
        
        if position_direction == 0:
            continue
        
        # Get volatility data
        asset_vols = vol_by_asset[asset_id]
        
        # Get timestamp for this position
        position_timestamp = row['timestamp']
        
        # Get most recent volatility before this position
        past_vols = asset_vols[asset_vols['timestamp'] < position_timestamp]
        
        if past_vols.empty:
            logger.debug(f"No past volatility data for {asset_id} at {position_timestamp}, skipping")
            continue
        
        current_vol = past_vols['volatility'].iloc[-1]
        
        if pd.isna(current_vol) or current_vol == 0:
            logger.debug(f"Invalid volatility for {asset_id}, skipping")
            continue
        
        # Calculate scaling factor
        vol_scale = target_volatility / current_vol
        
        # Scale position by volatility ratio
        original_size = abs(row[position_column])
        vol_size = original_size * vol_scale
        result.loc[i, position_column] = vol_size * position_direction
        
        # Update weight if weight column exists
        if weight_column in result.columns:
            original_weight = abs(row[weight_column])
            result.loc[i, weight_column] = original_weight * vol_scale * position_direction
    
    return result


def apply_fixed_fraction_sizing(
    positions_df: pd.DataFrame,
    portfolio_value: float,
    fraction: float = 0.01,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    price_column: str = 'price'
) -> pd.DataFrame:
    """
    Apply fixed fraction position sizing based on portfolio value.
    
    Args:
        positions_df: DataFrame with positions
        portfolio_value: Current portfolio value
        fraction: Fraction of portfolio to risk per position
        position_column: Column containing position size
        weight_column: Column to store target weight
        price_column: Column containing price
        
    Returns:
        DataFrame with fixed-fraction positions
    
    Example:
        >>> positions_df = apply_fixed_fraction_sizing(positions_df, portfolio_value=1000000, fraction=0.02)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply fixed fraction sizing: '{position_column}' column missing")
        return positions_df
    
    # Find appropriate price column
    price_cols = [price_column, 'close', 'execution_price', 'mid_price']
    price_col = None
    
    for col in price_cols:
        if col in result.columns:
            price_col = col
            break
    
    if price_col is None:
        logger.warning("No price column found for fixed fraction sizing")
        return positions_df
    
    # Calculate target position value
    target_value = portfolio_value * fraction
    
    # Apply sizing to each position
    for i, row in result.iterrows():
        # Get current position direction
        position_direction = np.sign(row[position_column])
        
        if position_direction == 0:
            continue
        
        # Calculate position size in units
        price = row[price_col]
        if pd.isna(price) or price <= 0:
            logger.debug(f"Invalid price for fixed fraction sizing, skipping")
            continue
        
        # Calculate new position size
        new_size = target_value / price
        result.loc[i, position_column] = new_size * position_direction
        
        # Update weight if weight column exists
        if weight_column in result.columns:
            new_weight = target_value / portfolio_value
            result.loc[i, weight_column] = new_weight * position_direction
    
    return result