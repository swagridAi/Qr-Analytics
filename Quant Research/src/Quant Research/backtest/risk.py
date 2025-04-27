"""
Risk management functions for backtesting.

This module provides implementations of various risk management techniques including:
- Position sizing (Kelly, volatility targeting, fixed fraction)
- Stop-loss mechanisms
- Drawdown protection
- Exposure controls
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_research.core.models import Trade

logger = logging.getLogger(__name__)


# ===============================================================================
# Position Sizing Functions
# ===============================================================================

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


def apply_position_limits(
    positions_df: pd.DataFrame,
    max_position_size: Union[float, Dict[str, float]],
    max_portfolio_pct: Optional[float] = None,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    portfolio_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply position size limits to control maximum exposure.
    
    Args:
        positions_df: DataFrame with positions
        max_position_size: Maximum absolute position size (per asset or overall)
        max_portfolio_pct: Maximum percentage of portfolio per position
        position_column: Column containing position size
        weight_column: Column to store target weight
        portfolio_value: Current portfolio value (required if max_portfolio_pct is specified)
        
    Returns:
        DataFrame with limited positions
        
    Example:
        >>> positions_df = apply_position_limits(
        >>>     positions_df, 
        >>>     max_position_size={'AAPL': 1000, 'default': 500},
        >>>     max_portfolio_pct=0.05,
        >>>     portfolio_value=1000000
        >>> )
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply position limits: '{position_column}' column missing")
        return positions_df
    
    # Apply absolute position size limits
    if isinstance(max_position_size, dict):
        # Apply per-asset limits
        for i, row in result.iterrows():
            asset_id = row['asset_id']
            position_direction = np.sign(row[position_column])
            current_size = abs(row[position_column])
            
            # Get limit for this asset (or default)
            limit = max_position_size.get(asset_id, max_position_size.get('default', float('inf')))
            
            # Apply limit
            if current_size > limit:
                result.loc[i, position_column] = limit * position_direction
    else:
        # Apply global limit
        result[position_column] = result[position_column].clip(
            lower=-max_position_size, upper=max_position_size
        )
    
    # Apply portfolio percentage limits if specified
    if max_portfolio_pct is not None:
        if portfolio_value is None or portfolio_value <= 0:
            logger.warning("Valid portfolio_value required for percentage limits")
        else:
            # Find appropriate price column
            price_cols = ['price', 'close', 'execution_price', 'mid_price']
            price_col = None
            
            for col in price_cols:
                if col in result.columns:
                    price_col = col
                    break
            
            if price_col is not None:
                for i, row in result.iterrows():
                    position_direction = np.sign(row[position_column])
                    current_size = abs(row[position_column])
                    price = row[price_col]
                    
                    if pd.isna(price) or price <= 0:
                        continue
                    
                    # Calculate position value
                    position_value = current_size * price
                    
                    # Calculate maximum allowed size
                    max_value = portfolio_value * max_portfolio_pct
                    
                    # Apply limit if needed
                    if position_value > max_value:
                        new_size = max_value / price
                        result.loc[i, position_column] = new_size * position_direction
            else:
                logger.warning("No price column found for percentage position limits")
    
    # Update weights if weight column exists
    if weight_column in result.columns:
        # Recalculate weights based on new position sizes
        for i, row in result.iterrows():
            position_direction = np.sign(row[position_column])
            current_size = abs(row[position_column])
            original_weight = abs(row[weight_column])
            original_size = abs(row[position_column])
            
            # Scale weight proportionally to position size change
            if original_size > 0:
                new_weight = original_weight * (current_size / original_size)
                result.loc[i, weight_column] = new_weight * position_direction
    
    return result


# ===============================================================================
# Stop Loss Functions
# ===============================================================================

def apply_stop_loss(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    stop_loss_pct: float,
    trailing: bool = False,
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Apply stop loss to positions based on price movement.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        stop_loss_pct: Stop loss percentage (e.g., 5 for 5%)
        trailing: Whether to use trailing stop loss
        position_column: Column containing position size
        weight_column: Column to store target weight
        
    Returns:
        DataFrame with stop loss applied
        
    Example:
        >>> positions_df = apply_stop_loss(positions_df, prices_df, stop_loss_pct=3, trailing=True)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply stop loss: '{position_column}' column missing")
        return positions_df
    
    # Verify price data
    required_cols = ['timestamp', 'asset_id', 'close', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in prices_df.columns]
    
    if missing_cols:
        logger.warning(f"Price data missing required columns for stop loss: {missing_cols}")
        # Try to adapt if we at least have close prices
        if 'close' in prices_df.columns and 'timestamp' in prices_df.columns and 'asset_id' in prices_df.columns:
            # If high/low are missing, use close price instead
            if 'high' not in prices_df.columns:
                prices_df['high'] = prices_df['close']
            if 'low' not in prices_df.columns:
                prices_df['low'] = prices_df['close']
        else:
            return positions_df
    
    # Initialize stop tracking state
    stop_state = {}
    
    # Process positions in chronological order
    positions_by_time = result.sort_values('timestamp')
    positions_grouped = positions_by_time.groupby(['asset_id'])
    
    for asset_id, asset_positions in positions_grouped:
        # Get price data for this asset
        asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
        
        if asset_prices.empty:
            logger.warning(f"No price data found for {asset_id}, skipping stop loss")
            continue
        
        # Process each position for this asset
        last_position = 0
        entry_price = None
        
        for i, row in asset_positions.iterrows():
            current_position = row[position_column]
            position_direction = np.sign(current_position)
            current_time = row['timestamp']
            
            # Check if position changed direction or was closed
            position_changed = (np.sign(last_position) != position_direction) or (last_position != 0 and current_position == 0)
            
            if position_changed:
                # Reset stop tracking for new position
                if asset_id in stop_state:
                    del stop_state[asset_id]
                
                # Store new position entry
                if current_position != 0:
                    # Get entry price (latest price before or at position timestamp)
                    entry_data = asset_prices[asset_prices['timestamp'] <= current_time]
                    
                    if not entry_data.empty:
                        entry_price = entry_data.iloc[-1]['close']
                        
                        # Initialize stop level
                        stop_level = (
                            entry_price * (1 - stop_loss_pct / 100) if position_direction > 0 else 
                            entry_price * (1 + stop_loss_pct / 100)
                        )
                        
                        stop_state[asset_id] = {
                            'direction': position_direction,
                            'entry_price': entry_price,
                            'stop_level': stop_level,
                            'high_since_entry': entry_price if position_direction > 0 else float('-inf'),
                            'low_since_entry': entry_price if position_direction < 0 else float('inf')
                        }
            
            # Check for stop loss trigger if we have an active position
            if current_position != 0 and asset_id in stop_state:
                state = stop_state[asset_id]
                
                # Get price data between last position and current position
                price_window = asset_prices[
                    (asset_prices['timestamp'] > row['timestamp']) & 
                    (asset_prices['timestamp'] <= current_time)
                ]
                
                if not price_window.empty:
                    # Update high/low since entry
                    if position_direction > 0:
                        # For long positions, track high for trailing stop
                        max_high = price_window['high'].max()
                        if max_high > state['high_since_entry']:
                            state['high_since_entry'] = max_high
                            
                            # Update trailing stop if enabled
                            if trailing:
                                new_stop = max_high * (1 - stop_loss_pct / 100)
                                if new_stop > state['stop_level']:
                                    state['stop_level'] = new_stop
                    else:
                        # For short positions, track low for trailing stop
                        min_low = price_window['low'].min()
                        if min_low < state['low_since_entry']:
                            state['low_since_entry'] = min_low
                            
                            # Update trailing stop if enabled
                            if trailing:
                                new_stop = min_low * (1 + stop_loss_pct / 100)
                                if new_stop < state['stop_level']:
                                    state['stop_level'] = new_stop
                    
                    # Check if stop was triggered
                    stop_triggered = False
                    
                    if position_direction > 0:
                        # Long position stop is triggered if price goes below stop level
                        if price_window['low'].min() <= state['stop_level']:
                            stop_triggered = True
                    else:
                        # Short position stop is triggered if price goes above stop level
                        if price_window['high'].max() >= state['stop_level']:
                            stop_triggered = True
                    
                    if stop_triggered:
                        # Zero out the position
                        result.loc[i, position_column] = 0
                        
                        # Update weight if weight column exists
                        if weight_column in result.columns:
                            result.loc[i, weight_column] = 0
                        
                        # Log stop loss
                        price_change_pct = (
                            (state['stop_level'] / state['entry_price'] - 1) * 100 if position_direction > 0 else
                            (state['entry_price'] / state['stop_level'] - 1) * 100
                        )
                        
                        logger.info(
                            f"Stop loss triggered for {asset_id} at {current_time}, "
                            f"direction={position_direction}, loss={price_change_pct:.2f}%"
                        )
                        
                        # Remove from tracking
                        del stop_state[asset_id]
            
            # Update for next iteration
            last_position = current_position
    
    return result


def apply_volatility_stop(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    atr_multiple: float = 3.0,
    atr_periods: int = 14,
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Apply Average True Range (ATR) based stop loss.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        atr_multiple: Multiple of ATR for stop distance
        atr_periods: Number of periods for ATR calculation
        position_column: Column containing position size
        weight_column: Column to store target weight
        
    Returns:
        DataFrame with ATR stop loss applied
        
    Example:
        >>> positions_df = apply_volatility_stop(positions_df, prices_df, atr_multiple=2.5)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply ATR stop: '{position_column}' column missing")
        return positions_df
    
    # Verify price data
    required_cols = ['timestamp', 'asset_id', 'close', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in prices_df.columns]
    
    if missing_cols:
        logger.warning(f"Price data missing required columns for ATR stop: {missing_cols}")
        return positions_df
    
    # Calculate ATR for each asset
    atr_by_asset = {}
    
    for asset_id in result['asset_id'].unique():
        # Get price data for this asset
        asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
        
        if len(asset_prices) < atr_periods + 1:
            logger.warning(f"Not enough price history for {asset_id} ATR calculation")
            continue
        
        # Calculate True Range (TR)
        asset_prices['prev_close'] = asset_prices['close'].shift(1)
        
        # TR is the greatest of:
        # 1. Current High - Current Low
        # 2. |Current High - Previous Close|
        # 3. |Current Low - Previous Close|
        asset_prices['tr1'] = asset_prices['high'] - asset_prices['low']
        asset_prices['tr2'] = (asset_prices['high'] - asset_prices['prev_close']).abs()
        asset_prices['tr3'] = (asset_prices['low'] - asset_prices['prev_close']).abs()
        asset_prices['tr'] = asset_prices[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR (simple moving average of TR)
        asset_prices['atr'] = asset_prices['tr'].rolling(window=atr_periods).mean()
        
        # Store for use
        atr_by_asset[asset_id] = asset_prices
    
    # Initialize stop tracking state
    stop_state = {}
    
    # Process positions in chronological order
    positions_by_time = result.sort_values('timestamp')
    positions_grouped = positions_by_time.groupby(['asset_id'])
    
    for asset_id, asset_positions in positions_grouped:
        # Skip if no ATR data for this asset
        if asset_id not in atr_by_asset:
            continue
        
        # Get price/ATR data for this asset
        asset_data = atr_by_asset[asset_id]
        
        # Process each position for this asset
        last_position = 0
        
        for i, row in asset_positions.iterrows():
            current_position = row[position_column]
            position_direction = np.sign(current_position)
            current_time = row['timestamp']
            
            # Check if position changed direction or was closed
            position_changed = (np.sign(last_position) != position_direction) or (last_position != 0 and current_position == 0)
            
            if position_changed:
                # Reset stop tracking for new position
                if asset_id in stop_state:
                    del stop_state[asset_id]
                
                # Store new position entry
                if current_position != 0:
                    # Get entry data (latest price/ATR before or at position timestamp)
                    entry_data = asset_data[asset_data['timestamp'] <= current_time]
                    
                    if not entry_data.empty and not pd.isna(entry_data.iloc[-1]['atr']):
                        entry_price = entry_data.iloc[-1]['close']
                        current_atr = entry_data.iloc[-1]['atr']
                        
                        # Calculate ATR-based stop distance
                        stop_distance = current_atr * atr_multiple
                        
                        # Set stop level
                        stop_level = (
                            entry_price - stop_distance if position_direction > 0 else 
                            entry_price + stop_distance
                        )
                        
                        stop_state[asset_id] = {
                            'direction': position_direction,
                            'entry_price': entry_price,
                            'stop_level': stop_level,
                            'atr': current_atr
                        }
            
            # Check for stop loss trigger if we have an active position
            if current_position != 0 and asset_id in stop_state:
                state = stop_state[asset_id]
                
                # Get price data between last position and current position
                price_window = asset_data[
                    (asset_data['timestamp'] > row['timestamp']) & 
                    (asset_data['timestamp'] <= current_time)
                ]
                
                if not price_window.empty:
                    # Check if stop was triggered
                    stop_triggered = False
                    
                    if position_direction > 0:
                        # Long position stop is triggered if price goes below stop level
                        if price_window['low'].min() <= state['stop_level']:
                            stop_triggered = True
                    else:
                        # Short position stop is triggered if price goes above stop level
                        if price_window['high'].max() >= state['stop_level']:
                            stop_triggered = True
                    
                    if stop_triggered:
                        # Zero out the position
                        result.loc[i, position_column] = 0
                        
                        # Update weight if weight column exists
                        if weight_column in result.columns:
                            result.loc[i, weight_column] = 0
                        
                        # Log stop loss
                        price_change_pct = (
                            (state['stop_level'] / state['entry_price'] - 1) * 100 if position_direction > 0 else
                            (state['entry_price'] / state['stop_level'] - 1) * 100
                        )
                        
                        logger.info(
                            f"ATR stop triggered for {asset_id} at {current_time}, "
                            f"direction={position_direction}, loss={price_change_pct:.2f}%, "
                            f"ATR={state['atr']:.4f}, multiple={atr_multiple}"
                        )
                        
                        # Remove from tracking
                        del stop_state[asset_id]
            
            # Update for next iteration
            last_position = current_position
    
    return result


def apply_time_stop(
    positions_df: pd.DataFrame,
    max_holding_periods: int,
    position_column: str = 'position',
    weight_column: str = 'target_weight'
) -> pd.DataFrame:
    """
    Apply time-based stop to close positions after a maximum holding period.
    
    Args:
        positions_df: DataFrame with positions
        max_holding_periods: Maximum number of periods to hold a position
        position_column: Column containing position size
        weight_column: Column to store target weight
        
    Returns:
        DataFrame with time stop applied
        
    Example:
        >>> positions_df = apply_time_stop(positions_df, max_holding_periods=10)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply time stop: '{position_column}' column missing")
        return positions_df
    
    # Verify timestamp column exists
    if 'timestamp' not in result.columns:
        logger.warning("Cannot apply time stop: 'timestamp' column missing")
        return positions_df
    
    # Initialize holding period tracking
    holding_periods = {}
    
    # Process positions in chronological order
    positions_by_time = result.sort_values('timestamp')
    positions_grouped = positions_by_time.groupby(['asset_id'])
    
    for asset_id, asset_positions in positions_grouped:
        # Process each position for this asset
        last_position = 0
        position_start = None
        
        for i, row in asset_positions.iterrows():
            current_position = row[position_column]
            position_direction = np.sign(current_position)
            current_time = row['timestamp']
            
            # Check if position changed direction or was closed
            position_changed = (np.sign(last_position) != position_direction) or (last_position != 0 and current_position == 0)
            
            if position_changed:
                # Reset holding period tracking for new position
                if current_position != 0:
                    # Start tracking new position
                    holding_periods[asset_id] = {
                        'direction': position_direction,
                        'start_time': current_time,
                        'periods': 0
                    }
                else:
                    # Position closed, remove tracking
                    if asset_id in holding_periods:
                        del holding_periods[asset_id]
            
            # Update holding period and check for time stop
            if current_position != 0 and asset_id in holding_periods:
                # Increment holding period
                holding_periods[asset_id]['periods'] += 1
                
                # Check if max period reached
                if holding_periods[asset_id]['periods'] >= max_holding_periods:
                    # Close position
                    result.loc[i, position_column] = 0
                    
                    # Update weight if weight column exists
                    if weight_column in result.columns:
                        result.loc[i, weight_column] = 0
                    
                    # Log time stop
                    logger.info(
                        f"Time stop triggered for {asset_id} at {current_time}, "
                        f"direction={position_direction}, periods={max_holding_periods}"
                    )
                    
                    # Remove from tracking
                    del holding_periods[asset_id]
            
            # Update for next iteration
            last_position = current_position
    
    return result


# ===============================================================================
# Drawdown Protection Functions
# ===============================================================================

def apply_drawdown_guard(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    max_drawdown_pct: float,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    equity_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply drawdown protection to reduce positions during drawdowns.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        max_drawdown_pct: Maximum allowed drawdown percentage
        position_column: Column containing position size
        weight_column: Column to store target weight
        equity_column: Optional column containing equity curve
        
    Returns:
        DataFrame with drawdown protection applied
        
    Example:
        >>> positions_df = apply_drawdown_guard(positions_df, prices_df, max_drawdown_pct=10)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply drawdown guard: '{position_column}' column missing")
        return positions_df
    
    # If equity column is provided, use it directly
    if equity_column is not None and equity_column in result.columns:
        # Calculate drawdown from equity curve
        result['peak'] = result[equity_column].cummax()
        result['drawdown_pct'] = (result[equity_column] / result['peak'] - 1) * 100
    else:
        # Otherwise, calculate returns from price data
        if 'timestamp' not in result.columns or 'asset_id' not in result.columns:
            logger.warning("Cannot calculate returns: missing timestamp or asset_id")
            return positions_df
        
        # Calculate returns for each asset
        returns_by_asset = {}
        
        for asset_id in result['asset_id'].unique():
            # Get price data for this asset
            asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
            
            if len(asset_prices) < 2:
                continue
            
            # Calculate returns
            asset_prices['return'] = asset_prices['close'].pct_change()
            
            # Calculate cumulative returns
            asset_prices['cum_return'] = (1 + asset_prices['return']).cumprod()
            
            # Calculate drawdown
            asset_prices['peak'] = asset_prices['cum_return'].cummax()
            asset_prices['drawdown_pct'] = (asset_prices['cum_return'] / asset_prices['peak'] - 1) * 100
            
            # Store for use
            returns_by_asset[asset_id] = asset_prices
        
        # Merge drawdown data with positions
        for asset_id, asset_returns in returns_by_asset.items():
            # Get positions for this asset
            asset_positions = result[result['asset_id'] == asset_id]
            
            # Skip if no positions for this asset
            if asset_positions.empty:
                continue
            
            # Merge drawdown data
            for i, row in asset_positions.iterrows():
                position_time = row['timestamp']
                
                # Get most recent drawdown before this position
                past_data = asset_returns[asset_returns['timestamp'] <= position_time]
                
                if not past_data.empty:
                    # Get current drawdown
                    current_drawdown = past_data.iloc[-1]['drawdown_pct']
                    
                    # Store in result
                    result.loc[i, 'drawdown_pct'] = current_drawdown
    
    # Apply drawdown scaling to positions
    if 'drawdown_pct' in result.columns:
        for i, row in result.iterrows():
            # Get current drawdown
            current_drawdown = abs(row['drawdown_pct'])
            
            if pd.isna(current_drawdown):
                continue
            
            # Check if drawdown exceeds maximum
            if current_drawdown > max_drawdown_pct:
                # Close position completely if significantly beyond max
                if current_drawdown > max_drawdown_pct * 1.5:
                    result.loc[i, position_column] = 0
                    
                    if weight_column in result.columns:
                        result.loc[i, weight_column] = 0
                    
                    logger.info(
                        f"Full drawdown stop for {row['asset_id']} at {row['timestamp']}, "
                        f"drawdown={current_drawdown:.2f}%, max={max_drawdown_pct:.2f}%"
                    )
                else:
                    # Scale down position based on drawdown severity
                    excess_drawdown = current_drawdown - max_drawdown_pct
                    max_excess = max_drawdown_pct * 0.5  # Complete reduction at 1.5x max
                    
                    # Calculate scaling factor (1.0 at max_drawdown, 0.0 at 1.5x max_drawdown)
                    scale_factor = 1.0 - (excess_drawdown / max_excess)
                    scale_factor = max(0.0, min(1.0, scale_factor))
                    
                    # Apply scaling
                    result.loc[i, position_column] = row[position_column] * scale_factor
                    
                    if weight_column in result.columns:
                        result.loc[i, weight_column] = row[weight_column] * scale_factor
                    
                    logger.info(
                        f"Partial drawdown reduction for {row['asset_id']} at {row['timestamp']}, "
                        f"drawdown={current_drawdown:.2f}%, scale={scale_factor:.2f}"
                    )
    
    # Clean up temporary columns
    if 'drawdown_pct' in result.columns:
        result.drop('drawdown_pct', axis=1, inplace=True)
    
    if 'peak' in result.columns:
        result.drop('peak', axis=1, inplace=True)
    
    return result


def apply_trend_filter(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    ma_periods: int = 200,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    long_only_above_ma: bool = True,
    short_only_below_ma: bool = True
) -> pd.DataFrame:
    """
    Apply moving average trend filter to control position direction.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        ma_periods: Number of periods for moving average calculation
        position_column: Column containing position size
        weight_column: Column to store target weight
        long_only_above_ma: Only allow long positions when price is above MA
        short_only_below_ma: Only allow short positions when price is below MA
        
    Returns:
        DataFrame with trend filter applied
        
    Example:
        >>> positions_df = apply_trend_filter(positions_df, prices_df, ma_periods=50)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply trend filter: '{position_column}' column missing")
        return positions_df
    
    # Verify price data
    required_cols = ['timestamp', 'asset_id', 'close']
    missing_cols = [col for col in required_cols if col not in prices_df.columns]
    
    if missing_cols:
        logger.warning(f"Price data missing required columns for MA calculation: {missing_cols}")
        return positions_df
    
    # Calculate moving average for each asset
    ma_by_asset = {}
    
    for asset_id in result['asset_id'].unique():
        # Get price data for this asset
        asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
        
        if len(asset_prices) < ma_periods:
            logger.warning(f"Not enough price history for {asset_id} MA calculation")
            continue
        
        # Calculate moving average
        asset_prices['ma'] = asset_prices['close'].rolling(window=ma_periods).mean()
        
        # Store for use
        ma_by_asset[asset_id] = asset_prices
    
    # Apply trend filter to positions
    for i, row in result.iterrows():
        asset_id = row['asset_id']
        
        # Skip if no MA data for this asset
        if asset_id not in ma_by_asset:
            continue
        
        # Get current position direction
        position_direction = np.sign(row[position_column])
        
        if position_direction == 0:
            continue
        
        # Get MA data
        asset_ma = ma_by_asset[asset_id]
        
        # Get timestamp for this position
        position_timestamp = row['timestamp']
        
        # Get most recent MA before this position
        past_ma = asset_ma[asset_ma['timestamp'] <= position_timestamp]
        
        if past_ma.empty or pd.isna(past_ma.iloc[-1]['ma']):
            continue
        
        # Get current price and MA
        current_price = past_ma.iloc[-1]['close']
        current_ma = past_ma.iloc[-1]['ma']
        
        # Check trend conditions
        above_ma = current_price > current_ma
        
        # Apply filters
        if position_direction > 0 and not above_ma and long_only_above_ma:
            # Long position but price below MA - close position
            result.loc[i, position_column] = 0
            
            if weight_column in result.columns:
                result.loc[i, weight_column] = 0
            
            logger.info(
                f"Trend filter closed long position for {asset_id} at {position_timestamp}, "
                f"price={current_price:.4f}, MA={current_ma:.4f}"
            )
        
        elif position_direction < 0 and above_ma and short_only_below_ma:
            # Short position but price above MA - close position
            result.loc[i, position_column] = 0
            
            if weight_column in result.columns:
                result.loc[i, weight_column] = 0
            
            logger.info(
                f"Trend filter closed short position for {asset_id} at {position_timestamp}, "
                f"price={current_price:.4f}, MA={current_ma:.4f}"
            )
    
    return result


# ===============================================================================
# Portfolio-Level Risk Functions
# ===============================================================================

def apply_sector_limits(
    positions_df: pd.DataFrame,
    sector_mappings: Dict[str, str],
    max_sector_exposure: Dict[str, float],
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    default_limit: float = 0.25
) -> pd.DataFrame:
    """
    Apply sector exposure limits to control concentration risk.
    
    Args:
        positions_df: DataFrame with positions
        sector_mappings: Dictionary mapping asset_id to sector
        max_sector_exposure: Dictionary mapping sector to maximum exposure
        position_column: Column containing position size
        weight_column: Column to store target weight
        default_limit: Default sector limit if not specified
        
    Returns:
        DataFrame with sector limits applied
        
    Example:
        >>> sector_mappings = {'AAPL': 'Technology', 'MSFT': 'Technology', 'XOM': 'Energy'}
        >>> max_sector_exposure = {'Technology': 0.25, 'Energy': 0.15}
        >>> positions_df = apply_sector_limits(positions_df, sector_mappings, max_sector_exposure)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply sector limits: '{position_column}' column missing")
        return positions_df
    
    # Add sector information to positions
    result['sector'] = result['asset_id'].map(sector_mappings)
    
    # Skip assets with no sector mapping
    result.loc[result['sector'].isna(), 'sector'] = 'Unknown'
    
    # Group positions by timestamp and sector
    positions_by_time = result.sort_values('timestamp')
    
    for timestamp, time_positions in positions_by_time.groupby('timestamp'):
        # Calculate sector exposures
        sector_exposures = {}
        
        for sector, sector_positions in time_positions.groupby('sector'):
            # Sum absolute weights for sector
            if weight_column in sector_positions.columns:
                sector_exposures[sector] = sector_positions[weight_column].abs().sum()
            else:
                # If no weight column, use normalized position sizes
                total_pos = time_positions[position_column].abs().sum()
                if total_pos > 0:
                    sector_exposures[sector] = sector_positions[position_column].abs().sum() / total_pos
                else:
                    sector_exposures[sector] = 0
        
        # Check sector limits
        for sector, exposure in sector_exposures.items():
            # Get limit for this sector
            limit = max_sector_exposure.get(sector, default_limit)
            
            # Skip if within limit
            if exposure <= limit:
                continue
            
            # Scale down positions in this sector
            scale_factor = limit / exposure
            
            # Apply scaling
            sector_mask = (time_positions['sector'] == sector) & (time_positions['timestamp'] == timestamp)
            sector_indices = result[sector_mask].index
            
            result.loc[sector_indices, position_column] = result.loc[sector_indices, position_column] * scale_factor
            
            if weight_column in result.columns:
                result.loc[sector_indices, weight_column] = result.loc[sector_indices, weight_column] * scale_factor
            
            logger.info(
                f"Scaled down {sector} exposure at {timestamp} from {exposure:.2f} to {limit:.2f}"
            )
    
    # Clean up temporary columns
    if 'sector' in result.columns:
        result.drop('sector', axis=1, inplace=True)
    
    return result


def apply_correlation_scaling(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    lookback: int = 60,
    max_correlation: float = 0.7,
    position_column: str = 'position',
    weight_column: str = 'target_weight',
    min_periods: int = 20
) -> pd.DataFrame:
    """
    Scale down positions of highly correlated assets.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        lookback: Number of periods for correlation calculation
        max_correlation: Maximum allowed correlation
        position_column: Column containing position size
        weight_column: Column to store target weight
        min_periods: Minimum periods required for correlation calculation
        
    Returns:
        DataFrame with correlation scaling applied
        
    Example:
        >>> positions_df = apply_correlation_scaling(positions_df, prices_df, max_correlation=0.6)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply correlation scaling: '{position_column}' column missing")
        return positions_df
    
    # Verify price data
    required_cols = ['timestamp', 'asset_id', 'close']
    missing_cols = [col for col in required_cols if col not in prices_df.columns]
    
    if missing_cols:
        logger.warning(f"Price data missing required columns for correlation: {missing_cols}")
        return positions_df
    
    # Process positions by timestamp
    positions_by_time = result.sort_values('timestamp')
    
    for timestamp, time_positions in positions_by_time.groupby('timestamp'):
        # Skip if less than 2 assets at this timestamp
        if len(time_positions) < 2:
            continue
        
        # Get active assets
        active_assets = time_positions[time_positions[position_column] != 0]['asset_id'].unique()
        
        # Skip if less than 2 active positions
        if len(active_assets) < 2:
            continue
        
        # Prepare returns data for correlation calculation
        returns_data = {}
        
        for asset_id in active_assets:
            # Get price data for this asset
            asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
            
            # Skip if not enough data
            if len(asset_prices) < min_periods:
                continue
            
            # Get data up to current timestamp
            past_prices = asset_prices[asset_prices['timestamp'] <= timestamp].tail(lookback + 1)
            
            # Calculate returns
            if len(past_prices) >= min_periods:
                past_prices['return'] = past_prices['close'].pct_change()
                returns_data[asset_id] = past_prices['return'].dropna().values
        
        # Skip if less than 2 assets with return data
        if len(returns_data) < 2:
            continue
        
        # Calculate correlation matrix
        asset_list = list(returns_data.keys())
        returns_matrix = pd.DataFrame({asset: returns_data[asset] for asset in asset_list})
        corr_matrix = returns_matrix.corr()
        
        # Find highly correlated pairs
        for i in range(len(asset_list)):
            for j in range(i+1, len(asset_list)):
                asset1 = asset_list[i]
                asset2 = asset_list[j]
                correlation = corr_matrix.loc[asset1, asset2]
                
                if abs(correlation) > max_correlation:
                    # Scale down positions based on correlation excess
                    scale_factor = max_correlation / abs(correlation)
                    
                    # Get position indices
                    asset1_indices = result[(result['asset_id'] == asset1) & (result['timestamp'] == timestamp)].index
                    asset2_indices = result[(result['asset_id'] == asset2) & (result['timestamp'] == timestamp)].index
                    
                    # Scale down both positions
                    result.loc[asset1_indices, position_column] = result.loc[asset1_indices, position_column] * scale_factor
                    result.loc[asset2_indices, position_column] = result.loc[asset2_indices, position_column] * scale_factor
                    
                    if weight_column in result.columns:
                        result.loc[asset1_indices, weight_column] = result.loc[asset1_indices, weight_column] * scale_factor
                        result.loc[asset2_indices, weight_column] = result.loc[asset2_indices, weight_column] * scale_factor
                    
                    logger.info(
                        f"Scaled down {asset1} and {asset2} positions due to high correlation ({correlation:.2f}) at {timestamp}"
                    )
    
    return result


def calculate_portfolio_var(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    confidence_level: float = 0.95,
    lookback: int = 252,
    method: str = 'historical',
    position_column: str = 'position',
    min_periods: int = 60
) -> pd.DataFrame:
    """
    Calculate Value at Risk (VaR) for portfolio positions.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        lookback: Number of periods for return calculation
        method: VaR calculation method ('historical', 'parametric')
        position_column: Column containing position size
        min_periods: Minimum periods required for calculation
        
    Returns:
        DataFrame with VaR metrics added
        
    Example:
        >>> var_df = calculate_portfolio_var(positions_df, prices_df, confidence_level=0.99)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot calculate VaR: '{position_column}' column missing")
        return positions_df
    
    # Verify price data
    required_cols = ['timestamp', 'asset_id', 'close']
    missing_cols = [col for col in required_cols if col not in prices_df.columns]
    
    if missing_cols:
        logger.warning(f"Price data missing required columns for VaR: {missing_cols}")
        return positions_df
    
    # Process positions by timestamp
    positions_by_time = result.sort_values('timestamp')
    
    for timestamp, time_positions in positions_by_time.groupby('timestamp'):
        # Skip if no active positions
        if (time_positions[position_column] == 0).all():
            continue
        
        # Get active assets
        active_assets = time_positions[time_positions[position_column] != 0]['asset_id'].unique()
        
        # Prepare returns data for VaR calculation
        returns_data = {}
        position_values = {}
        
        for asset_id in active_assets:
            # Get price data for this asset
            asset_prices = prices_df[prices_df['asset_id'] == asset_id].sort_values('timestamp')
            
            # Skip if not enough data
            if len(asset_prices) < min_periods:
                continue
            
            # Get data up to current timestamp
            past_prices = asset_prices[asset_prices['timestamp'] <= timestamp].tail(lookback + 1)
            
            # Calculate returns
            if len(past_prices) >= min_periods:
                past_prices['return'] = past_prices['close'].pct_change()
                returns_data[asset_id] = past_prices['return'].dropna().values
                
                # Get position value
                asset_position = time_positions.loc[time_positions['asset_id'] == asset_id, position_column].iloc[0]
                asset_price = past_prices['close'].iloc[-1]
                position_values[asset_id] = asset_position * asset_price
        
        # Skip if not enough assets with return data
        if len(returns_data) < 1:
            continue
        
        # Calculate VaR
        if method == 'historical':
            # Create weighted portfolio returns
            total_value = sum(abs(val) for val in position_values.values())
            
            if total_value > 0:
                portfolio_returns = np.zeros(len(next(iter(returns_data.values()))))
                
                for asset_id, returns in returns_data.items():
                    weight = position_values[asset_id] / total_value
                    # Apply sign based on position direction
                    signed_weight = weight if position_values[asset_id] > 0 else -weight
                    portfolio_returns += returns * signed_weight
                
                # Calculate historical VaR
                var_percentile = 1 - confidence_level
                var = -np.percentile(portfolio_returns, var_percentile * 100)
                cvar = -portfolio_returns[portfolio_returns <= -var].mean()
                
                # Store results
                for i in time_positions.index:
                    result.loc[i, 'portfolio_var'] = var
                    result.loc[i, 'portfolio_cvar'] = cvar
                    result.loc[i, 'var_pct_of_portfolio'] = var * 100
                    result.loc[i, 'cvar_pct_of_portfolio'] = cvar * 100
                
                logger.info(
                    f"Portfolio VaR at {timestamp}: {var:.4f} ({var*100:.2f}%), "
                    f"CVaR: {cvar:.4f} ({cvar*100:.2f}%)"
                )
        
        elif method == 'parametric':
            # Create returns matrix
            asset_list = list(returns_data.keys())
            returns_matrix = pd.DataFrame({asset: returns_data[asset] for asset in asset_list})
            
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov()
            
            # Create position vector
            position_vector = pd.Series([position_values.get(asset, 0) for asset in asset_list], index=asset_list)
            
            # Calculate parametric VaR
            portfolio_variance = position_vector.dot(cov_matrix).dot(position_vector)
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # VaR calculation using normal distribution assumption
            from scipy import stats
            z_score = stats.norm.ppf(confidence_level)
            var = z_score * portfolio_vol
            
            # Store results
            for i in time_positions.index:
                result.loc[i, 'portfolio_var'] = var
                result.loc[i, 'portfolio_vol'] = portfolio_vol
                result.loc[i, 'var_pct_of_portfolio'] = var * 100
            
            logger.info(
                f"Parametric Portfolio VaR at {timestamp}: {var:.4f} ({var*100:.2f}%), "
                f"Volatility: {portfolio_vol:.4f}"
            )
    
    return result