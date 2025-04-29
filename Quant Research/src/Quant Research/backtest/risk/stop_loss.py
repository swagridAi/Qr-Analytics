"""
Stop-loss functions for backtesting.

This module provides implementations of various stop-loss mechanisms including:
- Fixed percentage stop-loss
- Volatility-based (ATR) stops
- Time-based stops
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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
        holding_periods = 0
        
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