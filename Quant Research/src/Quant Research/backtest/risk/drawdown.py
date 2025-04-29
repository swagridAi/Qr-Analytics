"""
Drawdown protection functions for backtesting.

This module provides implementations of drawdown protection mechanisms including:
- Drawdown-based position scaling
- Trend filtering
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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