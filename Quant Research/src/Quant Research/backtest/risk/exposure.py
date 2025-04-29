"""
Exposure control functions for backtesting.

This module provides implementations of exposure control mechanisms including:
- Position limit controls
- Sector exposure limits
- Correlation-based position scaling
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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