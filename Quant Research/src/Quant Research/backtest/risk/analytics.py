"""
Risk analytics functions for backtesting.

This module provides implementations of various risk metrics and analytics including:
- Value at Risk (VaR) calculation
- Conditional VaR (CVaR) estimation
- Portfolio risk metrics
- Stress testing and scenario analysis
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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


def calculate_risk_contributions(
    positions_df: pd.DataFrame, 
    prices_df: pd.DataFrame,
    lookback: int = 60,
    position_column: str = 'position',
    min_periods: int = 20
) -> pd.DataFrame:
    """
    Calculate risk contributions of individual positions to portfolio risk.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price data
        lookback: Number of periods for risk calculation
        position_column: Column containing position size
        min_periods: Minimum periods required for calculation
        
    Returns:
        DataFrame with risk contribution metrics added
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot calculate risk contributions: '{position_column}' column missing")
        return positions_df
    
    # Verify price data
    required_cols = ['timestamp', 'asset_id', 'close']
    missing_cols = [col for col in required_cols if col not in prices_df.columns]
    
    if missing_cols:
        logger.warning(f"Price data missing required columns for risk calculation: {missing_cols}")
        return positions_df
    
    # Process positions by timestamp
    positions_by_time = result.sort_values('timestamp')
    
    for timestamp, time_positions in positions_by_time.groupby('timestamp'):
        # Skip if no active positions
        if (time_positions[position_column] == 0).all():
            continue
        
        # Get active assets
        active_assets = time_positions[time_positions[position_column] != 0]['asset_id'].unique()
        
        if len(active_assets) < 2:
            # Need at least 2 assets for portfolio risk analysis
            continue
        
        # Prepare returns data and position values
        returns_by_asset = {}
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
                returns_by_asset[asset_id] = past_prices['return'].dropna()
                
                # Get position value
                asset_position = time_positions.loc[time_positions['asset_id'] == asset_id, position_column].iloc[0]
                asset_price = past_prices['close'].iloc[-1]
                position_values[asset_id] = asset_position * asset_price
        
        # Skip if not enough assets with return data
        if len(returns_by_asset) < 2:
            continue
        
        # Create returns matrix and position vector
        common_index = pd.concat(returns_by_asset.values()).index.unique()
        returns_matrix = pd.DataFrame(index=common_index)
        
        for asset_id, returns in returns_by_asset.items():
            returns_matrix[asset_id] = returns
        
        # Fill missing values (if any)
        returns_matrix = returns_matrix.fillna(0)
        
        # Calculate covariance matrix
        cov_matrix = returns_matrix.cov()
        
        # Create position vector
        assets = list(position_values.keys())
        position_vector = pd.Series([position_values.get(asset, 0) for asset in assets], index=assets)
        
        # Calculate portfolio variance and marginal contributions
        portfolio_variance = position_vector.dot(cov_matrix).dot(position_vector)
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Calculate marginal contributions
        marginal_contributions = cov_matrix.dot(position_vector)
        
        # Calculate component contributions
        component_contributions = position_vector * marginal_contributions
        
        # Calculate percentage contributions
        total_contribution = component_contributions.sum()
        percentage_contributions = component_contributions / total_contribution if total_contribution != 0 else pd.Series(0, index=assets)
        
        # Store results in the positions DataFrame
        for asset_id in assets:
            asset_indices = time_positions[time_positions['asset_id'] == asset_id].index
            
            if not asset_indices.empty:
                pct_contribution = percentage_contributions.get(asset_id, 0) * 100
                result.loc[asset_indices, 'risk_contribution_pct'] = pct_contribution
                
                # Also store marginal contribution
                marginal = marginal_contributions.get(asset_id, 0)
                result.loc[asset_indices, 'marginal_risk_contribution'] = marginal
        
        # Store portfolio volatility for reference
        result.loc[time_positions.index, 'portfolio_volatility'] = portfolio_vol
    
    return result


def calculate_stress_test(
    positions_df: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]],
    position_column: str = 'position',
    price_column: str = 'price'
) -> pd.DataFrame:
    """
    Perform stress testing on portfolio positions under different scenarios.
    
    Args:
        positions_df: DataFrame with positions
        scenarios: Dictionary of scenarios, each containing asset_id to price change mappings
        position_column: Column containing position size
        price_column: Column containing current price
        
    Returns:
        DataFrame with stress test results
    """
    if positions_df.empty:
        return pd.DataFrame()
    
    # Verify required columns
    required_cols = ['timestamp', 'asset_id', position_column, price_column]
    missing_cols = [col for col in required_cols if col not in positions_df.columns]
    
    if missing_cols:
        logger.warning(f"Cannot perform stress test: missing columns {missing_cols}")
        return pd.DataFrame()
    
    # Group positions by timestamp
    results = []
    
    for timestamp, time_positions in positions_df.groupby('timestamp'):
        # Calculate base portfolio value
        base_value = 0
        for _, row in time_positions.iterrows():
            position = row[position_column]
            price = row[price_column]
            base_value += position * price
        
        # Apply each scenario
        for scenario_name, price_changes in scenarios.items():
            scenario_value = 0
            
            # Calculate value under this scenario
            for _, row in time_positions.iterrows():
                asset_id = row['asset_id']
                position = row[position_column]
                current_price = row[price_column]
                
                # Apply price change for this asset (default to 0% if not specified)
                price_change_pct = price_changes.get(asset_id, 0) / 100
                scenario_price = current_price * (1 + price_change_pct)
                
                # Add to scenario value
                scenario_value += position * scenario_price
            
            # Calculate change and percentage change
            value_change = scenario_value - base_value
            pct_change = (value_change / base_value) * 100 if base_value != 0 else 0
            
            # Store result
            results.append({
                'timestamp': timestamp,
                'scenario': scenario_name,
                'base_value': base_value,
                'scenario_value': scenario_value,
                'value_change': value_change,
                'pct_change': pct_change
            })
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)