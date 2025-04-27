"""
Fee and execution models for backtesting.

This module provides implementations of various fee models, slippage models,
and execution cost calculations for realistic backtest simulations.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from quant_research.core.models import Trade

logger = logging.getLogger(__name__)


# ===============================================================================
# Commission Models
# ===============================================================================

def apply_fixed_commission(
    positions_df: pd.DataFrame, 
    commission_pct: float = 0.001, 
    min_commission: float = 1.0,
    notional_column: str = 'notional_value',
    fee_column: str = 'commission'
) -> pd.DataFrame:
    """
    Apply a fixed percentage commission with minimum fee.
    
    Args:
        positions_df: DataFrame with positions
        commission_pct: Commission percentage (e.g., 0.001 for 0.1%)
        min_commission: Minimum commission per trade
        notional_column: Column containing notional value of position
        fee_column: Column to store commission amount
        
    Returns:
        DataFrame with commissions applied
        
    Example:
        >>> positions_df = apply_fixed_commission(positions_df, commission_pct=0.00075, min_commission=1.0)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify notional value column exists
    if notional_column not in result.columns:
        # Try to calculate it if 'position' and 'price' columns exist
        if 'position' in result.columns and 'price' in result.columns:
            result[notional_column] = result['position'].abs() * result['price']
        elif 'position' in result.columns and 'close' in result.columns:
            result[notional_column] = result['position'].abs() * result['close']
        else:
            logger.warning(f"Cannot calculate commission: '{notional_column}' column missing")
            return positions_df
    
    # Calculate commission as percentage of notional value
    result[fee_column] = result[notional_column].abs() * commission_pct
    
    # Apply minimum commission
    if min_commission > 0:
        result[fee_column] = result[fee_column].clip(lower=min_commission)
    
    # Calculate net notional (adjusted for commission)
    if 'net_notional_value' not in result.columns:
        # For buys (positive position), add commission to cost
        # For sells (negative position), subtract commission from proceeds
        buy_mask = result['position'] > 0
        sell_mask = result['position'] < 0
        
        result['net_notional_value'] = result[notional_column].copy()
        
        if 'net_notional_value' in result.columns:
            result.loc[buy_mask, 'net_notional_value'] = result.loc[buy_mask, notional_column] + result.loc[buy_mask, fee_column]
            result.loc[sell_mask, 'net_notional_value'] = result.loc[sell_mask, notional_column] - result.loc[sell_mask, fee_column]
    
    return result


def apply_tiered_commission(
    positions_df: pd.DataFrame, 
    tiers: List[Dict[str, Union[float, int]]], 
    notional_column: str = 'notional_value',
    fee_column: str = 'commission'
) -> pd.DataFrame:
    """
    Apply a tiered commission schedule based on trade size.
    
    Args:
        positions_df: DataFrame with positions
        tiers: List of tier dictionaries, each with 'threshold' and 'rate' keys
        notional_column: Column containing notional value of position
        fee_column: Column to store commission amount
        
    Returns:
        DataFrame with commissions applied
        
    Example:
        >>> tiers = [
        >>>     {'threshold': 0, 'rate': 0.0020},     # 0.20% for trades up to 10k
        >>>     {'threshold': 10000, 'rate': 0.0015}, # 0.15% for trades 10k-50k
        >>>     {'threshold': 50000, 'rate': 0.0010}  # 0.10% for trades above 50k
        >>> ]
        >>> positions_df = apply_tiered_commission(positions_df, tiers)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify notional value column exists
    if notional_column not in result.columns:
        # Try to calculate it if 'position' and 'price' columns exist
        if 'position' in result.columns and 'price' in result.columns:
            result[notional_column] = result['position'].abs() * result['price']
        elif 'position' in result.columns and 'close' in result.columns:
            result[notional_column] = result['position'].abs() * result['close']
        else:
            logger.warning(f"Cannot calculate commission: '{notional_column}' column missing")
            return positions_df
    
    # Sort tiers by threshold (ascending)
    sorted_tiers = sorted(tiers, key=lambda x: x['threshold'])
    
    # Initialize commission column
    result[fee_column] = 0.0
    
    # Apply tiered commission rates
    for i, row in result.iterrows():
        notional = abs(row[notional_column])
        
        # Find applicable tier
        applicable_tier = sorted_tiers[0]
        for tier in sorted_tiers:
            if notional >= tier['threshold']:
                applicable_tier = tier
            else:
                break
        
        # Apply rate
        result.loc[i, fee_column] = notional * applicable_tier['rate']
    
    # Calculate net notional (adjusted for commission)
    if 'net_notional_value' not in result.columns:
        # For buys (positive position), add commission to cost
        # For sells (negative position), subtract commission from proceeds
        buy_mask = result['position'] > 0
        sell_mask = result['position'] < 0
        
        result['net_notional_value'] = result[notional_column].copy()
        
        if 'net_notional_value' in result.columns:
            result.loc[buy_mask, 'net_notional_value'] = result.loc[buy_mask, notional_column] + result.loc[buy_mask, fee_column]
            result.loc[sell_mask, 'net_notional_value'] = result.loc[sell_mask, notional_column] - result.loc[sell_mask, fee_column]
    
    return result


def apply_exchange_fees(
    positions_df: pd.DataFrame, 
    exchange_fees: Dict[str, float], 
    exchange_column: str = 'exchange_id',
    notional_column: str = 'notional_value',
    fee_column: str = 'commission',
    default_fee: float = 0.001
) -> pd.DataFrame:
    """
    Apply exchange-specific fee rates.
    
    Args:
        positions_df: DataFrame with positions
        exchange_fees: Dictionary mapping exchange IDs to fee rates
        exchange_column: Column containing exchange identifier
        notional_column: Column containing notional value of position
        fee_column: Column to store commission amount
        default_fee: Default fee rate for unknown exchanges
        
    Returns:
        DataFrame with commissions applied
        
    Example:
        >>> exchange_fees = {
        >>>     'binance': 0.001,
        >>>     'coinbase': 0.0015,
        >>>     'kraken': 0.0026
        >>> }
        >>> positions_df = apply_exchange_fees(positions_df, exchange_fees)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify required columns exist
    if exchange_column not in result.columns:
        logger.warning(f"Cannot apply exchange fees: '{exchange_column}' column missing")
        return positions_df
    
    # Verify notional value column exists
    if notional_column not in result.columns:
        # Try to calculate it if 'position' and 'price' columns exist
        if 'position' in result.columns and 'price' in result.columns:
            result[notional_column] = result['position'].abs() * result['price']
        elif 'position' in result.columns and 'close' in result.columns:
            result[notional_column] = result['position'].abs() * result['close']
        else:
            logger.warning(f"Cannot calculate commission: '{notional_column}' column missing")
            return positions_df
    
    # Apply exchange-specific fees
    result[fee_column] = 0.0
    
    for exchange, fee_rate in exchange_fees.items():
        mask = result[exchange_column] == exchange
        result.loc[mask, fee_column] = result.loc[mask, notional_column].abs() * fee_rate
    
    # Apply default fee for exchanges not in the dictionary
    default_mask = ~result[exchange_column].isin(exchange_fees.keys())
    result.loc[default_mask, fee_column] = result.loc[default_mask, notional_column].abs() * default_fee
    
    # Calculate net notional (adjusted for commission)
    if 'net_notional_value' not in result.columns:
        # For buys (positive position), add commission to cost
        # For sells (negative position), subtract commission from proceeds
        buy_mask = result['position'] > 0
        sell_mask = result['position'] < 0
        
        result['net_notional_value'] = result[notional_column].copy()
        
        if 'net_notional_value' in result.columns:
            result.loc[buy_mask, 'net_notional_value'] = result.loc[buy_mask, notional_column] + result.loc[buy_mask, fee_column]
            result.loc[sell_mask, 'net_notional_value'] = result.loc[sell_mask, notional_column] - result.loc[sell_mask, fee_column]
    
    return result


# ===============================================================================
# Slippage Models
# ===============================================================================

def apply_simple_slippage(
    positions_df: pd.DataFrame, 
    slippage_bps: float = 5,
    position_column: str = 'position',
    price_column: str = 'price',
    slippage_column: str = 'slippage'
) -> pd.DataFrame:
    """
    Apply a simple fixed basis points slippage model.
    
    Args:
        positions_df: DataFrame with positions
        slippage_bps: Slippage in basis points (e.g., 5 for 0.05%)
        position_column: Column containing position size
        price_column: Column containing price
        slippage_column: Column to store slippage amount
        
    Returns:
        DataFrame with slippage applied
        
    Example:
        >>> positions_df = apply_simple_slippage(positions_df, slippage_bps=3)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply slippage: '{position_column}' column missing")
        return positions_df
    
    # Find best price column to use
    if price_column not in result.columns:
        # Try alternative price columns
        alternative_columns = ['close', 'mid_price', 'execution_price']
        for col in alternative_columns:
            if col in result.columns:
                price_column = col
                break
        else:
            logger.warning(f"Cannot apply slippage: no price column found")
            return positions_df
    
    # Calculate slippage amount based on position direction
    slippage_factor = slippage_bps / 10000  # Convert bps to decimal
    
    # For buys (positive position), price is increased
    # For sells (negative position), price is decreased
    buy_mask = result[position_column] > 0
    sell_mask = result[position_column] < 0
    
    # Calculate slippage amount
    result[slippage_column] = 0.0
    result.loc[buy_mask, slippage_column] = result.loc[buy_mask, price_column] * slippage_factor
    result.loc[sell_mask, slippage_column] = -result.loc[sell_mask, price_column] * slippage_factor
    
    # Calculate execution price with slippage
    result['execution_price'] = result[price_column] + result[slippage_column]
    
    # Recalculate notional value with slippage
    if 'notional_value' in result.columns:
        # Update notional value based on new execution price
        result['notional_value'] = result[position_column].abs() * result['execution_price']
    
    return result


def apply_market_impact(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    impact_factor: float = 0.1,
    position_column: str = 'position',
    volume_column: str = 'volume',
    price_column: str = 'price',
    impact_column: str = 'market_impact'
) -> pd.DataFrame:
    """
    Apply a market impact model based on position size relative to volume.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price and volume data
        impact_factor: Factor to scale market impact (higher = more impact)
        position_column: Column containing position size
        volume_column: Column containing volume data
        price_column: Column containing price
        impact_column: Column to store market impact amount
        
    Returns:
        DataFrame with market impact applied
        
    Example:
        >>> positions_df = apply_market_impact(positions_df, prices_df, impact_factor=0.2)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply market impact: '{position_column}' column missing")
        return positions_df
    
    # Find best price column to use
    if price_column not in result.columns:
        # Try alternative price columns
        alternative_columns = ['close', 'mid_price', 'execution_price']
        for col in alternative_columns:
            if col in result.columns:
                price_column = col
                break
        else:
            logger.warning(f"Cannot apply market impact: no price column found")
            return positions_df
    
    # Merge with price data to get volume
    if volume_column not in result.columns:
        if volume_column not in prices_df.columns:
            logger.warning(f"Cannot apply market impact: '{volume_column}' column missing from price data")
            return positions_df
        
        # Merge on timestamp and asset_id
        result = pd.merge(
            result,
            prices_df[['timestamp', 'asset_id', volume_column]],
            on=['timestamp', 'asset_id'],
            how='left'
        )
    
    # Calculate participation rate (position size / volume)
    result['participation_rate'] = result[position_column].abs() / result[volume_column]
    
    # Cap participation rate to avoid extreme values
    result['participation_rate'] = result['participation_rate'].clip(upper=0.5)
    
    # Calculate market impact
    # sqrt model: impact ~ sqrt(participation_rate)
    result[impact_column] = impact_factor * result[price_column] * np.sqrt(result['participation_rate'])
    
    # Direction of impact depends on position direction
    buy_mask = result[position_column] > 0
    sell_mask = result[position_column] < 0
    
    # For buys, price is increased; for sells, price is decreased
    result.loc[sell_mask, impact_column] = -result.loc[sell_mask, impact_column]
    
    # Calculate execution price with market impact
    result['execution_price'] = result[price_column] + result[impact_column]
    
    # Recalculate notional value with market impact
    if 'notional_value' in result.columns:
        # Update notional value based on new execution price
        result['notional_value'] = result[position_column].abs() * result['execution_price']
    
    # Clean up temporary columns
    result.drop('participation_rate', axis=1, inplace=True)
    
    return result


def apply_probabilistic_slippage(
    positions_df: pd.DataFrame,
    slippage_std: float = 10,
    position_column: str = 'position',
    price_column: str = 'price',
    slippage_column: str = 'slippage',
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply a probabilistic slippage model based on a normal distribution.
    
    Args:
        positions_df: DataFrame with positions
        slippage_std: Standard deviation of slippage in basis points
        position_column: Column containing position size
        price_column: Column containing price
        slippage_column: Column to store slippage amount
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with probabilistic slippage applied
        
    Example:
        >>> positions_df = apply_probabilistic_slippage(positions_df, slippage_std=8, seed=42)
    """
    if positions_df.empty:
        return positions_df
    
    # Make a copy to avoid modifying the original
    result = positions_df.copy()
    
    # Verify position column exists
    if position_column not in result.columns:
        logger.warning(f"Cannot apply slippage: '{position_column}' column missing")
        return positions_df
    
    # Find best price column to use
    if price_column not in result.columns:
        # Try alternative price columns
        alternative_columns = ['close', 'mid_price', 'execution_price']
        for col in alternative_columns:
            if col in result.columns:
                price_column = col
                break
        else:
            logger.warning(f"Cannot apply slippage: no price column found")
            return positions_df
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random slippage from normal distribution
    n = len(result)
    random_bps = np.random.normal(0, slippage_std, n)
    
    # Convert bps to decimal
    random_factors = random_bps / 10000
    
    # Calculate slippage amount
    result[slippage_column] = result[price_column] * random_factors
    
    # Direction bias based on position direction:
    # Buys (positive position) tend to get slightly worse prices
    # Sells (negative position) tend to get slightly worse prices
    buy_mask = result[position_column] > 0
    sell_mask = result[position_column] < 0
    
    # Add a small directional bias
    bias_factor = slippage_std / 20000  # Small fraction of std
    result.loc[buy_mask, slippage_column] += result.loc[buy_mask, price_column] * bias_factor
    result.loc[sell_mask, slippage_column] -= result.loc[sell_mask, price_column] * bias_factor
    
    # Calculate execution price with slippage
    result['execution_price'] = result[price_column] + result[slippage_column]
    
    # Ensure execution price is positive
    result['execution_price'] = result['execution_price'].clip(lower=0.00001)
    
    # Recalculate notional value with slippage
    if 'notional_value' in result.columns:
        # Update notional value based on new execution price
        result['notional_value'] = result[position_column].abs() * result['execution_price']
    
    return result


# ===============================================================================
# Combined Execution Models
# ===============================================================================

def apply_execution_costs(
    positions_df: pd.DataFrame,
    prices_df: Optional[pd.DataFrame] = None,
    commission_model: str = 'fixed',
    commission_params: Dict[str, Any] = None,
    slippage_model: str = 'simple',
    slippage_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Apply both commission and slippage costs in a single step.
    
    Args:
        positions_df: DataFrame with positions
        prices_df: DataFrame with price and volume data (for some models)
        commission_model: Type of commission model ('fixed', 'tiered', 'exchange')
        commission_params: Parameters for commission model
        slippage_model: Type of slippage model ('simple', 'market_impact', 'probabilistic')
        slippage_params: Parameters for slippage model
        
    Returns:
        DataFrame with all execution costs applied
        
    Example:
        >>> positions_df = apply_execution_costs(
        >>>     positions_df, 
        >>>     prices_df,
        >>>     commission_model='fixed', 
        >>>     commission_params={'commission_pct': 0.00075, 'min_commission': 1.0},
        >>>     slippage_model='simple',
        >>>     slippage_params={'slippage_bps': 3}
        >>> )
    """
    if positions_df.empty:
        return positions_df
    
    # Default parameter dictionaries
    if commission_params is None:
        commission_params = {}
    
    if slippage_params is None:
        slippage_params = {}
    
    # Apply slippage first (affects price)
    if slippage_model == 'simple':
        result = apply_simple_slippage(positions_df, **slippage_params)
    elif slippage_model == 'market_impact':
        if prices_df is None:
            logger.warning("Market impact model requires prices_df, falling back to simple slippage")
            result = apply_simple_slippage(positions_df, **slippage_params)
        else:
            result = apply_market_impact(positions_df, prices_df, **slippage_params)
    elif slippage_model == 'probabilistic':
        result = apply_probabilistic_slippage(positions_df, **slippage_params)
    else:
        logger.warning(f"Unknown slippage model '{slippage_model}', no slippage applied")
        result = positions_df.copy()
    
    # Then apply commission (based on slippage-adjusted prices)
    if commission_model == 'fixed':
        result = apply_fixed_commission(result, **commission_params)
    elif commission_model == 'tiered':
        result = apply_tiered_commission(result, **commission_params)
    elif commission_model == 'exchange':
        result = apply_exchange_fees(result, **commission_params)
    else:
        logger.warning(f"Unknown commission model '{commission_model}', no commission applied")
    
    # Calculate total execution cost
    if 'slippage' in result.columns and 'commission' in result.columns:
        # For buys: total_cost = slippage + commission
        # For sells: total_cost = slippage - commission
        buy_mask = result['position'] > 0
        sell_mask = result['position'] < 0
        
        result['execution_cost'] = 0.0
        result.loc[buy_mask, 'execution_cost'] = (
            result.loc[buy_mask, 'slippage'] + 
            result.loc[buy_mask, 'commission']
        )
        result.loc[sell_mask, 'execution_cost'] = (
            result.loc[sell_mask, 'slippage'] - 
            result.loc[sell_mask, 'commission']
        )
        
        # Calculate cost as percentage of notional
        if 'notional_value' in result.columns:
            result['cost_pct'] = result['execution_cost'] / result['notional_value'].abs() * 100
    
    return result


# ===============================================================================
# Analysis Tools
# ===============================================================================

def calculate_transaction_costs(
    positions_df: pd.DataFrame, 
    group_by: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate summary of transaction costs.
    
    Args:
        positions_df: DataFrame with positions and execution costs
        group_by: Optional list of columns to group by for analysis
        
    Returns:
        DataFrame with cost summary
        
    Example:
        >>> costs_summary = calculate_transaction_costs(positions_df, group_by=['asset_id'])
    """
    if positions_df.empty:
        return pd.DataFrame()
    
    # Check required columns
    required_cols = ['position', 'notional_value']
    missing_cols = [col for col in required_cols if col not in positions_df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns for cost analysis: {missing_cols}")
        return pd.DataFrame()
    
    # Collect cost columns
    cost_columns = []
    for col in ['slippage', 'commission', 'execution_cost']:
        if col in positions_df.columns:
            cost_columns.append(col)
    
    if not cost_columns:
        logger.warning("No cost columns found in positions DataFrame")
        return pd.DataFrame()
    
    # Prepare data for analysis
    analysis_data = positions_df.copy()
    
    # Ensure notional value is absolute
    analysis_data['abs_notional'] = analysis_data['notional_value'].abs()
    
    # Define aggregation functions
    agg_funcs = {
        'position': ['count', 'sum'],
        'abs_notional': ['sum']
    }
    
    # Add cost columns to aggregation
    for col in cost_columns:
        agg_funcs[col] = ['sum', 'mean']
    
    # Group by specified columns or analyze as a whole
    if group_by:
        valid_group_cols = [col for col in group_by if col in analysis_data.columns]
        if valid_group_cols:
            result = analysis_data.groupby(valid_group_cols).agg(agg_funcs)
        else:
            logger.warning(f"None of the specified group_by columns {group_by} found in DataFrame")
            result = analysis_data.agg(agg_funcs)
    else:
        result = analysis_data.agg(agg_funcs)
    
    # Calculate cost metrics
    if 'slippage' in cost_columns:
        result[('slippage', 'pct_of_notional')] = result[('slippage', 'sum')] / result[('abs_notional', 'sum')] * 100
    
    if 'commission' in cost_columns:
        result[('commission', 'pct_of_notional')] = result[('commission', 'sum')] / result[('abs_notional', 'sum')] * 100
    
    if 'execution_cost' in cost_columns:
        result[('execution_cost', 'pct_of_notional')] = result[('execution_cost', 'sum')] / result[('abs_notional', 'sum')] * 100
    
    # Reset index for easier access
    result = result.reset_index()
    
    return result


def estimate_trading_costs(
    avg_position_size: float,
    num_trades: int,
    avg_price: float = 100.0,
    commission_pct: float = 0.001,
    slippage_bps: float = 5,
    market_impact_factor: float = 0.1,
    avg_participation_rate: float = 0.05
) -> Dict[str, float]:
    """
    Estimate expected trading costs for a strategy.
    
    Args:
        avg_position_size: Average position size in shares/contracts
        num_trades: Number of trades expected
        avg_price: Average price of the asset
        commission_pct: Commission rate as percentage
        slippage_bps: Average slippage in basis points
        market_impact_factor: Market impact factor
        avg_participation_rate: Average participation rate (position/volume)
        
    Returns:
        Dictionary with cost estimates
        
    Example:
        >>> cost_estimate = estimate_trading_costs(
        >>>     avg_position_size=1000,
        >>>     num_trades=500,
        >>>     avg_price=50,
        >>>     commission_pct=0.0005,
        >>>     slippage_bps=3
        >>> )
    """
    # Calculate notional value
    avg_notional = avg_position_size * avg_price
    total_notional = avg_notional * num_trades
    
    # Commission costs
    commission_cost_per_trade = avg_notional * commission_pct
    total_commission = commission_cost_per_trade * num_trades
    
    # Simple slippage costs
    slippage_factor = slippage_bps / 10000
    slippage_cost_per_trade = avg_notional * slippage_factor
    total_slippage = slippage_cost_per_trade * num_trades
    
    # Market impact costs
    impact_factor = market_impact_factor * avg_price * np.sqrt(avg_participation_rate)
    impact_cost_per_trade = avg_position_size * impact_factor
    total_impact = impact_cost_per_trade * num_trades
    
    # Total costs
    simple_total = total_commission + total_slippage
    full_total = total_commission + total_impact
    
    # Cost metrics
    result = {
        'avg_position_size': avg_position_size,
        'avg_notional': avg_notional,
        'num_trades': num_trades,
        'total_notional': total_notional,
        'commission_per_trade': commission_cost_per_trade,
        'total_commission': total_commission,
        'commission_pct_of_notional': (total_commission / total_notional) * 100,
        'slippage_per_trade': slippage_cost_per_trade,
        'total_slippage': total_slippage,
        'slippage_pct_of_notional': (total_slippage / total_notional) * 100,
        'impact_per_trade': impact_cost_per_trade,
        'total_impact': total_impact,
        'impact_pct_of_notional': (total_impact / total_notional) * 100,
        'simple_total_cost': simple_total,
        'simple_cost_pct': (simple_total / total_notional) * 100,
        'full_total_cost': full_total,
        'full_cost_pct': (full_total / total_notional) * 100
    }
    
    return result