"""
Cross-Exchange Arbitrage strategy implementation.

This module implements a strategy that exploits price differences of the same
asset across different exchanges, accounting for fees, latency, and execution risks.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from quant_research.core.models import Signal, Trade
from quant_research.backtest.base import BaseStrategy, StrategyType, register_strategy
from quant_research.backtest import risk, fees
from quant_research.backtest.utils import (
    validate_dataframe,
    ensure_columns,
    normalize_positions,
    cap_position_sizes
)

logger = logging.getLogger(__name__)


class CrossExchangeArbitrageStrategy(BaseStrategy):
    """
    Cross-exchange arbitrage strategy that exploits price differences across venues.
    
    This strategy identifies arbitrage opportunities between exchanges,
    generates balanced buy/sell positions, simulates execution with slippage,
    and applies appropriate risk controls.
    """
    
    # Class variables for strategy metadata
    strategy_type = StrategyType.ARBITRAGE
    name = "cross_exchange_arbitrage"
    description = "Strategy that exploits price differences across exchanges"
    version = "1.0.0"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the cross-exchange arbitrage strategy.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            "min_spread_pct": 0.2,
            "min_profit_usd": 10.0,
            "max_leverage": 1.0,
            "max_position_usd": 10000.0,
            "spread_weight": True,
            "slippage_model": "fixed",
            "fixed_slippage_bps": 3,
            "latency_ms": 100,
            "min_adjusted_spread": 0.05,
            "max_inventory_deviation": 0.1,
            "fee_structure": {
                "binance": 0.1,
                "coinbase": 0.2,
                "kraken": 0.16,
                "ftx": 0.07,
                "huobi": 0.2
            },
            "max_inventory": {
                "BTC": 1.0,
                "ETH": 10.0,
                "SOL": 100.0,
                "default": 1000.0
            }
        }
    
    @classmethod
    def get_parameter_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about strategy parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata
        """
        return {
            "min_spread_pct": {
                "type": "float",
                "description": "Minimum spread percentage (after fees) to consider an opportunity",
                "default": 0.2,
                "min": 0.01,
                "max": 5.0
            },
            "min_profit_usd": {
                "type": "float",
                "description": "Minimum profit in USD to consider an opportunity",
                "default": 10.0,
                "min": 0.1
            },
            "max_leverage": {
                "type": "float",
                "description": "Maximum allowed leverage",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0
            },
            "max_position_usd": {
                "type": "float",
                "description": "Maximum position size in USD",
                "default": 10000.0,
                "min": 100.0
            },
            "spread_weight": {
                "type": "bool",
                "description": "Whether to scale position size by spread attractiveness",
                "default": True
            },
            "slippage_model": {
                "type": "string",
                "description": "Slippage model to use ('fixed' or 'latency')",
                "default": "fixed",
                "allowed_values": ["fixed", "latency", "none"]
            },
            "fixed_slippage_bps": {
                "type": "int",
                "description": "Fixed slippage in basis points",
                "default": 3,
                "min": 0,
                "max": 100
            },
            "latency_ms": {
                "type": "int",
                "description": "Execution latency in milliseconds (for latency slippage model)",
                "default": 100,
                "min": 0,
                "max": 1000
            },
            "min_adjusted_spread": {
                "type": "float",
                "description": "Minimum spread after slippage to proceed with execution",
                "default": 0.05,
                "min": 0.01
            },
            "max_inventory_deviation": {
                "type": "float",
                "description": "Maximum allowed deviation between buy and sell leg quantities",
                "default": 0.1,
                "min": 0.01,
                "max": 0.5
            },
            "fee_structure": {
                "type": "dict",
                "description": "Fee structure by exchange (in percentage)",
                "default": {
                    "binance": 0.1,
                    "coinbase": 0.2,
                    "kraken": 0.16,
                    "ftx": 0.07,
                    "huobi": 0.2
                }
            },
            "max_inventory": {
                "type": "dict",
                "description": "Maximum inventory allowed per asset",
                "default": {
                    "BTC": 1.0,
                    "ETH": 10.0,
                    "SOL": 100.0,
                    "default": 1000.0
                }
            }
        }
    
    def initialize(self) -> None:
        """
        Perform strategy-specific initialization.
        """
        logger.info(
            f"Initializing {self.name} strategy with min_spread_pct={self.params['min_spread_pct']}, "
            f"slippage_model={self.params['slippage_model']}"
        )
        
        # Initialize opportunity tracking
        self._opportunities = pd.DataFrame()
        
        # Initialize inventory tracking
        self._inventory = {}
    
    def _validate_price_data(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate price data for arbitrage analysis.
        
        Args:
            prices_df: DataFrame with price data
            
        Returns:
            Validated DataFrame with required columns
        """
        # Define required columns
        required_columns = ['timestamp', 'asset_id', 'exchange_id', 'bid', 'ask']
        
        # Validate dataframe
        is_valid, missing_cols = validate_dataframe(
            prices_df, 
            required_columns, 
            self.name
        )
        
        if not is_valid:
            logger.warning(f"Invalid price DataFrame: missing columns {missing_cols}")
            
            # Try to recover with available data if possible
            if 'bid' not in prices_df.columns and 'close' in prices_df.columns:
                logger.info("Creating 'bid' column from 'close'")
                prices_df['bid'] = prices_df['close'] * 0.9999
            
            if 'ask' not in prices_df.columns and 'close' in prices_df.columns:
                logger.info("Creating 'ask' column from 'close'")
                prices_df['ask'] = prices_df['close'] * 1.0001
            
            if 'exchange_id' not in prices_df.columns:
                logger.warning("Missing 'exchange_id' column, cannot perform cross-exchange analysis")
                return pd.DataFrame(columns=required_columns)
            
            # Recheck validation
            is_valid, missing_cols = validate_dataframe(
                prices_df, 
                required_columns, 
                self.name
            )
            
            if not is_valid:
                logger.error(f"Still missing required columns after recovery attempt: {missing_cols}")
                return pd.DataFrame(columns=required_columns)
        
        # Ensure volume columns exist (required for slippage modeling)
        prices_df = ensure_columns(
            prices_df, 
            {
                'bid_volume': 100000,  # Default volume
                'ask_volume': 100000   # Default volume
            }, 
            self.name
        )
        
        return prices_df
    
    def _identify_arbitrage_opportunities(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify arbitrage opportunities across exchanges.
        
        Args:
            prices_df: DataFrame with price data
            
        Returns:
            DataFrame with arbitrage opportunities
        """
        logger.info("Identifying cross-exchange arbitrage opportunities")
        
        # Extract parameters
        min_spread_pct = self.params['min_spread_pct']
        min_profit_usd = self.params['min_profit_usd']
        fee_structure = self.params['fee_structure']
        
        # Validate price data
        validated_prices = self._validate_price_data(prices_df)
        
        if validated_prices.empty:
            logger.warning("No valid price data for arbitrage analysis")
            return pd.DataFrame()
        
        # Pivot to get bid/ask for each exchange for each asset
        bids_pivot = validated_prices.pivot_table(
            index=['timestamp', 'asset_id'],
            columns='exchange_id',
            values='bid'
        ).reset_index()
        
        asks_pivot = validated_prices.pivot_table(
            index=['timestamp', 'asset_id'],
            columns='exchange_id',
            values='ask'
        ).reset_index()
        
        # Merge bid and ask pivots
        pivoted = pd.merge(
            bids_pivot,
            asks_pivot,
            on=['timestamp', 'asset_id'],
            suffixes=('_bid', '_ask')
        )
        
        # Get list of exchanges from column names
        exchange_ids = [col.split('_bid')[0] for col in pivoted.columns if col.endswith('_bid')]
        
        # Initialize list to store opportunities
        opportunities = []
        
        # Loop through timestamp and asset_id combinations
        for _, row in pivoted.iterrows():
            timestamp = row['timestamp']
            asset_id = row['asset_id']
            
            # For each pair of exchanges
            for buy_exchange in exchange_ids:
                for sell_exchange in exchange_ids:
                    if buy_exchange == sell_exchange:
                        continue
                    
                    buy_ask = row[f"{buy_exchange}_ask"]
                    sell_bid = row[f"{sell_exchange}_bid"]
                    
                    # Skip if missing data
                    if pd.isna(buy_ask) or pd.isna(sell_bid):
                        continue
                    
                    # Calculate raw spread
                    raw_spread_pct = (sell_bid / buy_ask - 1) * 100
                    
                    # Calculate fees
                    buy_fee_pct = fee_structure.get(buy_exchange, 0.1)  # Default to 0.1%
                    sell_fee_pct = fee_structure.get(sell_exchange, 0.1)
                    total_fee_pct = buy_fee_pct + sell_fee_pct
                    
                    # Calculate net spread after fees
                    net_spread_pct = raw_spread_pct - total_fee_pct
                    
                    # Check if spread is profitable
                    if net_spread_pct > min_spread_pct:
                        # Calculate profit for a standardized position size (e.g., $1000)
                        position_size = 1000.0  # USD
                        estimated_profit = (position_size * net_spread_pct / 100)
                        
                        if estimated_profit >= min_profit_usd:
                            opportunities.append({
                                'timestamp': timestamp,
                                'asset_id': asset_id,
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': buy_ask,
                                'sell_price': sell_bid,
                                'raw_spread_pct': raw_spread_pct,
                                'total_fee_pct': total_fee_pct,
                                'net_spread_pct': net_spread_pct,
                                'estimated_profit': estimated_profit
                            })
        
        # Convert to DataFrame
        opps_df = pd.DataFrame(opportunities)
        
        if len(opps_df) == 0:
            logger.warning("No arbitrage opportunities found")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'buy_exchange', 'sell_exchange',
                'buy_price', 'sell_price', 'raw_spread_pct', 'total_fee_pct',
                'net_spread_pct', 'estimated_profit'
            ])
        
        # Store opportunities for later use
        self._opportunities = opps_df
        
        logger.info(f"Found {len(opps_df)} arbitrage opportunities")
        return opps_df
    
    def _generate_positions_from_opportunities(self, opps_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions from arbitrage opportunities.
        
        Args:
            opps_df: DataFrame with arbitrage opportunities
            prices_df: DataFrame with price data
            
        Returns:
            DataFrame with positions
        """
        logger.info("Generating positions from arbitrage opportunities")
        
        # Extract parameters
        max_leverage = self.params['max_leverage']
        max_position_usd = self.params['max_position_usd']
        spread_weight = self.params['spread_weight']
        max_inventory = self.params['max_inventory']
        
        if len(opps_df) == 0:
            logger.warning("No arbitrage opportunities to generate positions from")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'exchange_id', 'position', 'target_weight', 'leg_type'
            ])
        
        # Initialize positions list
        positions = []
        
        # Track inventory per asset across different opportunities
        current_inventory = self._inventory.copy()
        if not current_inventory:
            current_inventory = {asset_id: 0.0 for asset_id in opps_df['asset_id'].unique()}
        
        # Process opportunities
        for timestamp, time_opps in opps_df.groupby('timestamp'):
            # Sort by profitability (best opportunities first)
            time_opps = time_opps.sort_values('net_spread_pct', ascending=False)
            
            # Calculate position sizes based on spread attractiveness
            for _, opp in time_opps.iterrows():
                asset_id = opp['asset_id']
                
                # Check if we have room in inventory
                asset_max_inventory = max_inventory.get(
                    asset_id, 
                    max_inventory.get('default', float('inf'))
                )
                
                if abs(current_inventory.get(asset_id, 0.0)) >= asset_max_inventory:
                    continue
                
                # Calculate position size
                if spread_weight:
                    # Scale position by spread attractiveness (more aggressive for better spreads)
                    position_size = min(
                        max_position_usd * (opp['net_spread_pct'] / 5.0),  # Scale by spread
                        max_position_usd,
                        asset_max_inventory - abs(current_inventory.get(asset_id, 0.0))
                    )
                else:
                    position_size = min(
                        max_position_usd,
                        asset_max_inventory - abs(current_inventory.get(asset_id, 0.0))
                    )
                
                # Ensure minimum size
                if position_size < 100:  # Minimum $100 to avoid dust positions
                    continue
                    
                # Add buy leg position
                positions.append({
                    'timestamp': timestamp,
                    'asset_id': asset_id,
                    'exchange_id': opp['buy_exchange'],
                    'position': position_size / opp['buy_price'],  # Convert USD to quantity
                    'target_weight': position_size / (max_leverage * max_position_usd),
                    'leg_type': 'buy',
                    'price': opp['buy_price'],
                    'spread_pct': opp['net_spread_pct']
                })
                
                # Add sell leg position
                positions.append({
                    'timestamp': timestamp,
                    'asset_id': asset_id,
                    'exchange_id': opp['sell_exchange'],
                    'position': -position_size / opp['sell_price'],  # Negative for sell
                    'target_weight': -position_size / (max_leverage * max_position_usd),
                    'leg_type': 'sell',
                    'price': opp['sell_price'],
                    'spread_pct': opp['net_spread_pct']
                })
                
                # Update inventory
                current_inventory[asset_id] = current_inventory.get(asset_id, 0.0) + \
                                             position_size / opp['buy_price'] - \
                                             position_size / opp['sell_price']
        
        # Store updated inventory
        self._inventory = current_inventory
        
        # Convert to DataFrame
        positions_df = pd.DataFrame(positions)
        
        if len(positions_df) == 0:
            logger.warning("No positions generated from arbitrage opportunities")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'exchange_id', 'position', 'target_weight', 'leg_type'
            ])
        
        logger.info(f"Generated {len(positions_df)} position entries")
        return positions_df
    
    def _simulate_execution_slippage(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate execution slippage for arbitrage positions.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            DataFrame with slippage-adjusted positions
        """
        logger.info("Simulating execution slippage")
        
        # Extract parameters
        slippage_model = self.params['slippage_model']
        fixed_slippage_bps = self.params['fixed_slippage_bps']
        latency_ms = self.params['latency_ms']
        
        if len(positions_df) == 0:
            return positions_df
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
        if slippage_model == 'fixed':
            # Apply fixed slippage to each leg
            for leg_type in ['buy', 'sell']:
                mask = result['leg_type'] == leg_type
                
                if leg_type == 'buy':
                    # Higher price for buys
                    result.loc[mask, 'execution_price'] = result.loc[mask, 'price'] * (1 + fixed_slippage_bps / 10000)
                else:
                    # Lower price for sells
                    result.loc[mask, 'execution_price'] = result.loc[mask, 'price'] * (1 - fixed_slippage_bps / 10000)
        
        elif slippage_model == 'latency':
            # Simulate price drift during latency
            # This is a simplified model - in a real system, this would be more complex
            drift_factor = latency_ms / 1000 * 0.0001  # 1 bp per 100ms as an example
            
            for leg_type in ['buy', 'sell']:
                mask = result['leg_type'] == leg_type
                
                if leg_type == 'buy':
                    # Prices tend to move against you - up for buys
                    result.loc[mask, 'execution_price'] = result.loc[mask, 'price'] * (1 + drift_factor)
                else:
                    # Down for sells
                    result.loc[mask, 'execution_price'] = result.loc[mask, 'price'] * (1 - drift_factor)
        else:
            # No slippage model or unknown model
            result['execution_price'] = result['price']
        
        # Calculate slippage impact
        result['slippage_pct'] = 0.0
        
        # For buys: (execution_price - quoted_price) / quoted_price * 100
        buy_mask = result['leg_type'] == 'buy'
        if any(buy_mask):
            result.loc[buy_mask, 'slippage_pct'] = (
                (result.loc[buy_mask, 'execution_price'] - result.loc[buy_mask, 'price']) / 
                result.loc[buy_mask, 'price'] * 100
            )
        
        # For sells: (quoted_price - execution_price) / quoted_price * 100
        sell_mask = result['leg_type'] == 'sell'
        if any(sell_mask):
            result.loc[sell_mask, 'slippage_pct'] = (
                (result.loc[sell_mask, 'price'] - result.loc[sell_mask, 'execution_price']) / 
                result.loc[sell_mask, 'price'] * 100
            )
        
        # Calculate impact on spread
        result['adjusted_spread_pct'] = result['spread_pct'] - result['slippage_pct']
        
        logger.info(f"Average slippage: {result['slippage_pct'].mean():.2f} bps")
        
        return result
    
    def _balance_inventory(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance inventory between buy and sell legs.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            DataFrame with balanced positions
        """
        logger.info("Balancing inventory between legs")
        
        # Extract parameters
        max_inventory_deviation = self.params['max_inventory_deviation']
        
        if len(positions_df) == 0:
            return positions_df
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
        # Check inventory balance between legs
        # Group by timestamp and asset_id
        for (timestamp, asset_id), group in result.groupby(['timestamp', 'asset_id']):
            # Check if we have both buy and sell legs
            buy_leg = group[group['leg_type'] == 'buy']
            sell_leg = group[group['leg_type'] == 'sell']
            
            if len(buy_leg) == 0 or len(sell_leg) == 0:
                # Skip if we don't have both legs
                continue
            
            # Calculate quantities in absolute terms
            buy_qty = buy_leg['position'].sum()
            sell_qty = -sell_leg['position'].sum()  # Negate to get positive value
            
            # Check for deviation
            avg_qty = (buy_qty + sell_qty) / 2
            max_allowed_deviation = avg_qty * max_inventory_deviation
            
            if abs(buy_qty - sell_qty) > max_allowed_deviation:
                # Inventory imbalance detected
                logger.warning(
                    f"Inventory imbalance for {asset_id} at {timestamp}: "
                    f"buy={buy_qty:.2f}, sell={sell_qty:.2f}, diff={buy_qty - sell_qty:.2f}"
                )
                
                # Reduce the larger leg to match the smaller one
                if buy_qty > sell_qty:
                    # Reduce buy leg
                    scale_factor = sell_qty / buy_qty
                    idx = buy_leg.index
                    result.loc[idx, 'position'] = result.loc[idx, 'position'] * scale_factor
                    result.loc[idx, 'target_weight'] = result.loc[idx, 'target_weight'] * scale_factor
                else:
                    # Reduce sell leg
                    scale_factor = buy_qty / sell_qty
                    idx = sell_leg.index
                    result.loc[idx, 'position'] = result.loc[idx, 'position'] * scale_factor
                    result.loc[idx, 'target_weight'] = result.loc[idx, 'target_weight'] * scale_factor
        
        return result
    
    def generate_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions from price data (arbitrage doesn't rely heavily on signals).
        
        Args:
            signals_df: DataFrame with signals (mostly unused for arbitrage)
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with positions [timestamp, asset_id, exchange_id, position, target_weight]
        """
        logger.info("Generating positions for cross-exchange arbitrage")
        
        # Find arbitrage opportunities
        opps_df = self._identify_arbitrage_opportunities(prices_df)
        
        if len(opps_df) == 0:
            logger.warning("No arbitrage opportunities found")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'exchange_id', 'position', 'target_weight'
            ])
        
        # Generate positions from opportunities
        positions_df = self._generate_positions_from_opportunities(opps_df, prices_df)
        
        if len(positions_df) == 0:
            logger.warning("No positions generated from opportunities")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'exchange_id', 'position', 'target_weight'
            ])
        
        # Simulate execution slippage
        positions_df = self._simulate_execution_slippage(positions_df)
        
        # Filter by minimum adjusted spread
        min_adjusted_spread = self.params['min_adjusted_spread']
        if 'adjusted_spread_pct' in positions_df.columns:
            initial_count = len(positions_df)
            positions_df = positions_df[positions_df['adjusted_spread_pct'] > min_adjusted_spread]
            filtered_count = initial_count - len(positions_df)
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} positions with insufficient adjusted spread")
        
        # Balance inventory between legs
        positions_df = self._balance_inventory(positions_df)
        
        # Normalize leverage
        positions_df = normalize_positions(
            positions_df,
            max_leverage=self.params['max_leverage'],
            group_columns=['timestamp'],
            position_column='position',
            weight_column='target_weight'
        )
        
        return positions_df
    
    def apply_risk_controls(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk controls to positions.
        
        For arbitrage, most risk controls are already applied during position generation,
        so this is primarily a pass-through function.
        
        Args:
            positions_df: DataFrame with positions
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with risk-adjusted positions
        """
        logger.info("Applying risk controls to arbitrage positions")
        
        if len(positions_df) == 0:
            return positions_df
        
        # For arbitrage, most risk management is done in position generation
        # This function is primarily a pass-through
        
        # But we may want to add some additional portfolio-level controls
        
        # Cap position sizes per asset
        max_inventory = self.params['max_inventory']
        positions_df = cap_position_sizes(
            positions_df,
            max_position=max_inventory,
            asset_column='asset_id',
            position_column='position'
        )
        
        return positions_df


# Register the strategy with the registry
register_strategy(CrossExchangeArbitrageStrategy)


# For backward compatibility with the engine registration system
def run_strategy(signals_df: pd.DataFrame, prices_df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Run the cross-exchange arbitrage strategy (legacy entry point).
    
    This function maintains compatibility with the old engine registration system.
    
    Args:
        signals_df: DataFrame with signals
        prices_df: DataFrame with prices
        **params: Strategy parameters
        
    Returns:
        DataFrame with positions
    """
    strategy = CrossExchangeArbitrageStrategy(**params)
    return strategy.run_strategy(signals_df, prices_df)