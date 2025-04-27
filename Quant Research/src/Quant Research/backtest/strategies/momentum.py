"""
Momentum strategy implementation.

This module implements a time-series momentum strategy that converts signals
into trading positions with appropriate risk controls.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from quant_research.core.models import Signal, Trade
from quant_research.backtest.base import BaseStrategy, StrategyType, register_strategy
from quant_research.backtest import risk
from quant_research.backtest.utils import (
    validate_dataframe,
    ensure_columns,
    normalize_positions,
    calculate_metrics
)

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Time-series momentum strategy that follows signals from momentum analytics.
    
    This strategy generates positions based on momentum signals, scales positions
    by volatility if enabled, and applies appropriate risk controls.
    """
    
    # Class variables for strategy metadata
    strategy_type = StrategyType.MOMENTUM
    name = "momentum"
    description = "Time-series momentum strategy that follows signals from momentum analytics"
    version = "1.0.0"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the momentum strategy.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            "vol_lookback": 20,
            "vol_scaling": True,
            "max_leverage": 1.0,
            "stop_loss_pct": None,
            "max_drawdown_pct": None,
            "use_kelly_sizing": False,
            "kelly_fraction": 0.5,
            "target_volatility": None
        }
    
    @classmethod
    def get_parameter_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about strategy parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata
        """
        return {
            "vol_lookback": {
                "type": "int",
                "description": "Period (in bars) for volatility calculation",
                "default": 20,
                "min": 2,
                "max": 252
            },
            "vol_scaling": {
                "type": "bool",
                "description": "Whether to scale positions by volatility",
                "default": True
            },
            "max_leverage": {
                "type": "float",
                "description": "Maximum allowed leverage",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0
            },
            "stop_loss_pct": {
                "type": "float",
                "description": "Stop loss percentage",
                "default": None,
                "min": 0.1
            },
            "max_drawdown_pct": {
                "type": "float",
                "description": "Maximum drawdown percentage",
                "default": None,
                "min": 0.1
            },
            "use_kelly_sizing": {
                "type": "bool",
                "description": "Whether to use Kelly criterion for position sizing",
                "default": False
            },
            "kelly_fraction": {
                "type": "float",
                "description": "Fraction of Kelly to use (0-1)",
                "default": 0.5,
                "min": 0.1,
                "max": 1.0
            },
            "target_volatility": {
                "type": "float",
                "description": "Target annualized volatility",
                "default": None,
                "min": 0.01
            }
        }
    
    def initialize(self) -> None:
        """
        Perform strategy-specific initialization.
        """
        logger.info(f"Initializing {self.name} strategy with vol_scaling={self.params['vol_scaling']}")
    
    def _filter_momentum_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals to get momentum-specific signals only.
        
        Args:
            signals_df: DataFrame with all signals
            
        Returns:
            DataFrame with momentum signals only
        """
        # Validate input
        is_valid, missing_cols = validate_dataframe(
            signals_df, 
            ['timestamp', 'asset_id', 'strategy'], 
            self.name
        )
        
        if not is_valid:
            logger.warning(f"Invalid signals DataFrame: missing columns {missing_cols}")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'direction', 'strength'])
        
        # Filter for momentum signals
        momentum_signals = signals_df[signals_df['strategy'] == 'momentum'].copy()
        
        if len(momentum_signals) == 0:
            logger.warning("No momentum signals found in input data")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'direction', 'strength'])
        
        # Ensure required columns
        required_columns = {
            'direction': 0.0,  # Default to neutral
            'strength': 0.5    # Default to medium strength
        }
        
        momentum_signals = ensure_columns(momentum_signals, required_columns, self.name)
        
        return momentum_signals
    
    def _calculate_volatility(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility for each asset.
        
        Args:
            prices_df: DataFrame with price data
            
        Returns:
            DataFrame with volatility values
        """
        vol_lookback = self.params['vol_lookback']
        
        # Validate input
        is_valid, missing_cols = validate_dataframe(
            prices_df, 
            ['timestamp', 'asset_id', 'close'], 
            self.name
        )
        
        if not is_valid:
            logger.warning(f"Invalid prices DataFrame: missing columns {missing_cols}")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'volatility'])
        
        # Pivot price data
        prices_pivot = prices_df.pivot(index='timestamp', columns='asset_id', values='close')
        
        # Calculate returns
        returns = prices_pivot.pct_change().dropna()
        
        # Calculate rolling volatility
        vol = returns.rolling(vol_lookback).std()
        
        # Convert back to long format
        vol = vol.stack().reset_index()
        vol.columns = ['timestamp', 'asset_id', 'volatility']
        
        # Fill missing values with mean
        vol['volatility'] = vol['volatility'].fillna(vol['volatility'].mean())
        
        return vol
    
    def generate_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions from momentum signals.
        
        Args:
            signals_df: DataFrame with momentum signals
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with positions [timestamp, asset_id, position, target_weight]
        """
        logger.info("Generating positions from momentum signals")
        
        # Extract parameters
        vol_scaling = self.params['vol_scaling']
        max_leverage = self.params['max_leverage']
        
        # Filter for momentum signals
        momentum_signals = self._filter_momentum_signals(signals_df)
        
        if len(momentum_signals) == 0:
            logger.warning("No momentum signals to generate positions from")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
        
        # Merge signals with prices
        merged = pd.merge(
            momentum_signals,
            prices_df[['timestamp', 'asset_id', 'close']],
            on=['timestamp', 'asset_id'],
            how='left'
        )
        
        # Apply volatility scaling if enabled
        if vol_scaling:
            # Calculate volatility
            vol_df = self._calculate_volatility(prices_df)
            
            # Merge with signals
            merged = pd.merge(
                merged,
                vol_df,
                on=['timestamp', 'asset_id'],
                how='left'
            )
            
            # Fill missing volatility values
            merged['volatility'] = merged['volatility'].fillna(merged['volatility'].mean())
            
            # Scale by inverse volatility
            merged['position'] = merged['direction'] * merged['strength'] / merged['volatility']
        else:
            # Simple position based on signal strength and direction
            merged['position'] = merged['direction'] * merged['strength']
        
        # Normalize positions to enforce leverage constraint
        positions_df = normalize_positions(
            merged, 
            max_leverage=max_leverage,
            group_columns=['timestamp'],
            position_column='position',
            weight_column='target_weight'
        )
        
        # Select only necessary columns
        result = positions_df[['timestamp', 'asset_id', 'position', 'target_weight']]
        
        logger.info(f"Generated {len(result)} position entries")
        return result
    
    def apply_risk_controls(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk controls to positions.
        
        Args:
            positions_df: DataFrame with positions
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with risk-adjusted positions
        """
        logger.info("Applying risk controls to positions")
        
        if len(positions_df) == 0:
            logger.warning("No positions to apply risk controls to")
            return positions_df
        
        # Extract risk parameters
        stop_loss_pct = self.params['stop_loss_pct']
        max_drawdown_pct = self.params['max_drawdown_pct']
        use_kelly_sizing = self.params['use_kelly_sizing']
        kelly_fraction = self.params['kelly_fraction']
        target_volatility = self.params['target_volatility']
        vol_lookback = self.params['vol_lookback']
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
        # Apply stop loss
        if stop_loss_pct is not None:
            result = risk.apply_stop_loss(
                result, 
                prices_df, 
                stop_loss_pct
            )
        
        # Apply drawdown guard
        if max_drawdown_pct is not None:
            result = risk.apply_drawdown_guard(
                result,
                prices_df,
                max_drawdown_pct
            )
        
        # Apply Kelly criterion position sizing
        if use_kelly_sizing:
            result = risk.apply_kelly_sizing(
                result,
                prices_df,
                lookback=vol_lookback * 2,
                fraction=kelly_fraction
            )
        
        # Apply volatility targeting
        if target_volatility is not None:
            result = risk.apply_vol_targeting(
                result,
                prices_df,
                target_volatility,
                vol_lookback
            )
        
        return result


# Register the strategy with the registry
register_strategy(MomentumStrategy)


# For backward compatibility with the engine registration system
def run_strategy(signals_df: pd.DataFrame, prices_df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Run the momentum strategy (legacy entry point).
    
    This function maintains compatibility with the old engine registration system.
    
    Args:
        signals_df: DataFrame with signals
        prices_df: DataFrame with prices
        **params: Strategy parameters
        
    Returns:
        DataFrame with positions
    """
    strategy = MomentumStrategy(**params)
    return strategy.run_strategy(signals_df, prices_df)