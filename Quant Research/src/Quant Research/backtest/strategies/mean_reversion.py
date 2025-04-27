"""
Mean Reversion strategy implementation.

This module implements a mean reversion strategy that trades based on statistical
deviations from equilibrium prices, primarily using z-score inputs from stat-arb analytics.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
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


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy that trades based on statistical deviations from equilibrium prices.
    
    This strategy generates positions based on z-score signals from statistical arbitrage
    models, with asymmetric entry/exit rules and appropriate risk controls.
    """
    
    # Class variables for strategy metadata
    strategy_type = StrategyType.MEAN_REVERSION
    name = "mean_reversion"
    description = "Mean reversion strategy that trades based on statistical deviations from equilibrium prices"
    version = "1.0.0"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the mean reversion strategy.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "max_holding_periods": 10,
            "max_leverage": 1.0,
            "asymmetric_sizing": True,
            "stop_loss_pct": 2.0,
            "max_drawdown_pct": None,
            "time_stop_periods": None,
            "target_volatility": None,
            "vol_lookback": 20
        }
    
    @classmethod
    def get_parameter_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about strategy parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata
        """
        return {
            "entry_threshold": {
                "type": "float",
                "description": "Z-score threshold for position entry",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0
            },
            "exit_threshold": {
                "type": "float",
                "description": "Z-score threshold for position exit",
                "default": 0.5,
                "min": 0.1,
                "max": 2.0
            },
            "max_holding_periods": {
                "type": "int",
                "description": "Maximum number of periods to hold a position",
                "default": 10,
                "min": 1,
                "max": 100
            },
            "max_leverage": {
                "type": "float",
                "description": "Maximum allowed leverage",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0
            },
            "asymmetric_sizing": {
                "type": "bool",
                "description": "Whether to scale position size based on z-score magnitude",
                "default": True
            },
            "stop_loss_pct": {
                "type": "float",
                "description": "Stop loss percentage (typically tighter for mean reversion)",
                "default": 2.0,
                "min": 0.1
            },
            "max_drawdown_pct": {
                "type": "float",
                "description": "Maximum drawdown percentage",
                "default": None,
                "min": 0.1
            },
            "time_stop_periods": {
                "type": "int",
                "description": "Time-based stop (number of periods)",
                "default": None,
                "min": 1
            },
            "target_volatility": {
                "type": "float",
                "description": "Target annualized volatility",
                "default": None,
                "min": 0.01
            },
            "vol_lookback": {
                "type": "int",
                "description": "Period (in bars) for volatility calculation",
                "default": 20,
                "min": 2,
                "max": 252
            }
        }
    
    def initialize(self) -> None:
        """
        Perform strategy-specific initialization.
        """
        logger.info(
            f"Initializing {self.name} strategy with entry_threshold={self.params['entry_threshold']}, "
            f"exit_threshold={self.params['exit_threshold']}"
        )
        
        # Initialize position tracking state
        self._position_state = {}
    
    def _filter_mean_reversion_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals to get mean reversion-specific signals only.
        
        Args:
            signals_df: DataFrame with all signals
            
        Returns:
            DataFrame with mean reversion signals only
        """
        # Validate input
        is_valid, missing_cols = validate_dataframe(
            signals_df, 
            ['timestamp', 'asset_id', 'strategy'], 
            self.name
        )
        
        if not is_valid:
            logger.warning(f"Invalid signals DataFrame: missing columns {missing_cols}")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'z_score'])
        
        # Filter for mean reversion signals
        mean_rev_signals = signals_df[signals_df['strategy'] == 'mean_reversion'].copy()
        
        if len(mean_rev_signals) == 0:
            logger.warning("No mean reversion signals found in input data")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'z_score'])
        
        # Ensure z-score column exists
        if 'z_score' not in mean_rev_signals.columns:
            logger.error("Z-score column missing from mean reversion signals")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'z_score'])
        
        return mean_rev_signals
    
    def _track_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Track positions over time based on z-score changes and holding periods.
        
        Args:
            signals_df: DataFrame with mean reversion signals
            prices_df: DataFrame with price data
            
        Returns:
            DataFrame with tracked positions
        """
        # Extract parameters
        entry_threshold = self.params['entry_threshold']
        exit_threshold = self.params['exit_threshold']
        max_holding_periods = self.params['max_holding_periods']
        asymmetric_sizing = self.params['asymmetric_sizing']
        
        # Validate input
        if len(signals_df) == 0:
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'position', 'z_score', 'holding_periods'
            ])
        
        # Merge signals with prices
        merged = pd.merge(
            signals_df,
            prices_df[['timestamp', 'asset_id', 'close']],
            on=['timestamp', 'asset_id'],
            how='left'
        )
        
        # Initialize positions list
        positions = []
        
        # Group by asset_id to track positions over time
        for asset_id, asset_data in merged.groupby('asset_id'):
            # Sort by timestamp
            asset_data = asset_data.sort_values('timestamp')
            
            # Initialize position tracking for this asset
            current_position = 0
            entry_price = None
            holding_periods = 0
            
            # Initialize or retrieve asset state from internal tracking
            if asset_id not in self._position_state:
                self._position_state[asset_id] = {
                    'position': 0,
                    'entry_price': None,
                    'holding_periods': 0
                }
            else:
                current_position = self._position_state[asset_id]['position']
                entry_price = self._position_state[asset_id]['entry_price']
                holding_periods = self._position_state[asset_id]['holding_periods']
            
            # Process each signal
            for idx, row in asset_data.iterrows():
                z_score = row['z_score']
                timestamp = row['timestamp']
                
                # Position sizing based on z-score magnitude and direction
                if current_position == 0:  # No position, check for entry
                    if z_score <= -entry_threshold:  # Long entry
                        current_position = 1.0
                        if asymmetric_sizing:
                            # Scale by how extreme the z-score is
                            current_position = min(abs(z_score) / entry_threshold, 2.0)
                        entry_price = row['close']
                        holding_periods = 0
                    elif z_score >= entry_threshold:  # Short entry
                        current_position = -1.0
                        if asymmetric_sizing:
                            # Scale by how extreme the z-score is
                            current_position = -min(abs(z_score) / entry_threshold, 2.0)
                        entry_price = row['close']
                        holding_periods = 0
                else:  # Have position, check for exit or adjustment
                    holding_periods += 1
                    
                    # Exit conditions
                    if (current_position > 0 and z_score >= -exit_threshold) or \
                       (current_position < 0 and z_score <= exit_threshold) or \
                       (holding_periods >= max_holding_periods):
                        # Exit position
                        current_position = 0
                        entry_price = None
                        holding_periods = 0
                    else:
                        # Position adjustment for changing z-score (optional)
                        if asymmetric_sizing:
                            if current_position > 0 and z_score < -entry_threshold:
                                # Increase long position as z-score becomes more negative
                                current_position = min(abs(z_score) / entry_threshold, 2.0)
                            elif current_position < 0 and z_score > entry_threshold:
                                # Increase short position as z-score becomes more positive
                                current_position = -min(abs(z_score) / entry_threshold, 2.0)
                
                # Record the position
                positions.append({
                    'timestamp': timestamp,
                    'asset_id': asset_id,
                    'position': current_position,
                    'z_score': z_score,
                    'holding_periods': holding_periods
                })
                
                # Update internal state
                self._position_state[asset_id] = {
                    'position': current_position,
                    'entry_price': entry_price,
                    'holding_periods': holding_periods
                }
        
        # Convert to DataFrame
        positions_df = pd.DataFrame(positions)
        
        if len(positions_df) == 0:
            logger.warning("No positions generated after mean reversion logic")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'position', 'z_score', 'holding_periods'
            ])
        
        return positions_df
    
    def generate_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions from mean reversion signals.
        
        Args:
            signals_df: DataFrame with mean reversion signals
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with positions [timestamp, asset_id, position, target_weight]
        """
        logger.info("Generating positions from mean reversion signals")
        
        # Extract parameters
        max_leverage = self.params['max_leverage']
        
        # Filter for mean reversion signals
        mean_rev_signals = self._filter_mean_reversion_signals(signals_df)
        
        if len(mean_rev_signals) == 0:
            logger.warning("No mean reversion signals to generate positions from")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
        
        # Track positions based on z-scores and holding periods
        positions_df = self._track_positions(mean_rev_signals, prices_df)
        
        if len(positions_df) == 0:
            logger.warning("No positions generated from mean reversion logic")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
        
        # Normalize positions to respect leverage constraints
        positions_df = normalize_positions(
            positions_df, 
            max_leverage=max_leverage,
            group_columns=['timestamp'],
            position_column='position',
            weight_column='target_weight'
        )
        
        # Select only necessary columns
        result = positions_df[['timestamp', 'asset_id', 'position', 'target_weight']]
        
        logger.info(f"Generated {len(result)} position entries")
        return result
    
    def apply_time_stop(self, positions_df: pd.DataFrame, max_periods: int) -> pd.DataFrame:
        """
        Apply a time-based stop to exit positions after a maximum holding period.
        
        Args:
            positions_df: DataFrame with positions
            max_periods: Maximum number of periods to hold a position
            
        Returns:
            DataFrame with time-stop adjusted positions
        """
        # Check if we have holding period information
        if 'holding_periods' not in positions_df.columns:
            logger.warning("Holding periods information not available for time stop")
            return positions_df
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
        # Apply time-based stop
        time_stop_mask = (result['position'] != 0) & (result['holding_periods'] >= max_periods)
        result.loc[time_stop_mask, 'position'] = 0
        result.loc[time_stop_mask, 'target_weight'] = 0
        
        # Log how many positions were closed
        closed_count = time_stop_mask.sum()
        if closed_count > 0:
            logger.info(f"Closed {closed_count} positions due to time stop")
        
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
        time_stop_periods = self.params['time_stop_periods']
        target_volatility = self.params['target_volatility']
        vol_lookback = self.params['vol_lookback']
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
        # Apply stop loss - typically tighter for mean reversion
        if stop_loss_pct is not None:
            result = risk.apply_stop_loss(
                result, 
                prices_df, 
                stop_loss_pct
            )
        
        # Apply time-based stop
        if time_stop_periods is not None and 'holding_periods' in result.columns:
            result = self.apply_time_stop(
                result,
                time_stop_periods
            )
        
        # Apply drawdown guard
        if max_drawdown_pct is not None:
            result = risk.apply_drawdown_guard(
                result,
                prices_df,
                max_drawdown_pct
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
register_strategy(MeanReversionStrategy)


# For backward compatibility with the engine registration system
def run_strategy(signals_df: pd.DataFrame, prices_df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Run the mean reversion strategy (legacy entry point).
    
    This function maintains compatibility with the old engine registration system.
    
    Args:
        signals_df: DataFrame with signals
        prices_df: DataFrame with prices
        **params: Strategy parameters
        
    Returns:
        DataFrame with positions
    """
    strategy = MeanReversionStrategy(**params)
    return strategy.run_strategy(signals_df, prices_df)