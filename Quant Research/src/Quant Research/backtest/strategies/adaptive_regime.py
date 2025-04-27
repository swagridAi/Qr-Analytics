"""
Adaptive Regime strategy implementation.

This module implements a strategy that dynamically adjusts allocations
based on detected market regimes (e.g., trending, mean-reverting, high volatility).
It consumes regime probability signals from analytics/regimes and adjusts
positions accordingly.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from quant_research.core.models import Signal, Trade
from quant_research.backtest.base import BaseStrategy, StrategyType, register_strategy, StrategyRegistry
from quant_research.backtest import risk
from quant_research.backtest.utils import (
    validate_dataframe,
    ensure_columns,
    normalize_positions,
    blend_positions,
    smooth_positions
)

logger = logging.getLogger(__name__)


class AdaptiveRegimeStrategy(BaseStrategy):
    """
    Adaptive regime strategy that dynamically adjusts allocations based on market regimes.
    
    This strategy identifies the current market regime from regime detection signals,
    then dynamically allocates capital across different sub-strategies optimized for
    each regime type. It supports smooth transitions between regimes.
    """
    
    # Class variables for strategy metadata
    strategy_type = StrategyType.REGIME_ADAPTIVE
    name = "adaptive_regime"
    description = "Strategy that dynamically adjusts allocations based on detected market regimes"
    version = "1.0.0"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the adaptive regime strategy.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            "use_hmm": True,
            "use_change_point": True,
            "regime_thresholds": {
                "trending": 0.6,
                "mean_reverting": 0.6,
                "high_volatility": 0.7,
                "low_volatility": 0.7,
                "normal": 0.5
            },
            "allocation_matrix": {
                "trending": {
                    "momentum": 0.7,
                    "mean_reversion": 0.2,
                    "cross_exchange_arbitrage": 0.1
                },
                "mean_reverting": {
                    "momentum": 0.2,
                    "mean_reversion": 0.7,
                    "cross_exchange_arbitrage": 0.1
                },
                "high_volatility": {
                    "momentum": 0.3,
                    "mean_reversion": 0.2,
                    "cross_exchange_arbitrage": 0.5
                },
                "low_volatility": {
                    "momentum": 0.3,
                    "mean_reversion": 0.5,
                    "cross_exchange_arbitrage": 0.2
                },
                "normal": {
                    "momentum": 0.33,
                    "mean_reversion": 0.33,
                    "cross_exchange_arbitrage": 0.34
                }
            },
            "smooth_transitions": True,
            "transition_window": 3,
            "momentum_params": {},
            "mean_reversion_params": {},
            "cross_exchange_params": {},
            "max_allocation": 1.0,
            "target_volatility": None,
            "max_drawdown_pct": None,
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
            "use_hmm": {
                "type": "bool",
                "description": "Whether to use HMM regime signals",
                "default": True
            },
            "use_change_point": {
                "type": "bool",
                "description": "Whether to use change point regime signals",
                "default": True
            },
            "regime_thresholds": {
                "type": "dict",
                "description": "Probability thresholds to classify regimes",
                "default": {
                    "trending": 0.6,
                    "mean_reverting": 0.6,
                    "high_volatility": 0.7,
                    "low_volatility": 0.7,
                    "normal": 0.5
                }
            },
            "allocation_matrix": {
                "type": "dict",
                "description": "Mapping of regimes to strategy allocations",
                "default": {
                    "trending": {
                        "momentum": 0.7,
                        "mean_reversion": 0.2,
                        "cross_exchange_arbitrage": 0.1
                    },
                    "mean_reverting": {
                        "momentum": 0.2,
                        "mean_reversion": 0.7,
                        "cross_exchange_arbitrage": 0.1
                    },
                    "high_volatility": {
                        "momentum": 0.3,
                        "mean_reversion": 0.2,
                        "cross_exchange_arbitrage": 0.5
                    },
                    "low_volatility": {
                        "momentum": 0.3,
                        "mean_reversion": 0.5,
                        "cross_exchange_arbitrage": 0.2
                    },
                    "normal": {
                        "momentum": 0.33,
                        "mean_reversion": 0.33,
                        "cross_exchange_arbitrage": 0.34
                    }
                }
            },
            "smooth_transitions": {
                "type": "bool",
                "description": "Whether to smooth transitions between regimes",
                "default": True
            },
            "transition_window": {
                "type": "int",
                "description": "Number of bars to smooth regime transitions",
                "default": 3,
                "min": 1,
                "max": 20
            },
            "momentum_params": {
                "type": "dict",
                "description": "Parameters for momentum sub-strategy",
                "default": {}
            },
            "mean_reversion_params": {
                "type": "dict",
                "description": "Parameters for mean reversion sub-strategy",
                "default": {}
            },
            "cross_exchange_params": {
                "type": "dict",
                "description": "Parameters for cross-exchange arbitrage sub-strategy",
                "default": {}
            },
            "max_allocation": {
                "type": "float",
                "description": "Maximum allocation (scaling factor)",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0
            },
            "target_volatility": {
                "type": "float",
                "description": "Target annualized volatility",
                "default": None,
                "min": 0.01
            },
            "max_drawdown_pct": {
                "type": "float",
                "description": "Maximum drawdown percentage",
                "default": None,
                "min": 0.1
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
            f"Initializing {self.name} strategy with smooth_transitions={self.params['smooth_transitions']}, "
            f"transition_window={self.params['transition_window']}"
        )
        
        # Initialize regime tracking state
        self._regime_state = {}
        
        # Initialize transition state
        self._transition_state = {}
        
        # Initialize sub-strategies
        self._initialize_sub_strategies()
    
    def _initialize_sub_strategies(self) -> None:
        """
        Initialize sub-strategy instances.
        """
        self._sub_strategies = {}
        
        # Get strategy parameters
        momentum_params = self.params['momentum_params']
        mean_reversion_params = self.params['mean_reversion_params']
        cross_exchange_params = self.params['cross_exchange_params']
        
        # Create momentum strategy instance
        momentum_strategy = StrategyRegistry.create_strategy("momentum", **momentum_params)
        if momentum_strategy:
            self._sub_strategies["momentum"] = momentum_strategy
            logger.info("Initialized momentum sub-strategy")
        else:
            logger.warning("Failed to initialize momentum sub-strategy")
        
        # Create mean reversion strategy instance
        mean_reversion_strategy = StrategyRegistry.create_strategy("mean_reversion", **mean_reversion_params)
        if mean_reversion_strategy:
            self._sub_strategies["mean_reversion"] = mean_reversion_strategy
            logger.info("Initialized mean reversion sub-strategy")
        else:
            logger.warning("Failed to initialize mean reversion sub-strategy")
        
        # Create cross-exchange arbitrage strategy instance
        arb_strategy = StrategyRegistry.create_strategy("cross_exchange_arbitrage", **cross_exchange_params)
        if arb_strategy:
            self._sub_strategies["cross_exchange_arbitrage"] = arb_strategy
            logger.info("Initialized cross-exchange arbitrage sub-strategy")
        else:
            logger.warning("Failed to initialize cross-exchange arbitrage sub-strategy")
        
        # Check if we have at least one sub-strategy
        if not self._sub_strategies:
            logger.error("No sub-strategies could be initialized")
    
    def _filter_regime_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter signals to get regime-specific signals only.
        
        Args:
            signals_df: DataFrame with all signals
            
        Returns:
            DataFrame with regime signals only
        """
        # Extract parameters
        use_hmm = self.params['use_hmm']
        use_change_point = self.params['use_change_point']
        
        # Validate input
        is_valid, missing_cols = validate_dataframe(
            signals_df, 
            ['timestamp', 'asset_id', 'strategy'], 
            self.name
        )
        
        if not is_valid:
            logger.warning(f"Invalid signals DataFrame: missing columns {missing_cols}")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'regime', 'probability'])
        
        # Filter for regime signals only
        regime_signals = signals_df[
            (signals_df['strategy'] == 'regime_hmm' if use_hmm else False) |
            (signals_df['strategy'] == 'regime_cp' if use_change_point else False)
        ].copy()
        
        if len(regime_signals) == 0:
            logger.warning("No regime signals found in input data")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'regime', 'probability'])
        
        # Check for required columns
        required_cols = ['regime', 'probability']
        missing_regime_cols = [col for col in required_cols if col not in regime_signals.columns]
        
        if missing_regime_cols:
            logger.error(f"Regime signals missing required columns: {missing_regime_cols}")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'strategy', 'regime', 'probability'])
        
        return regime_signals
    
    def _identify_market_regimes(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify the current market regime for each asset.
        
        Args:
            signals_df: DataFrame with regime signals
            
        Returns:
            DataFrame with regime classifications
        """
        logger.info("Identifying market regimes from signals")
        
        # Extract parameters
        regime_thresholds = self.params['regime_thresholds']
        
        # Filter for regime signals
        regime_signals = self._filter_regime_signals(signals_df)
        
        if len(regime_signals) == 0:
            logger.warning("No regime signals to analyze")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'regime', 'probability'])
        
        # Initialize results list
        regime_classifications = []
        
        # Process signals by timestamp and asset
        for (timestamp, asset_id), group in regime_signals.groupby(['timestamp', 'asset_id']):
            # Extract regime probabilities
            # The expected format from analytics is a row per regime with probability
            regime_probs = {}
            
            # HMM signals processing
            hmm_signals = group[group['strategy'] == 'regime_hmm']
            if not hmm_signals.empty:
                for _, row in hmm_signals.iterrows():
                    regime_probs[row['regime']] = row['probability']
            
            # Change point signals processing
            cp_signals = group[group['strategy'] == 'regime_cp']
            if not cp_signals.empty:
                for _, row in cp_signals.iterrows():
                    # If both HMM and CP detect the same regime, use the higher probability
                    if row['regime'] in regime_probs:
                        regime_probs[row['regime']] = max(regime_probs[row['regime']], row['probability'])
                    else:
                        regime_probs[row['regime']] = row['probability']
            
            # Determine dominant regime
            if regime_probs:
                # Find regime with highest probability exceeding its threshold
                dominant_regime = None
                max_prob_diff = -float('inf')
                
                for regime, prob in regime_probs.items():
                    threshold = regime_thresholds.get(regime, 0.5)
                    prob_diff = prob - threshold
                    
                    if prob_diff > 0 and prob_diff > max_prob_diff:
                        dominant_regime = regime
                        max_prob_diff = prob_diff
                
                # If no regime exceeds threshold, default to 'normal'
                if dominant_regime is None:
                    dominant_regime = 'normal'
                    dominant_prob = regime_probs.get('normal', 0.5)
                else:
                    dominant_prob = regime_probs[dominant_regime]
                
                # Check for regime transition
                current_regime = dominant_regime
                current_prob = dominant_prob
                
                # Track previous regime for this asset
                asset_key = str(asset_id)
                prev_regime_info = self._regime_state.get(asset_key)
                
                if prev_regime_info is not None:
                    prev_regime = prev_regime_info['regime']
                    
                    # Update transition state if regime changed
                    if prev_regime != current_regime:
                        self._transition_state[asset_key] = {
                            'from_regime': prev_regime,
                            'to_regime': current_regime,
                            'steps_remaining': self.params['transition_window'] if self.params['smooth_transitions'] else 0
                        }
                        logger.info(f"Regime transition for {asset_id}: {prev_regime} -> {current_regime}")
                
                # Update regime state
                self._regime_state[asset_key] = {
                    'regime': current_regime,
                    'probability': current_prob,
                    'timestamp': timestamp
                }
                
                # Add to results
                regime_classifications.append({
                    'timestamp': timestamp,
                    'asset_id': asset_id,
                    'regime': current_regime,
                    'probability': current_prob,
                    'all_probs': regime_probs  # Store all probabilities for reference
                })
        
        # Convert to DataFrame
        regime_df = pd.DataFrame(regime_classifications)
        
        if len(regime_df) == 0:
            logger.warning("No regime classifications generated")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'regime', 'probability'])
        
        # Get regime distribution stats
        regime_distribution = regime_df.groupby('regime').size()
        logger.info(f"Identified regimes: {regime_distribution.to_dict()}")
        
        return regime_df
    
    def _determine_strategy_allocations(self, regime_df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine strategy allocations based on identified regimes.
        
        Args:
            regime_df: DataFrame with regime classifications
            
        Returns:
            DataFrame with strategy allocations
        """
        logger.info("Determining strategy allocations based on regimes")
        
        # Extract parameters
        allocation_matrix = self.params['allocation_matrix']
        smooth_transitions = self.params['smooth_transitions']
        transition_window = self.params['transition_window']
        
        if len(regime_df) == 0:
            logger.warning("No regime data to determine allocations")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'strategy', 'allocation'
            ])
        
        # Check if allocation matrix covers all regimes
        missing_regimes = set(regime_df['regime'].unique()) - set(allocation_matrix.keys())
        if missing_regimes:
            logger.warning(f"Missing allocation data for regimes: {missing_regimes}")
            # Add default allocations for missing regimes
            for regime in missing_regimes:
                allocation_matrix[regime] = {
                    'momentum': 0.33,
                    'mean_reversion': 0.33,
                    'cross_exchange_arbitrage': 0.34
                }
        
        # Initialize results list
        allocations = []
        
        # Process by asset
        for asset_id, asset_regimes in regime_df.groupby('asset_id'):
            # Sort by timestamp
            asset_regimes = asset_regimes.sort_values('timestamp')
            
            # Process each regime observation
            for idx, row in asset_regimes.iterrows():
                timestamp = row['timestamp']
                regime = row['regime']
                probability = row['probability']
                
                # Get base allocations for current regime
                current_allocations = allocation_matrix.get(regime, {
                    'momentum': 0.33,
                    'mean_reversion': 0.33,
                    'cross_exchange_arbitrage': 0.34
                })
                
                # Check for transition blending
                asset_key = str(asset_id)
                transition_info = self._transition_state.get(asset_key)
                effective_allocations = current_allocations.copy()
                
                if smooth_transitions and transition_info and transition_info['steps_remaining'] > 0:
                    # Get allocations for both 'from' and 'to' regimes
                    from_regime = transition_info['from_regime']
                    from_allocations = allocation_matrix.get(from_regime, {
                        'momentum': 0.33,
                        'mean_reversion': 0.33,
                        'cross_exchange_arbitrage': 0.34
                    })
                    
                    # Calculate blend factor (1.0 = fully transitioned to new regime)
                    progress = (transition_window - transition_info['steps_remaining']) / transition_window
                    
                    # Blend allocations
                    effective_allocations = {}
                    all_strategies = set(list(from_allocations.keys()) + list(current_allocations.keys()))
                    
                    for strategy in all_strategies:
                        from_alloc = from_allocations.get(strategy, 0.0)
                        to_alloc = current_allocations.get(strategy, 0.0)
                        effective_allocations[strategy] = from_alloc * (1 - progress) + to_alloc * progress
                    
                    # Decrement steps remaining
                    transition_info['steps_remaining'] -= 1
                    self._transition_state[asset_key] = transition_info
                    
                    logger.debug(
                        f"Blending allocations for {asset_id}: {from_regime} -> {regime}, "
                        f"progress={progress:.2f}, steps_remaining={transition_info['steps_remaining']}"
                    )
                
                # Add allocations to results
                for strategy, allocation in effective_allocations.items():
                    # Skip if the strategy isn't available
                    if strategy not in self._sub_strategies:
                        logger.warning(f"Strategy '{strategy}' not available, skipping allocation")
                        continue
                    
                    allocations.append({
                        'timestamp': timestamp,
                        'asset_id': asset_id,
                        'strategy': strategy,
                        'allocation': allocation,
                        'regime': regime,
                        'probability': probability
                    })
        
        # Convert to DataFrame
        allocations_df = pd.DataFrame(allocations)
        
        if len(allocations_df) == 0:
            logger.warning("No allocations generated")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'strategy', 'allocation'
            ])
        
        logger.info(f"Generated {len(allocations_df)} allocation entries")
        return allocations_df
    
    def _execute_sub_strategies(
        self, 
        signals_df: pd.DataFrame, 
        prices_df: pd.DataFrame, 
        allocations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Execute sub-strategies with allocations and combine results.
        
        Args:
            signals_df: DataFrame with signals
            prices_df: DataFrame with prices
            allocations_df: DataFrame with strategy allocations
            
        Returns:
            DataFrame with combined positions
        """
        logger.info("Executing sub-strategies with adaptive weights")
        
        if len(allocations_df) == 0:
            logger.warning("No allocations to execute sub-strategies")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'position', 'target_weight', 'strategy', 'allocation'
            ])
        
        # Get unique timestamps to process
        timestamps = allocations_df['timestamp'].unique()
        
        # Initialize list to store positions from all sub-strategies
        all_positions = []
        
        # Process each timestamp
        for timestamp in timestamps:
            # Filter data for this timestamp
            time_signals = signals_df[signals_df['timestamp'] == timestamp].copy()
            time_prices = prices_df[prices_df['timestamp'] == timestamp].copy()
            time_allocations = allocations_df[allocations_df['timestamp'] == timestamp].copy()
            
            if len(time_signals) == 0 or len(time_prices) == 0 or len(time_allocations) == 0:
                logger.warning(f"Missing data for timestamp {timestamp}, skipping")
                continue
            
            # Get active strategies for this timestamp
            active_strategies = time_allocations[time_allocations['allocation'] > 0]['strategy'].unique()
            
            # Execute each active strategy
            for strategy_name in active_strategies:
                if strategy_name not in self._sub_strategies:
                    logger.warning(f"Strategy '{strategy_name}' not available, skipping execution")
                    continue
                
                strategy = self._sub_strategies[strategy_name]
                
                try:
                    # Run the strategy
                    strategy_positions = strategy.run_strategy(time_signals, time_prices)
                    
                    if len(strategy_positions) == 0:
                        logger.warning(f"No positions generated by {strategy_name} strategy")
                        continue
                    
                    # Add strategy info
                    strategy_positions['strategy'] = strategy_name
                    
                    # Merge with allocations
                    strategy_allocations = time_allocations[time_allocations['strategy'] == strategy_name]
                    strategy_positions = pd.merge(
                        strategy_positions,
                        strategy_allocations[['asset_id', 'allocation']],
                        on='asset_id',
                        how='left'
                    )
                    
                    # Scale positions by allocation
                    strategy_positions['position'] = strategy_positions['position'] * strategy_positions['allocation']
                    strategy_positions['target_weight'] = strategy_positions['target_weight'] * strategy_positions['allocation']
                    
                    # Add to collected positions
                    all_positions.append(strategy_positions)
                    
                    logger.info(f"Executed {strategy_name} strategy for {timestamp}, generated {len(strategy_positions)} positions")
                    
                except Exception as e:
                    logger.error(f"Error executing {strategy_name} strategy: {str(e)}")
        
        # Combine all positions
        if not all_positions:
            logger.warning("No positions generated from sub-strategies")
            return pd.DataFrame(columns=[
                'timestamp', 'asset_id', 'position', 'target_weight', 'strategy', 'allocation'
            ])
        
        combined_positions = pd.concat(all_positions, ignore_index=True)
        
        # Aggregate positions by timestamp and asset
        position_cols = ['timestamp', 'asset_id']
        aggregated_positions = combined_positions.groupby(position_cols).agg({
            'position': 'sum',
            'target_weight': 'sum'
        }).reset_index()
        
        # Add strategy info
        aggregated_positions['strategy'] = 'adaptive_regime'
        aggregated_positions['allocation'] = 1.0
        
        logger.info(f"Generated {len(aggregated_positions)} combined positions from sub-strategies")
        
        return aggregated_positions
    
    def generate_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions by identifying regimes and executing appropriate sub-strategies.
        
        Args:
            signals_df: DataFrame with signals
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with positions [timestamp, asset_id, position, target_weight]
        """
        logger.info("Generating positions based on regime detection")
        
        # 1. Identify current regimes
        regime_df = self._identify_market_regimes(signals_df)
        
        if len(regime_df) == 0:
            logger.warning("No regimes identified, cannot generate positions")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
        
        # 2. Determine strategy allocations
        allocations_df = self._determine_strategy_allocations(regime_df)
        
        if len(allocations_df) == 0:
            logger.warning("No allocations determined, cannot generate positions")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
        
        # 3. Execute sub-strategies with allocations
        positions_df = self._execute_sub_strategies(signals_df, prices_df, allocations_df)
        
        if len(positions_df) == 0:
            logger.warning("No positions generated from sub-strategies")
            return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])
        
        # 4. Apply max allocation scaling
        max_allocation = self.params['max_allocation']
        positions_df['position'] = positions_df['position'] * max_allocation
        positions_df['target_weight'] = positions_df['target_weight'] * max_allocation
        
        # 5. Normalize positions for leverage constraints
        positions_df = normalize_positions(
            positions_df,
            max_leverage=1.0,  # Already scaled by max_allocation
            group_columns=['timestamp'],
            position_column='position',
            weight_column='target_weight'
        )
        
        # Add regime information for reference
        positions_df = pd.merge(
            positions_df,
            regime_df[['timestamp', 'asset_id', 'regime', 'probability']],
            on=['timestamp', 'asset_id'],
            how='left'
        )
        
        return positions_df
    
    def apply_risk_controls(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk controls to positions.
        
        Args:
            positions_df: DataFrame with positions
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with risk-adjusted positions
        """
        logger.info("Applying risk controls to adaptive regime positions")
        
        if len(positions_df) == 0:
            logger.warning("No positions to apply risk controls to")
            return positions_df
        
        # Extract risk parameters
        max_drawdown_pct = self.params['max_drawdown_pct']
        target_volatility = self.params['target_volatility']
        vol_lookback = self.params['vol_lookback']
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
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
register_strategy(AdaptiveRegimeStrategy)


# For backward compatibility with the engine registration system
def run_strategy(signals_df: pd.DataFrame, prices_df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Run the adaptive regime strategy (legacy entry point).
    
    This function maintains compatibility with the old engine registration system.
    
    Args:
        signals_df: DataFrame with signals
        prices_df: DataFrame with prices
        **params: Strategy parameters
        
    Returns:
        DataFrame with positions
    """
    strategy = AdaptiveRegimeStrategy(**params)
    return strategy.run_strategy(signals_df, prices_df)