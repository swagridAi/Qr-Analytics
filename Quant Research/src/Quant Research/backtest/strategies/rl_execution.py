"""
Reinforcement Learning Execution Strategy.

This module implements a strategy that uses reinforcement learning
(specifically PPO - Proximal Policy Optimization) to optimize 
order execution against a micro-price environment.

Note: This is a more advanced strategy that requires additional dependencies:
- stable-baselines3
- gym (or gymnasium)
"""

import logging
import time
import os
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from quant_research.core.models import Signal, Trade
from quant_research.backtest.base import BaseStrategy, StrategyType, register_strategy, StrategyValidationError
from quant_research.backtest import risk
from quant_research.backtest.utils import (
    validate_dataframe,
    ensure_columns,
    normalize_positions,
    save_to_json,
    load_from_json
)

logger = logging.getLogger(__name__)


# Check if required packages are available
RL_AVAILABLE = False
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
    logger.info("RL dependencies available: stable-baselines3 and gym found")
except ImportError:
    logger.warning("RL dependencies not available. Install with: pip install stable-baselines3 gym")



class RLExecutionStrategy(BaseStrategy):
    """
    Reinforcement Learning Execution Strategy.
    
    This strategy uses a trained PPO agent to optimize order execution
    timing and sizing to minimize market impact and maximize execution quality.
    """
    
    # Class variables for strategy metadata
    strategy_type = StrategyType.ML_BASED
    name = "rl_execution"
    description = "Strategy that uses reinforcement learning (PPO) to optimize order execution"
    version = "1.0.0"
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the RL execution strategy.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            "model_dir": "./models/rl",
            "training_steps": 100000,
            "load_existing": True,
            "max_position": 1000,
            "max_steps": 100,
            "inventory_penalty": 0.01,
            "market_impact": 0.0001,
            "slippage_std": 0.0002,
            "seed": 42,
            "policy_kwargs": {"net_arch": [64, 64]},
            "learning_rate": 0.0003,
            "eval_episodes": 5
        }
    
    @classmethod
    def get_parameter_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about strategy parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata
        """
        return {
            "model_dir": {
                "type": "string",
                "description": "Directory to save/load RL models",
                "default": "./models/rl"
            },
            "training_steps": {
                "type": "int",
                "description": "Number of training steps for RL model",
                "default": 100000,
                "min": 1000,
                "max": 10000000
            },
            "load_existing": {
                "type": "bool",
                "description": "Whether to load existing model if available",
                "default": True
            },
            "max_position": {
                "type": "int",
                "description": "Maximum position size for execution",
                "default": 1000,
                "min": 1,
                "max": 100000
            },
            "max_steps": {
                "type": "int",
                "description": "Maximum steps per episode",
                "default": 100,
                "min": 10,
                "max": 1000
            },
            "inventory_penalty": {
                "type": "float",
                "description": "Penalty factor for holding inventory",
                "default": 0.01,
                "min": 0.0001,
                "max": 0.1
            },
            "market_impact": {
                "type": "float",
                "description": "Market impact factor",
                "default": 0.0001,
                "min": 0.00001,
                "max": 0.01
            },
            "slippage_std": {
                "type": "float",
                "description": "Standard deviation of random slippage",
                "default": 0.0002,
                "min": 0.00001,
                "max": 0.01
            },
            "seed": {
                "type": "int",
                "description": "Random seed for reproducibility",
                "default": 42
            },
            "policy_kwargs": {
                "type": "dict",
                "description": "Neural network architecture for policy",
                "default": {"net_arch": [64, 64]}
            },
            "learning_rate": {
                "type": "float",
                "description": "Learning rate for PPO algorithm",
                "default": 0.0003,
                "min": 0.00001,
                "max": 0.01
            },
            "eval_episodes": {
                "type": "int",
                "description": "Number of episodes to evaluate during position generation",
                "default": 5,
                "min": 1,
                "max": 50
            }
        }
    
    def initialize(self) -> None:
        """
        Perform strategy-specific initialization.
        """
        logger.info(
            f"Initializing {self.name} strategy with max_position={self.params['max_position']}, "
            f"inventory_penalty={self.params['inventory_penalty']}"
        )
        
        # Check if RL dependencies are available
        if not RL_AVAILABLE:
            logger.error("RL dependencies not available. Strategy initialization failed.")
            raise StrategyValidationError(
                "Required dependencies (stable-baselines3, gym) are not installed. "
                "Install with: pip install stable-baselines3 gym"
            )
        
        # Initialize instance variables
        self._model = None
        self._prepared_data = None
        self._execution_metrics = []
    
    def _prepare_price_data(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare price data for the RL environment.
        
        Args:
            prices_df: Raw price data
            
        Returns:
            DataFrame with prepared price data
        """
        logger.info("Preparing price data for RL environment")
        
        # Validate input
        is_valid, missing_cols = validate_dataframe(
            prices_df, 
            ['timestamp', 'asset_id'], 
            self.name
        )
        
        if not is_valid:
            logger.warning(f"Invalid prices DataFrame: missing columns {missing_cols}")
            return pd.DataFrame()
        
        # Ensure required columns with defaults
        required_columns = {
            'bid': None,  # Will be derived from close if missing
            'ask': None,  # Will be derived from close if missing
            'bid_volume': 100000,  # Default volume
            'ask_volume': 100000,  # Default volume
            'exchange_id': 'default'
        }
        
        prepared_data = ensure_columns(prices_df, required_columns, self.name)
        
        # Create bid/ask if derived from close
        if 'bid' in prepared_data.columns and prepared_data['bid'].isna().all() and 'close' in prepared_data.columns:
            prepared_data['bid'] = prepared_data['close'] * 0.9999
            logger.info("Created 'bid' column from 'close'")
        
        if 'ask' in prepared_data.columns and prepared_data['ask'].isna().all() and 'close' in prepared_data.columns:
            prepared_data['ask'] = prepared_data['close'] * 1.0001
            logger.info("Created 'ask' column from 'close'")
        
        # Verify bid/ask columns after potential derivation
        if 'bid' not in prepared_data.columns or 'ask' not in prepared_data.columns:
            logger.error("Unable to derive bid/ask prices, required for RL environment")
            return pd.DataFrame()
        
        # Create column aliases 
        if 'bid_price' not in prepared_data.columns:
            prepared_data['bid_price'] = prepared_data['bid']
        
        if 'ask_price' not in prepared_data.columns:
            prepared_data['ask_price'] = prepared_data['ask']
        
        # Calculate derived features
        prepared_data['mid_price'] = (prepared_data['bid_price'] + prepared_data['ask_price']) / 2
        prepared_data['spread'] = prepared_data['ask_price'] - prepared_data['bid_price']
        
        # Sort by timestamp
        prepared_data = prepared_data.sort_values('timestamp')
        
        logger.info(f"Prepared {len(prepared_data)} price data points for RL environment")
        
        # Store for later use
        self._prepared_data = prepared_data
        
        return prepared_data
    
    def _train_rl_model(self, price_data: pd.DataFrame) -> Optional[PPO]:
        """
        Train or load an RL model for execution optimization.
        
        Args:
            price_data: Prepared price data
            
        Returns:
            Trained PPO model or None if error
        """
        if not RL_AVAILABLE:
            logger.error("RL dependencies not available. Cannot train model.")
            return None
        
        if price_data.empty:
            logger.error("No price data available for training")
            return None
        
        # Extract parameters
        model_dir = self.params['model_dir']
        model_path = os.path.join(model_dir, f'ppo_execution_default.zip')
        training_steps = self.params['training_steps']
        policy_kwargs = self.params['policy_kwargs']
        learning_rate = self.params['learning_rate']
        load_existing = self.params['load_existing']
        seed = self.params['seed']
        
        # Check for asset-specific model path
        if 'asset_id' in price_data.columns:
            asset_id = price_data['asset_id'].iloc[0]
            model_path = os.path.join(model_dir, f'ppo_execution_{asset_id}.zip')
        
        # Check if we should load existing model
        if load_existing and os.path.exists(model_path):
            logger.info(f"Loading existing RL model from {model_path}")
            try:
                model = PPO.load(model_path)
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                logger.info("Will train a new model instead")
        
        logger.info(f"Training new RL model with {training_steps} steps")
        
        # Create environment parameters
        env_params = {
            'max_position': self.params['max_position'],
            'max_steps': self.params['max_steps'],
            'inventory_penalty': self.params['inventory_penalty'],
            'market_impact': self.params['market_impact'],
            'slippage_std': self.params['slippage_std'],
            'seed': seed
        }
        
        # Create environment creator function for vectorized environment
        def make_env():
            env = MicroPriceEnv(price_data, **env_params)
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env])
        
        # Initialize model
        model = PPO(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            verbose=1,
            seed=seed
        )
        
        # Train the model
        start_time = time.time()
        try:
            model.learn(total_timesteps=training_steps)
            training_duration = time.time() - start_time
            logger.info(f"Model training completed in {training_duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return None
        
        # Save the model
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
        
        return model
    
    def _generate_positions_with_rl(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions using the trained RL model.
        
        Args:
            price_data: Prepared price data
            
        Returns:
            DataFrame with positions
        """
        if self._model is None:
            logger.error("No RL model available. Cannot generate positions.")
            return pd.DataFrame()
        
        if price_data.empty:
            logger.error("No price data available for position generation")
            return pd.DataFrame()
        
        logger.info("Generating positions using RL model")
        
        # Extract parameters
        eval_episodes = self.params['eval_episodes']
        max_position = self.params['max_position']
        
        # Environment parameters
        env_params = {
            'max_position': max_position,
            'max_steps': self.params['max_steps'],
            'inventory_penalty': self.params['inventory_penalty'],
            'market_impact': self.params['market_impact'],
            'slippage_std': self.params['slippage_std'],
            'seed': self.params['seed']
        }
        
        # Create environment for evaluation
        env = MicroPriceEnv(price_data, **env_params)
        
        # Lists to store positions and metrics
        all_positions = []
        all_metrics = []
        
        # Run multiple episodes
        for episode in range(eval_episodes):
            logger.info(f"Evaluating episode {episode+1}/{eval_episodes}")
            
            obs = env.reset()
            episode_positions = []
            done = False
            
            step = 0
            while not done:
                # Get action from model
                action, _ = self._model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                # Record position for this step
                if step < len(price_data):
                    # Get timestamp and asset info from price data
                    price_idx = min(env.current_index - 1, len(price_data) - 1)
                    if price_idx >= 0:
                        row = price_data.iloc[price_idx]
                        timestamp = row['timestamp']
                        asset_id = row.get('asset_id', 'default')
                        exchange_id = row.get('exchange_id', 'default')
                        
                        # Calculate position change from action
                        action_value = float(action[0])
                        position_change = -action_value * env.current_position
                        
                        # Only record if actual position change
                        if abs(position_change) > 0.0001:
                            episode_positions.append({
                                'timestamp': timestamp,
                                'asset_id': asset_id,
                                'exchange_id': exchange_id,
                                'position': position_change,
                                'target_weight': position_change / max_position,
                                'episode': episode,
                                'step': step,
                                'remaining_position': env.current_position
                            })
                
                step += 1
            
            # Add episode metrics
            episode_metrics = {
                'episode': episode,
                'final_position': env.current_position,
                'vwap': info['vwap'],
                'steps_taken': info['steps_taken'],
                'trades_executed': info['trades_executed']
            }
            
            all_metrics.append(episode_metrics)
            all_positions.extend(episode_positions)
        
        # Store metrics for later analysis
        self._execution_metrics = all_metrics
        
        # Calculate aggregate metrics
        if all_metrics:
            avg_vwap = np.mean([m['vwap'] for m in all_metrics])
            avg_trades = np.mean([m['trades_executed'] for m in all_metrics])
            avg_remaining = np.mean([m['final_position'] for m in all_metrics])
            
            logger.info(f"RL Model Performance: VWAP={avg_vwap:.4f}, "
                        f"Trades/episode={avg_trades:.1f}, "
                        f"Remaining={avg_remaining:.1f}/{max_position}")
        
        # Convert to DataFrame
        positions_df = pd.DataFrame(all_positions)
        
        if len(positions_df) == 0:
            logger.warning("No positions generated by RL model")
            return pd.DataFrame()
        
        # Aggregate positions by timestamp and asset
        position_cols = ['timestamp', 'asset_id', 'exchange_id', 'position', 'target_weight']
        if all(col in positions_df.columns for col in position_cols):
            result = positions_df[position_cols].groupby(['timestamp', 'asset_id', 'exchange_id']).sum().reset_index()
            logger.info(f"Generated {len(result)} aggregated position entries using RL model")
            return result
        else:
            logger.error(f"Missing columns in positions DataFrame: {set(position_cols) - set(positions_df.columns)}")
            return pd.DataFrame()
    
    def generate_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions using RL for execution optimization.
        
        Args:
            signals_df: DataFrame with signals (not heavily used for RL execution)
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with positions [timestamp, asset_id, position, target_weight]
        """
        logger.info("Generating positions with RL execution strategy")
        
        # Check RL dependencies
        if not RL_AVAILABLE:
            logger.error("RL dependencies not available. Strategy execution failed.")
            return pd.DataFrame()
        
        # 1. Prepare price data for RL environment
        prepared_data = self._prepare_price_data(prices_df)
        
        if prepared_data.empty:
            logger.error("Failed to prepare price data for RL environment")
            return pd.DataFrame()
        
        # 2. Train or load RL model
        if self._model is None:
            self._model = self._train_rl_model(prepared_data)
        
        if self._model is None:
            logger.error("Failed to train/load RL model")
            return pd.DataFrame()
        
        # 3. Generate positions using RL model
        positions_df = self._generate_positions_with_rl(prepared_data)
        
        if positions_df.empty:
            logger.warning("No positions generated by RL model")
            return pd.DataFrame()
        
        return positions_df
    
    def apply_risk_controls(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk controls to positions.
        
        For RL execution strategy, most risk controls are embedded in the environment,
        so this mainly enforces position limits and does basic sanity checks.
        
        Args:
            positions_df: DataFrame with positions
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with risk-adjusted positions
        """
        logger.info("Applying risk controls to RL execution positions")
        
        if len(positions_df) == 0:
            logger.warning("No positions to apply risk controls to")
            return positions_df
        
        # Extract parameters
        max_position = self.params['max_position']
        
        # Make a copy to avoid modifying the original
        result = positions_df.copy()
        
        # Clip position sizes to max_position
        result['position'] = result['position'].clip(-max_position, max_position)
        
        # Recalculate target weights
        result['target_weight'] = result['position'] / max_position
        
        return result
    
    def save_execution_metrics(self, filepath: str) -> bool:
        """
        Save execution metrics to a file.
        
        Args:
            filepath: Path to save metrics
            
        Returns:
            True if successful, False otherwise
        """
        if not self._execution_metrics:
            logger.warning("No execution metrics to save")
            return False
        
        return save_to_json(self._execution_metrics, filepath)
    
    def load_execution_metrics(self, filepath: str) -> bool:
        """
        Load execution metrics from a file.
        
        Args:
            filepath: Path to load metrics from
            
        Returns:
            True if successful, False otherwise
        """
        metrics = load_from_json(filepath)
        if metrics is None:
            return False
        
        self._execution_metrics = metrics
        logger.info(f"Loaded {len(metrics)} execution metric records")
        return True


# Register the strategy with the registry
register_strategy(RLExecutionStrategy)


# For backward compatibility with the engine registration system
def run_strategy(signals_df: pd.DataFrame, prices_df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Run the RL execution strategy (legacy entry point).
    
    This function maintains compatibility with the old engine registration system.
    
    Args:
        signals_df: DataFrame with signals
        prices_df: DataFrame with prices
        **params: Strategy parameters
        
    Returns:
        DataFrame with positions
    """
    try:
        strategy = RLExecutionStrategy(**params)
        return strategy.run_strategy(signals_df, prices_df)
    except StrategyValidationError as e:
        logger.error(f"Strategy validation error: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'asset_id', 'position', 'target_weight'])