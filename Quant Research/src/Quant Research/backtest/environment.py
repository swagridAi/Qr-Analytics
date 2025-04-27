"""
Reinforcement Learning Environments for Quantitative Research.

This module provides gym-compatible environments that simulate
various market dynamics for training reinforcement learning agents.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Type
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os
import json

from quant_research.backtest.utils import validate_dataframe, ensure_columns

logger = logging.getLogger(__name__)

# Check if required packages are available
RL_AVAILABLE = False
try:
    import gym
    from gym import spaces
    RL_AVAILABLE = True
    logger.info("Gym package available for RL environments")
except ImportError:
    logger.warning("Gym package not available. Install with: pip install gym")


class MarketEnvironment(ABC):
    """
    Abstract base class for market simulation environments.
    
    This provides a common interface for different environment types
    used in reinforcement learning for trading strategies.
    """
    
    @abstractmethod
    def reset(self):
        """Reset the environment to initial state."""
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, done, info
        """
        pass
    
    @abstractmethod
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        pass
    
    @classmethod
    def check_dependencies(cls) -> bool:
        """
        Check if required dependencies are installed.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        return RL_AVAILABLE
    
    @staticmethod
    def prepare_price_data(prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare price data for use in environment.
        
        Args:
            prices_df: Raw price data
            
        Returns:
            Processed price data suitable for environment
        """
        if prices_df is None or prices_df.empty:
            logger.warning("Empty price data provided")
            return pd.DataFrame()
        
        # Validate basic columns
        is_valid, missing_cols = validate_dataframe(
            prices_df,
            ['timestamp', 'asset_id'],
            'MarketEnvironment'
        )
        
        if not is_valid:
            logger.warning(f"Invalid price data: missing columns {missing_cols}")
            return pd.DataFrame()
        
        # Ensure required columns with defaults
        required_columns = {
            'bid': None,  # Will be derived from close if missing
            'ask': None,  # Will be derived from close if missing
            'bid_volume': 100000,  # Default volume
            'ask_volume': 100000,  # Default volume
            'exchange_id': 'default'
        }
        
        prepared_data = ensure_columns(prices_df, required_columns, 'MarketEnvironment')
        
        # Create bid/ask if missing
        if 'close' in prepared_data.columns:
            if 'bid' in prepared_data.columns and prepared_data['bid'].isna().all():
                prepared_data['bid'] = prepared_data['close'] * 0.9999
            
            if 'ask' in prepared_data.columns and prepared_data['ask'].isna().all():
                prepared_data['ask'] = prepared_data['close'] * 1.0001
        
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
        
        logger.info(f"Prepared {len(prepared_data)} price data points for environment")
        return prepared_data


class MicroPriceEnv(gym.Env, MarketEnvironment):
    """
    A gym environment that simulates order book dynamics for execution optimization.
    
    This environment models the micro-price dynamics of a market, allowing an
    RL agent to learn optimal execution strategies.
    
    Features:
    - Realistic market impact modeling
    - Slippage simulation
    - Bid-ask spread dynamics
    - Inventory risk management
    - Time pressure
    """
    
    metadata = {'render.modes': ['human', 'log']}
    
    def __init__(self, price_data: pd.DataFrame, **params):
        """
        Initialize the environment with price data.
        
        Args:
            price_data: DataFrame with price data including bid, ask, volumes
            **params: Environment parameters
        """
        if not RL_AVAILABLE:
            raise ImportError(
                "Gym package is required for MicroPriceEnv but not available. "
                "Install with: pip install gym"
            )
        
        super(MicroPriceEnv, self).__init__()
        
        # Validate and store price data
        if price_data is None or price_data.empty:
            raise ValueError("Price data cannot be None or empty")
        
        required_columns = ['timestamp', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'mid_price']
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        
        if missing_columns:
            # Try to prepare the data
            logger.warning(f"Missing columns in price data: {missing_columns}")
            price_data = self.prepare_price_data(price_data)
            
            # Check again
            missing_columns = [col for col in required_columns if col not in price_data.columns]
            if missing_columns:
                raise ValueError(f"Price data missing required columns after preparation: {missing_columns}")
        
        self.price_data = price_data
        
        # Extract parameters with defaults
        self.max_position = params.get('max_position', 100)
        self.max_steps = params.get('max_steps', 100)
        self.inventory_penalty = params.get('inventory_penalty', 0.01)
        self.order_size = params.get('order_size', 10)
        self.market_impact = params.get('market_impact', 0.0001)
        self.slippage_std = params.get('slippage_std', 0.0002)
        self.price_history_len = params.get('price_history_len', 10)
        self.initial_cash = params.get('initial_cash', 1000000)
        self.reward_scaling = params.get('reward_scaling', 1.0)
        self.terminal_penalty_factor = params.get('terminal_penalty_factor', 10.0)
        self.seed_value = params.get('seed', None)
        
        # Time limit penalties
        self.time_decay = params.get('time_decay', True)
        self.time_decay_factor = params.get('time_decay_factor', 2.0)
        
        # Initialize state variables (will be reset properly in reset())
        self.current_step = 0
        self.current_position = 0
        self.current_cash = self.initial_cash
        self.executed_trades = []
        self.current_index = 0
        self.done = False
        self.price_history = []
        self.episode_rewards = []
        self.total_reward = 0
        
        # Define action space
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        # Define observation space
        feature_count = 2 + self.price_history_len * 4  # time, position, and price features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32
        )
        
        # Seed the environment if provided
        if self.seed_value is not None:
            self.seed(self.seed_value)
    
    def seed(self, seed=None):
        """Set the random seed for the environment."""
        np.random.seed(seed)
        return [seed]
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Reset state variables
        self.current_step = 0
        self.current_position = self.max_position
        self.current_cash = self.initial_cash
        self.executed_trades = []
        self.done = False
        self.episode_rewards = []
        self.total_reward = 0
        
        # Start at a random point in the price data
        max_start = max(0, len(self.price_data) - self.max_steps - 1)
        self.current_index = np.random.randint(0, max_start) if max_start > 0 else 0
        
        # Initialize price history
        self.price_history = []
        for i in range(min(self.price_history_len, self.max_steps)):
            idx = max(0, self.current_index - self.price_history_len + i)
            if idx < len(self.price_data):
                row = self.price_data.iloc[idx]
                self.price_history.append([
                    row['mid_price'], 
                    row['spread'], 
                    row['bid_volume'], 
                    row['ask_volume']
                ])
            else:
                # Fallback if not enough data
                self.price_history.append([0, 0, 0, 0])
        
        # Pad if needed
        while len(self.price_history) < self.price_history_len:
            self.price_history.append([0, 0, 0, 0])
        
        return self._get_observation()
    
    def _get_current_price_data(self):
        """
        Get price data for current step.
        
        Returns:
            Price data row
        """
        if self.current_index < len(self.price_data):
            return self.price_data.iloc[self.current_index]
        else:
            # Fallback for out of range
            logger.warning(f"Price index {self.current_index} out of range, using last available price")
            return self.price_data.iloc[-1].copy()
    
    def _get_observation(self):
        """
        Construct the observation vector.
        
        Returns:
            Observation array
        """
        # Normalized time
        normalized_time = self.current_step / self.max_steps
        
        # Normalized position
        normalized_position = self.current_position / self.max_position
        
        # Flatten price history
        price_features = np.array(self.price_history).flatten()
        
        # Combine all features
        observation = np.concatenate([
            [normalized_time, normalized_position], 
            price_features
        ])
        
        return observation
    
    def _calculate_reward(self, executed_qty, execution_price):
        """
        Calculate reward for the current action.
        
        The reward function balances execution quality against inventory risk.
        
        Args:
            executed_qty: Quantity executed (negative for sell)
            execution_price: Execution price
            
        Returns:
            Reward value
        """
        # Skip reward calculation if no execution
        if executed_qty == 0:
            # Still apply inventory penalty
            time_factor = (self.current_step / self.max_steps) ** self.time_decay_factor if self.time_decay else 1.0
            inventory_penalty = -self.inventory_penalty * self.current_position * time_factor
            return inventory_penalty * self.reward_scaling
        
        # Execution quality component (vs. mid price)
        current_data = self._get_current_price_data()
        mid_price = current_data['mid_price']
        
        # For a sell order (negative qty): higher price is better
        # For a buy order (positive qty): lower price is better
        price_improvement = (execution_price - mid_price) * (-np.sign(executed_qty))
        
        # Scale price improvement
        price_reward = price_improvement * abs(executed_qty)
        
        # Inventory risk penalty (more severe as we near the end)
        time_factor = (self.current_step / self.max_steps) ** self.time_decay_factor if self.time_decay else 1.0
        inventory_penalty = -self.inventory_penalty * self.current_position * time_factor
        
        # Final reward
        reward = (price_reward + inventory_penalty) * self.reward_scaling
        
        # Terminal bonus/penalty for remaining inventory
        if self.done and self.current_position > 0:
            # Penalty for not completing the order
            terminal_penalty = -self.current_position * self.inventory_penalty * self.terminal_penalty_factor
            reward += terminal_penalty * self.reward_scaling
        
        return reward
    
    def _calculate_market_impact(self, qty):
        """
        Calculate market impact of order.
        
        Market impact increases with order size relative to available volume
        and affects the execution price.
        
        Args:
            qty: Quantity to execute (negative for sell)
            
        Returns:
            Price impact
        """
        current_data = self._get_current_price_data()
        available_volume = current_data['bid_volume'] if qty < 0 else current_data['ask_volume']
        
        # Impact increases with order size relative to available volume
        volume_ratio = min(1.0, abs(qty) / available_volume)
        impact = self.market_impact * volume_ratio * abs(qty)
        
        return impact
    
    def _execute_order(self, qty):
        """
        Execute an order with market impact and slippage.
        
        Args:
            qty: Quantity to execute (negative for sell, positive for buy)
            
        Returns:
            execution_price: Price at which the order was executed
        """
        # Get current prices
        current_data = self._get_current_price_data()
        
        # Base price (bid for sell, ask for buy)
        base_price = current_data['bid_price'] if qty < 0 else current_data['ask_price']
        
        # Calculate market impact (moves price against you)
        impact = self._calculate_market_impact(qty)
        
        # Add random slippage
        slippage = np.random.normal(0, self.slippage_std)
        
        # Calculate execution price
        if qty < 0:  # Sell order
            execution_price = base_price - impact + slippage
        else:  # Buy order
            execution_price = base_price + impact + slippage
        
        # Update cash and position
        cash_change = execution_price * abs(qty)
        self.current_cash += cash_change if qty < 0 else -cash_change
        self.current_position -= qty  # Sell decreases position
        
        # Record the trade
        self.executed_trades.append({
            'step': self.current_step,
            'qty': qty,
            'price': execution_price,
            'impact': impact,
            'slippage': slippage,
            'timestamp': current_data.get('timestamp', None)
        })
        
        return execution_price
    
    def step(self, action):
        """
        Take a step in the environment by executing part of the order.
        
        Args:
            action: Percentage of remaining position to execute (0-1)
            
        Returns:
            observation: New state observation
            reward: Reward for this action
            done: Whether the episode is done
            info: Additional information
        """
        # Convert action to executed quantity
        action_value = float(action[0])  # Extract from array
        action_value = np.clip(action_value, 0, 1)  # Ensure in range [0, 1]
        
        # Calculate quantity to execute
        qty_to_execute = -action_value * self.current_position  # Negative for sell
        qty_to_execute = np.clip(qty_to_execute, -self.current_position, 0)
        
        # Execute the order if quantity is non-zero
        if abs(qty_to_execute) > 0.0001:  # Small threshold to avoid numerical issues
            execution_price = self._execute_order(qty_to_execute)
            reward = self._calculate_reward(qty_to_execute, execution_price)
        else:
            reward = self._calculate_reward(0, 0)  # Still apply inventory penalty
        
        # Update reward tracking
        self.episode_rewards.append(reward)
        self.total_reward += reward
        
        # Update step counter
        self.current_step += 1
        
        # Update price history
        current_data = self._get_current_price_data()
        self.price_history.append([
            current_data['mid_price'], 
            current_data['spread'], 
            current_data['bid_volume'], 
            current_data['ask_volume']
        ])
        self.price_history = self.price_history[-self.price_history_len:]
        
        # Check if episode is done
        self.done = (self.current_step >= self.max_steps) or (self.current_position <= 0)
        
        # Move to next price data point
        self.current_index += 1
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        # Prepare info dict
        info = {
            'position_left': self.current_position,
            'vwap': metrics['vwap'],
            'steps_taken': self.current_step,
            'trades_executed': len(self.executed_trades),
            'total_reward': self.total_reward,
            'execution_quality': metrics['execution_quality'],
            'completion_pct': metrics['completion_pct']
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics for the current episode.
        
        Returns:
            Dictionary of metrics
        """
        # Default values for metrics
        metrics = {
            'vwap': 0.0,
            'execution_quality': 0.0,
            'completion_pct': 0.0,
            'total_executed': 0.0
        }
        
        # Calculate VWAP if trades exist
        if self.executed_trades:
            total_qty = sum(abs(t['qty']) for t in self.executed_trades)
            if total_qty > 0:
                vwap = sum(t['price'] * abs(t['qty']) for t in self.executed_trades) / total_qty
                metrics['vwap'] = vwap
                
                # Calculate execution quality vs average mid price
                avg_mid = np.mean([t['price'] for t in self.executed_trades])
                metrics['execution_quality'] = (vwap / avg_mid - 1) * 100 * (-1)  # For sell orders
        
        # Calculate completion percentage
        initial_position = self.max_position
        remaining_position = self.current_position
        executed_position = initial_position - remaining_position
        metrics['total_executed'] = executed_position
        metrics['completion_pct'] = (executed_position / initial_position) * 100 if initial_position > 0 else 0
        
        return metrics
    
    def render(self, mode='human'):
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ('human' or 'log')
        """
        if mode == 'human':
            metrics = self._calculate_metrics()
            
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"Position remaining: {self.current_position}/{self.max_position}")
            print(f"Completion: {metrics['completion_pct']:.1f}%")
            print(f"VWAP: {metrics['vwap']:.4f}")
            print(f"Trades executed: {len(self.executed_trades)}")
            print(f"Reward: {self.total_reward:.2f}")
            
            # Show recent trades
            if self.executed_trades:
                print("\nRecent trades:")
                for trade in self.executed_trades[-min(3, len(self.executed_trades)):]:
                    print(f"  Step {trade['step']}: {trade['qty']:.2f} @ {trade['price']:.4f}")
        
        elif mode == 'log':
            metrics = self._calculate_metrics()
            logger.info(
                f"Step {self.current_step}/{self.max_steps}, "
                f"Position: {self.current_position}/{self.max_position}, "
                f"Completion: {metrics['completion_pct']:.1f}%, "
                f"VWAP: {metrics['vwap']:.4f}, "
                f"Trades: {len(self.executed_trades)}, "
                f"Reward: {self.total_reward:.2f}"
            )
    
    def get_execution_summary(self):
        """
        Get a summary of the execution performance.
        
        Returns:
            Dictionary with execution summary
        """
        metrics = self._calculate_metrics()
        
        summary = {
            'initial_position': self.max_position,
            'final_position': self.current_position,
            'total_executed': metrics['total_executed'],
            'completion_pct': metrics['completion_pct'],
            'vwap': metrics['vwap'],
            'num_trades': len(self.executed_trades),
            'avg_trade_size': metrics['total_executed'] / len(self.executed_trades) if self.executed_trades else 0,
            'total_reward': self.total_reward,
            'execution_quality': metrics['execution_quality'],
            'steps_taken': self.current_step
        }
        
        return summary


class OrderBookEnv(gym.Env, MarketEnvironment):
    """
    A gym environment that simulates a full order book for market making and HFT.
    
    This environment provides a more detailed simulation of order book dynamics
    suitable for high-frequency trading strategies and market making.
    
    Note: This is a more advanced environment that requires order book data.
    """
    
    metadata = {'render.modes': ['human', 'log']}
    
    def __init__(self, orderbook_data: pd.DataFrame, **params):
        """
        Initialize the environment with order book data.
        
        Args:
            orderbook_data: DataFrame with order book data
            **params: Environment parameters
        """
        if not RL_AVAILABLE:
            raise ImportError(
                "Gym package is required for OrderBookEnv but not available. "
                "Install with: pip install gym"
            )
        
        super(OrderBookEnv, self).__init__()
        
        logger.info("Initializing OrderBookEnv (experimental)")
        logger.warning("OrderBookEnv is experimental and not fully implemented")
        
        # Store parameters for later implementation
        self.params = params
        
        # Define minimal action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    
    def reset(self):
        """Reset the environment to initial state."""
        # Placeholder implementation
        return np.zeros(10, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, done, info
        """
        # Placeholder implementation
        return np.zeros(10, dtype=np.float32), 0.0, True, {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        logger.info("OrderBookEnv render (not implemented)")


def create_environment(
    env_type: str,
    price_data: pd.DataFrame,
    **params
) -> Optional[MarketEnvironment]:
    """
    Factory function to create environment instances.
    
    Args:
        env_type: Type of environment to create ('micro_price' or 'order_book')
        price_data: Price data for the environment
        **params: Environment parameters
        
    Returns:
        Environment instance or None if error
    """
    if not MarketEnvironment.check_dependencies():
        logger.error("Required dependencies missing. Cannot create environment.")
        return None
    
    if price_data is None or price_data.empty:
        logger.error("No price data provided. Cannot create environment.")
        return None
    
    # Prepare price data
    prepared_data = MarketEnvironment.prepare_price_data(price_data)
    
    if prepared_data.empty:
        logger.error("Failed to prepare price data. Cannot create environment.")
        return None
    
    # Create environment based on type
    try:
        if env_type.lower() == 'micro_price':
            return MicroPriceEnv(prepared_data, **params)
        elif env_type.lower() == 'order_book':
            return OrderBookEnv(prepared_data, **params)
        else:
            logger.error(f"Unknown environment type: {env_type}")
            return None
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        return None