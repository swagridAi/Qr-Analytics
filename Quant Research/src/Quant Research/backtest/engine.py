"""
Backtesting engine for quantitative research.

This module provides the core execution engine that orchestrates the backtesting process:
- Coordinates data providers, analytics, strategies, and risk management
- Manages the simulation timeline and event processing
- Calculates and reports performance metrics
- Handles configuration and parameter management
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Type, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import importlib
import pkgutil
import warnings
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import project components
from quant_research.core.models import Signal, Trade, PriceBar
from quant_research.core.storage import save_dataframe, load_dataframe
from quant_research.core import event_bus
from quant_research.backtest.base import BaseStrategy, StrategyRegistry, StrategyError
from quant_research.backtest import risk
from quant_research.backtest import fees
from quant_research.backtest.utils import (
    validate_dataframe,
    normalize_positions,
    calculate_returns,
    calculate_metrics,
    calculate_turnover,
    detect_survivorship_bias,
    detect_lookahead_bias,
    save_to_json,
    load_from_json
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine responsible for orchestrating the simulation.
    
    This class coordinates all components of the backtesting process:
    - Data management
    - Signal processing
    - Strategy execution
    - Risk management
    - Performance calculation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the backtest engine with configuration.
        
        Args:
            config: Dictionary with engine configuration
        """
        # Default configuration
        self.default_config = {
            "start_date": None,  # Start date for backtest
            "end_date": None,    # End date for backtest 
            "data_dir": "./data",  # Directory for data files
            "results_dir": "./results",  # Directory for results
            "signals_file": "signals.parquet",  # Signals data file
            "prices_file": "prices.parquet",  # Price data file
            "output_file": "performance.csv",  # Output file for performance
            "strategy": "momentum",  # Default strategy to use
            "strategy_params": {},  # Strategy-specific parameters
            "risk_params": {  # Risk management parameters
                "max_leverage": 1.0,
                "max_position_size": 0.1,
                "stop_loss_pct": None,
                "target_volatility": None
            },
            "fee_model": "fixed",  # Fee model to use
            "fee_params": {  # Fee model parameters
                "commission_pct": 0.001,
                "min_commission": 1.0
            },
            "execution_model": "simple",  # Execution model to use
            "execution_params": {  # Execution model parameters
                "slippage_bps": 5
            },
            "portfolio_initial_value": 1000000,  # Initial portfolio value
            "benchmark": "SPY",  # Benchmark for comparison
            "frequency": "daily",  # Data frequency
            "parallel": False,  # Whether to use parallel processing
            "max_workers": 4,  # Maximum number of parallel workers
            "debug": False  # Whether to enable debug logging
        }
        
        # Apply config
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize state
        self._reset_state()
        
        # Setup logging
        if self.config["debug"]:
            logging.getLogger("quant_research").setLevel(logging.DEBUG)
        
        # Ensure directories exist
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"BacktestEngine initialized with {self.config['strategy']} strategy")
    
    def _reset_state(self):
        """Reset the internal state for a fresh backtest."""
        self.signals_df = pd.DataFrame()
        self.prices_df = pd.DataFrame()
        self.positions_df = pd.DataFrame()
        self.trades_df = pd.DataFrame()
        self.performance_df = pd.DataFrame()
        
        self.current_date = None
        self.backtest_id = str(uuid.uuid4())[:8]
        self.strategy_instance = None
        self.execution_time = 0
        self.is_running = False
        self.has_run = False
        self.backtest_metrics = {}
    
    def _initialize_components(self):
        """Initialize the engine components based on configuration."""
        # Load or create strategy instance
        strategy_name = self.config["strategy"]
        strategy_params = self.config["strategy_params"]
        
        self.strategy_instance = StrategyRegistry.create_strategy(
            strategy_name, **strategy_params
        )
        
        if self.strategy_instance is None:
            logger.warning(
                f"Strategy '{strategy_name}' not found in registry. "
                "Will use legacy registration system."
            )
    
    def _load_data(self) -> Tuple[bool, str]:
        """
        Load price and signal data for the backtest.
        
        Returns:
            (success, message): Whether loading was successful and info message
        """
        logger.info("Loading data for backtest")
        
        try:
            # Determine file paths
            prices_path = os.path.join(self.config["data_dir"], self.config["prices_file"])
            signals_path = os.path.join(self.config["data_dir"], self.config["signals_file"])
            
            # Check if files exist
            if not os.path.exists(prices_path):
                logger.error(f"Price data file not found: {prices_path}")
                return False, f"Price data file not found: {prices_path}"
            
            if not os.path.exists(signals_path):
                logger.warning(f"Signals data file not found: {signals_path}")
                logger.info("Will proceed with price data only")
            
            # Load price data
            self.prices_df = load_dataframe(prices_path)
            if self.prices_df.empty:
                logger.error("Failed to load price data or file is empty")
                return False, "Failed to load price data"
            
            # Ensure required columns
            required_price_cols = ['timestamp', 'asset_id', 'close']
            missing_price_cols = [col for col in required_price_cols if col not in self.prices_df.columns]
            
            if missing_price_cols:
                logger.error(f"Price data missing required columns: {missing_price_cols}")
                return False, f"Price data missing required columns: {missing_price_cols}"
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.prices_df['timestamp']):
                logger.info("Converting timestamp column to datetime")
                self.prices_df['timestamp'] = pd.to_datetime(self.prices_df['timestamp'])
            
            # Load signals data if available
            if os.path.exists(signals_path):
                self.signals_df = load_dataframe(signals_path)
                if self.signals_df.empty:
                    logger.warning("Signals file is empty")
                else:
                    # Check required columns
                    required_signal_cols = ['timestamp', 'asset_id', 'strategy']
                    missing_signal_cols = [col for col in required_signal_cols 
                                          if col not in self.signals_df.columns]
                    
                    if missing_signal_cols:
                        logger.warning(f"Signal data missing columns: {missing_signal_cols}")
                    
                    # Convert timestamp to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(self.signals_df['timestamp']):
                        self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'])
            
            # Apply date filters if specified
            if self.config["start_date"]:
                start_date = pd.to_datetime(self.config["start_date"])
                self.prices_df = self.prices_df[self.prices_df['timestamp'] >= start_date]
                if not self.signals_df.empty:
                    self.signals_df = self.signals_df[self.signals_df['timestamp'] >= start_date]
            
            if self.config["end_date"]:
                end_date = pd.to_datetime(self.config["end_date"])
                self.prices_df = self.prices_df[self.prices_df['timestamp'] <= end_date]
                if not self.signals_df.empty:
                    self.signals_df = self.signals_df[self.signals_df['timestamp'] <= end_date]
            
            # Check if we have data after filtering
            if self.prices_df.empty:
                logger.error("No price data available after applying date filters")
                return False, "No price data available after applying date filters"
            
            # Log data stats
            logger.info(f"Loaded {len(self.prices_df)} price bars from {prices_path}")
            if not self.signals_df.empty:
                logger.info(f"Loaded {len(self.signals_df)} signals from {signals_path}")
            
            # Get date range for the backtest
            min_date = self.prices_df['timestamp'].min()
            max_date = self.prices_df['timestamp'].max()
            
            logger.info(f"Data spans from {min_date} to {max_date}")
            
            # Set current date to min date
            self.current_date = min_date
            
            # Check for bias
            self._check_data_bias()
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            logger.exception(f"Error loading data: {str(e)}")
            return False, f"Error loading data: {str(e)}"
    
    def _check_data_bias(self):
        """Check for common biases in the backtesting data."""
        # Check for lookahead bias
        if not self.signals_df.empty:
            lookahead_check = detect_lookahead_bias(self.signals_df, self.prices_df)
            if lookahead_check['potential_bias']:
                logger.warning(f"Potential lookahead bias detected: {lookahead_check['analysis']}")
        
        # Check for survivorship bias
        survivorship_check = detect_survivorship_bias(self.prices_df, self.prices_df)
        if survivorship_check['potential_bias']:
            logger.warning(f"Potential survivorship bias detected: {survivorship_check['analysis']}")
    
    def _apply_fees(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transaction fees to positions.
        
        Args:
            positions_df: DataFrame with position changes
            
        Returns:
            DataFrame with fees applied
        """
        fee_model = self.config["fee_model"]
        fee_params = self.config["fee_params"]
        
        logger.info(f"Applying {fee_model} fee model to positions")
        
        try:
            # Apply appropriate fee model
            if fee_model == "fixed":
                result = fees.apply_fixed_commission(
                    positions_df,
                    fee_params.get("commission_pct", 0.001),
                    fee_params.get("min_commission", 1.0)
                )
            elif fee_model == "tiered":
                result = fees.apply_tiered_commission(
                    positions_df,
                    fee_params.get("tiers", [])
                )
            elif fee_model == "exchange":
                result = fees.apply_exchange_fees(
                    positions_df,
                    fee_params.get("exchange_fees", {})
                )
            else:
                logger.warning(f"Unknown fee model '{fee_model}', using default fixed fees")
                result = fees.apply_fixed_commission(positions_df, 0.001, 1.0)
            
            return result
        
        except Exception as e:
            logger.error(f"Error applying fees: {str(e)}")
            # Return original positions if error
            return positions_df
    
    def _apply_slippage(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply execution slippage to positions.
        
        Args:
            positions_df: DataFrame with position changes
            
        Returns:
            DataFrame with slippage applied
        """
        execution_model = self.config["execution_model"]
        execution_params = self.config["execution_params"]
        
        logger.info(f"Applying {execution_model} execution model with slippage")
        
        try:
            # Apply appropriate execution model
            if execution_model == "simple":
                result = fees.apply_simple_slippage(
                    positions_df,
                    execution_params.get("slippage_bps", 5)
                )
            elif execution_model == "market_impact":
                result = fees.apply_market_impact(
                    positions_df,
                    self.prices_df,
                    execution_params.get("impact_factor", 0.1)
                )
            elif execution_model == "probabilistic":
                result = fees.apply_probabilistic_slippage(
                    positions_df,
                    execution_params.get("slippage_std", 10)
                )
            else:
                logger.warning(f"Unknown execution model '{execution_model}', using default simple slippage")
                result = fees.apply_simple_slippage(positions_df, 5)
            
            return result
        
        except Exception as e:
            logger.error(f"Error applying slippage: {str(e)}")
            # Return original positions if error
            return positions_df
    
    def _calculate_performance(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for the backtest.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            DataFrame with performance metrics
        """
        logger.info("Calculating backtest performance")
        
        try:
            # Calculate returns from positions
            returns_df = calculate_returns(positions_df, self.prices_df)
            
            if returns_df.empty:
                logger.warning("No returns data calculated")
                return pd.DataFrame()
            
            # Calculate cumulative performance
            performance_df = returns_df.copy()
            
            # Group by timestamp and calculate portfolio-level metrics
            portfolio_returns = performance_df.groupby('timestamp')['return'].sum().reset_index()
            portfolio_returns['cumulative_return'] = (1 + portfolio_returns['return']).cumprod() - 1
            
            # Calculate equity curve
            initial_value = self.config["portfolio_initial_value"]
            portfolio_returns['equity'] = initial_value * (1 + portfolio_returns['cumulative_return'])
            
            # Calculate drawdown
            portfolio_returns['peak'] = portfolio_returns['equity'].cummax()
            portfolio_returns['drawdown'] = (portfolio_returns['equity'] / portfolio_returns['peak']) - 1
            
            # Store performance data
            self.performance_df = portfolio_returns
            
            # Calculate and store aggregate metrics
            self.backtest_metrics = self._calculate_aggregate_metrics(portfolio_returns)
            
            logger.info(f"Performance calculation complete - final equity: {portfolio_returns['equity'].iloc[-1]:.2f}")
            
            return portfolio_returns
            
        except Exception as e:
            logger.exception(f"Error calculating performance: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_aggregate_metrics(self, performance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate aggregate performance metrics.
        
        Args:
            performance_df: DataFrame with performance data
            
        Returns:
            Dictionary with aggregate metrics
        """
        if performance_df.empty:
            return {}
        
        metrics = {}
        
        try:
            # Basic metrics
            returns = performance_df['return'].dropna()
            
            if len(returns) < 2:
                logger.warning("Not enough data points for meaningful metrics calculation")
                return {}
            
            # Return metrics
            metrics['total_return'] = float(performance_df['cumulative_return'].iloc[-1])
            metrics['annualized_return'] = float(((1 + metrics['total_return']) ** (252 / len(returns))) - 1)
            
            # Risk metrics
            metrics['volatility'] = float(returns.std() * np.sqrt(252))
            metrics['sharpe_ratio'] = float(metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0)
            
            # Downside risk
            downside_returns = returns[returns < 0]
            metrics['downside_deviation'] = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
            metrics['sortino_ratio'] = float(metrics['annualized_return'] / metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else 0)
            
            # Drawdown analysis
            metrics['max_drawdown'] = float(performance_df['drawdown'].min())
            
            # Win/loss metrics
            metrics['win_rate'] = float((returns > 0).sum() / len(returns))
            metrics['loss_rate'] = float((returns < 0).sum() / len(returns))
            metrics['avg_win'] = float(returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0
            metrics['avg_loss'] = float(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            
            # Turnover, if positions available
            if not self.positions_df.empty:
                metrics['turnover'] = float(calculate_turnover(self.positions_df))
            
            # Additional metadata
            metrics['backtest_id'] = self.backtest_id
            metrics['strategy'] = self.config['strategy']
            metrics['start_date'] = str(performance_df['timestamp'].min())
            metrics['end_date'] = str(performance_df['timestamp'].max())
            metrics['days'] = len(performance_df)
            metrics['initial_value'] = float(self.config["portfolio_initial_value"])
            metrics['final_value'] = float(performance_df['equity'].iloc[-1])
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating aggregate metrics: {str(e)}")
            return {}
    
    def _process_time_slice(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Process a single time slice of data.
        
        Args:
            date: Timestamp for the current slice
            
        Returns:
            DataFrame with positions for this time slice
        """
        # Get data for this time slice
        prices_slice = self.prices_df[self.prices_df['timestamp'] == date]
        
        if prices_slice.empty:
            logger.debug(f"No price data for {date}")
            return pd.DataFrame()
        
        # Get signals for this time slice (if any)
        signals_slice = pd.DataFrame()
        if not self.signals_df.empty:
            signals_slice = self.signals_df[self.signals_df['timestamp'] == date]
        
        # Execute strategy
        try:
            if self.strategy_instance:
                # Execute using class-based strategy
                positions_slice = self.strategy_instance.run_strategy(signals_slice, prices_slice)
            else:
                # Legacy strategy execution
                strategy_name = self.config["strategy"]
                strategy_params = self.config["strategy_params"]
                
                # Dynamically import the strategy module
                module_name = f"quant_research.backtest.strategies.{strategy_name}"
                try:
                    strategy_module = importlib.import_module(module_name)
                    positions_slice = strategy_module.run_strategy(
                        signals_slice, prices_slice, **strategy_params
                    )
                except (ImportError, AttributeError) as e:
                    logger.error(f"Error loading strategy '{strategy_name}': {str(e)}")
                    return pd.DataFrame()
            
            if positions_slice.empty:
                logger.debug(f"No positions generated for {date}")
                return pd.DataFrame()
            
            # Apply execution costs
            positions_slice = self._apply_slippage(positions_slice)
            positions_slice = self._apply_fees(positions_slice)
            
            return positions_slice
            
        except Exception as e:
            logger.exception(f"Error processing time slice {date}: {str(e)}")
            return pd.DataFrame()
    
    def run_backtest(self) -> Tuple[bool, str]:
        """
        Run the backtest from start to finish.
        
        Returns:
            (success, message): Whether the backtest was successful and info message
        """
        logger.info(f"Starting backtest {self.backtest_id} with {self.config['strategy']} strategy")
        
        # Reset state for a fresh run
        self._reset_state()
        
        # Record start time
        start_time = time.time()
        self.is_running = True
        
        try:
            # Load data
            data_success, data_message = self._load_data()
            if not data_success:
                self.is_running = False
                return False, data_message
            
            # Get unique timestamps for simulation
            timestamps = sorted(self.prices_df['timestamp'].unique())
            
            if not timestamps:
                self.is_running = False
                return False, "No timestamps found in price data"
            
            logger.info(f"Running backtest across {len(timestamps)} time periods")
            
            # Initialize results
            all_positions = []
            
            # Choose execution mode
            if self.config["parallel"] and len(timestamps) > 10:
                # Parallel execution
                logger.info(f"Using parallel execution with {self.config['max_workers']} workers")
                with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    # Submit all time slices
                    future_to_date = {
                        executor.submit(self._process_time_slice, date): date 
                        for date in timestamps
                    }
                    
                    # Process results as they complete
                    for i, future in enumerate(as_completed(future_to_date)):
                        date = future_to_date[future]
                        try:
                            positions_slice = future.result()
                            if not positions_slice.empty:
                                all_positions.append(positions_slice)
                        except Exception as e:
                            logger.error(f"Error processing {date}: {str(e)}")
                        
                        # Periodic progress updates
                        if (i + 1) % 10 == 0 or (i + 1) == len(timestamps):
                            logger.info(f"Processed {i + 1}/{len(timestamps)} time periods")
            else:
                # Sequential execution
                for i, date in enumerate(timestamps):
                    self.current_date = date
                    
                    # Process this time slice
                    positions_slice = self._process_time_slice(date)
                    
                    if not positions_slice.empty:
                        all_positions.append(positions_slice)
                    
                    # Periodic progress updates
                    if (i + 1) % 10 == 0 or (i + 1) == len(timestamps):
                        logger.info(f"Processed {i + 1}/{len(timestamps)} time periods")
            
            # Combine all positions
            if not all_positions:
                logger.warning("No positions generated during backtest")
                self.is_running = False
                self.has_run = True
                return True, "Backtest completed, but no positions were generated"
            
            self.positions_df = pd.concat(all_positions, ignore_index=True)
            
            # Calculate performance
            self.performance_df = self._calculate_performance(self.positions_df)
            
            # Record completion
            self.execution_time = time.time() - start_time
            self.is_running = False
            self.has_run = True
            
            logger.info(f"Backtest completed in {self.execution_time:.2f} seconds")
            
            # Save results if successful
            if not self.performance_df.empty:
                self._save_results()
                
                # Log final metrics
                metrics = self.get_metrics()
                logger.info(f"Backtest results: Return={metrics['total_return']:.2%}, "
                           f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                           f"MaxDD={metrics['max_drawdown']:.2%}")
            
            return True, f"Backtest completed successfully in {self.execution_time:.2f} seconds"
            
        except Exception as e:
            logger.exception(f"Backtest error: {str(e)}")
            self.is_running = False
            return False, f"Backtest error: {str(e)}"
    
    def _save_results(self) -> bool:
        """
        Save backtest results to disk.
        
        Returns:
            Whether saving was successful
        """
        try:
            # Create results directory with backtest ID
            results_dir = os.path.join(self.config["results_dir"], self.backtest_id)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save performance data
            if not self.performance_df.empty:
                perf_path = os.path.join(results_dir, self.config["output_file"])
                self.performance_df.to_csv(perf_path, index=False)
                logger.info(f"Performance data saved to {perf_path}")
            
            # Save positions
            if not self.positions_df.empty:
                pos_path = os.path.join(results_dir, "positions.parquet")
                save_dataframe(self.positions_df, pos_path)
                logger.info(f"Position data saved to {pos_path}")
            
            # Save metrics
            metrics_path = os.path.join(results_dir, "metrics.json")
            save_to_json(self.backtest_metrics, metrics_path)
            logger.info(f"Metrics saved to {metrics_path}")
            
            # Save config
            config_path = os.path.join(results_dir, "config.json")
            save_to_json(self.config, config_path)
            logger.info(f"Configuration saved to {config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the backtest.
        
        Returns:
            Dictionary with metrics
        """
        if not self.has_run:
            logger.warning("Backtest has not been run yet")
            return {}
        
        return self.backtest_metrics
    
    def plot_equity_curve(self, figsize=(10, 6)):
        """
        Plot the equity curve from the backtest.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure or None if plotting fails
        """
        if not self.has_run or self.performance_df.empty:
            logger.warning("No performance data available to plot")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.performance_df['timestamp']):
                x = pd.to_datetime(self.performance_df['timestamp'])
            else:
                x = self.performance_df['timestamp']
            
            # Plot equity curve
            ax.plot(x, self.performance_df['equity'], label='Portfolio Value', linewidth=2)
            
            # Plot drawdowns
            twin_ax = ax.twinx()
            twin_ax.fill_between(
                x, 
                self.performance_df['drawdown'] * 100, 
                0,
                alpha=0.3, 
                color='red', 
                label='Drawdown %'
            )
            twin_ax.set_ylim(bottom=self.performance_df['drawdown'].min() * 100 * 1.5, top=5)
            twin_ax.set_ylabel('Drawdown %')
            
            # Format dates on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.set_title(f"Equity Curve - {self.config['strategy']} Strategy")
            
            # Add metrics annotation
            metrics = self.get_metrics()
            metrics_text = (
                f"Return: {metrics.get('total_return', 0):.2%}\n"
                f"Ann. Return: {metrics.get('annualized_return', 0):.2%}\n"
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\n"
                f"Max DD: {metrics.get('max_drawdown', 0):.2%}\n"
                f"Win Rate: {metrics.get('win_rate', 0):.1%}"
            )
            
            # Place text box in top left corner
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(
                0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props
            )
            
            # Format and show plot
            fig.autofmt_xdate()
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            return None


class BacktestRunner:
    """
    Helper class for running multiple backtests with different parameters.
    
    This class provides functionality for:
    - Parameter sweeps
    - Optimization
    - Comparison of multiple strategies
    - Parallel execution
    """
    
    def __init__(self, base_config: Dict[str, Any] = None):
        """
        Initialize the backtest runner.
        
        Args:
            base_config: Base configuration for all backtests
        """
        self.base_config = base_config or {}
        self.results = []
        self.current_run = 0
        self.is_running = False
    
    def run_parameter_sweep(
        self, 
        param_grid: Dict[str, List[Any]], 
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run backtests with different parameter combinations.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            parallel: Whether to run backtests in parallel
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Starting parameter sweep with {param_grid}")
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Run backtests
        return self._run_backtests(param_combinations, parallel)
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries
        """
        # Check for nested parameters
        param_combinations = [{}]
        
        for param_name, param_values in param_grid.items():
            if not param_values:
                continue
            
            # Handle nested parameters (e.g., 'strategy_params.vol_lookback')
            if '.' in param_name:
                parent_key, child_key = param_name.split('.', 1)
                
                new_combinations = []
                for combo in param_combinations:
                    for value in param_values:
                        new_combo = combo.copy()
                        if parent_key not in new_combo:
                            new_combo[parent_key] = {}
                        
                        # Handle further nesting
                        if '.' in child_key:
                            self._set_nested_param(new_combo[parent_key], child_key, value)
                        else:
                            new_combo[parent_key][child_key] = value
                        
                        new_combinations.append(new_combo)
                
                param_combinations = new_combinations
            else:
                # Handle simple parameters
                new_combinations = []
                for combo in param_combinations:
                    for value in param_values:
                        new_combo = combo.copy()
                        new_combo[param_name] = value
                        new_combinations.append(new_combo)
                
                param_combinations = new_combinations
        
        return param_combinations
    
    def _set_nested_param(self, parent_dict: Dict[str, Any], key_path: str, value: Any):
        """
        Set a value in a nested dictionary structure.
        
        Args:
            parent_dict: Parent dictionary to modify
            key_path: Dot-separated key path
            value: Value to set
        """
        if '.' in key_path:
            key, remaining_path = key_path.split('.', 1)
            if key not in parent_dict:
                parent_dict[key] = {}
            self._set_nested_param(parent_dict[key], remaining_path, value)
        else:
            parent_dict[key_path] = value
    
    def _run_backtests(
        self, 
        param_combinations: List[Dict[str, Any]], 
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run backtests with the given parameter combinations.
        
        Args:
            param_combinations: List of parameter dictionaries
            parallel: Whether to run backtests in parallel
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Running {len(param_combinations)} backtests")
        
        self.is_running = True
        self.results = []
        self.current_run = 0
        
        if parallel and len(param_combinations) > 1:
            # Parallel execution
            max_workers = min(os.cpu_count() or 4, len(param_combinations))
            logger.info(f"Using parallel execution with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all backtests
                future_to_params = {
                    executor.submit(self._run_single_backtest, params): params 
                    for params in param_combinations
                }
                
                # Process results as they complete
                for future in as_completed(future_to_params):
                    self.current_run += 1
                    params = future_to_params[future]
                    
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # Log progress
                        logger.info(f"Completed backtest {self.current_run}/{len(param_combinations)}")
                    except Exception as e:
                        logger.error(f"Backtest failed: {str(e)}")
        else:
            # Sequential execution
            for i, params in enumerate(param_combinations):
                try:
                    result = self._run_single_backtest(params)
                    self.results.append(result)
                    
                    # Log progress
                    self.current_run = i + 1
                    logger.info(f"Completed backtest {self.current_run}/{len(param_combinations)}")
                except Exception as e:
                    logger.error(f"Backtest failed: {str(e)}")
        
        self.is_running = False
        
        # Sort results by performance
        if self.results:
            self.results.sort(key=lambda x: x.get('metrics', {}).get('sharpe_ratio', -999), reverse=True)
        
        return self.results
    
    def _run_single_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single backtest with the given parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Result dictionary
        """
        # Merge base config with parameters
        config = self.base_config.copy()
        
        # Use deep update to handle nested parameters
        self._deep_update(config, params)
        
        # Create and run engine
        engine = BacktestEngine(config)
        success, message = engine.run_backtest()
        
        # Prepare result
        result = {
            'params': params,
            'config': config,
            'success': success,
            'message': message,
            'metrics': engine.get_metrics(),
            'backtest_id': engine.backtest_id
        }
        
        return result
    
    def _deep_update(self, target_dict, source_dict):
        """
        Recursively update a dictionary.
        
        Args:
            target_dict: Dictionary to update
            source_dict: Dictionary with new values
        """
        for key, value in source_dict.items():
            if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
                self._deep_update(target_dict[key], value)
            else:
                target_dict[key] = value
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get parameters of the best performing backtest.
        
        Returns:
            Parameter dictionary for best backtest
        """
        if not self.results:
            logger.warning("No backtest results available")
            return {}
        
        # Sort by Sharpe ratio and return best params
        sorted_results = sorted(
            self.results, 
            key=lambda x: x.get('metrics', {}).get('sharpe_ratio', -999), 
            reverse=True
        )
        
        return sorted_results[0]['params']
    
    def summary_report(self) -> pd.DataFrame:
        """
        Generate a summary report of all backtest results.
        
        Returns:
            DataFrame with backtest summary
        """
        if not self.results:
            logger.warning("No backtest results available")
            return pd.DataFrame()
        
        # Extract key metrics for each backtest
        rows = []
        
        for result in self.results:
            if not result['success']:
                continue
                
            metrics = result['metrics']
            params = result['params']
            
            # Create flattened parameter string
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            
            row = {
                'backtest_id': metrics.get('backtest_id', ''),
                'parameters': param_str,
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'turnover': metrics.get('turnover', 0)
            }
            
            rows.append(row)
        
        # Create DataFrame and sort by Sharpe ratio
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
    
    def save_results(self, filename: str) -> bool:
        """
        Save backtest results to a file.
        
        Args:
            filename: File path to save results
            
        Returns:
            Whether saving was successful
        """
        if not self.results:
            logger.warning("No backtest results to save")
            return False
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save to JSON
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.results)} backtest results to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    def load_results(self, filename: str) -> bool:
        """
        Load backtest results from a file.
        
        Args:
            filename: File path to load results from
            
        Returns:
            Whether loading was successful
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"Results file not found: {filename}")
                return False
                
            # Load from JSON
            with open(filename, 'r') as f:
                self.results = json.load(f)
            
            logger.info(f"Loaded {len(self.results)} backtest results from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return False


# Legacy function for backward compatibility
def register_strategy(strategy_info: Dict[str, Any], strategy_func: Callable) -> None:
    """
    Register a strategy in the legacy system.
    
    This function enables backward compatibility with the old strategy registration system.
    New code should use the BaseStrategy class and StrategyRegistry instead.
    
    Args:
        strategy_info: Dictionary with strategy metadata
        strategy_func: Strategy function that generates positions
    """
    # Store in global registry
    if not hasattr(register_strategy, 'registry'):
        register_strategy.registry = {}
    
    strategy_name = strategy_info['name']
    register_strategy.registry[strategy_name] = {
        'info': strategy_info,
        'func': strategy_func
    }
    
    logger.info(f"Registered strategy '{strategy_name}' in legacy registry")


def get_registered_strategies() -> List[str]:
    """
    Get names of all registered strategies.
    
    Returns:
        List of strategy names
    """
    # Get strategies from new registry
    class_based = StrategyRegistry.list_strategies()
    
    # Get strategies from legacy registry
    legacy = []
    if hasattr(register_strategy, 'registry'):
        legacy = list(register_strategy.registry.keys())
    
    # Combine and deduplicate
    return list(set(class_based + legacy))