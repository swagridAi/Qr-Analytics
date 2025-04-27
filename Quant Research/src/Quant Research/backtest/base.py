"""
Base classes and interfaces for strategy implementations.

This module defines the core abstractions and common infrastructure
for all trading strategies in the quantitative research platform.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, ClassVar, TypeVar, Type
import pandas as pd
import numpy as np
import inspect
import json
import os
from enum import Enum

from quant_research.core.models import Signal, Trade

logger = logging.getLogger(__name__)


class StrategyError(Exception):
    """Base exception for strategy-related errors."""
    pass


class StrategyValidationError(StrategyError):
    """Exception raised when strategy parameters or data fail validation."""
    pass


class StrategyExecutionError(StrategyError):
    """Exception raised when an error occurs during strategy execution."""
    pass


class StrategyType(Enum):
    """Enumeration of strategy types."""
    MOMENTUM = 'momentum'
    MEAN_REVERSION = 'mean_reversion'
    ARBITRAGE = 'arbitrage'
    REGIME_ADAPTIVE = 'regime_adaptive'
    ML_BASED = 'ml_based'
    CUSTOM = 'custom'


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All concrete strategy implementations should inherit from this
    class and implement the required abstract methods.
    """
    
    # Class variables
    strategy_type: ClassVar[StrategyType] = StrategyType.CUSTOM
    name: ClassVar[str] = "base_strategy"
    description: ClassVar[str] = "Base strategy class"
    version: ClassVar[str] = "0.1.0"
    
    def __init__(self, **params):
        """
        Initialize the strategy with parameters.
        
        Args:
            **params: Strategy parameters
        """
        self.params = self.get_default_params()
        self.params.update(params)
        self.validate_params()
        self.initialize()
        
        logger.info(f"Initialized {self.name} strategy (version {self.version})")
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the strategy.
        
        Returns:
            Dictionary of default parameter values
        """
        return {}
    
    @classmethod
    def get_parameter_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about strategy parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata
        """
        return {}
    
    def validate_params(self) -> None:
        """
        Validate strategy parameters.
        
        Raises:
            StrategyValidationError: If parameters are invalid
        """
        # Default implementation does basic type checking
        parameter_info = self.get_parameter_info()
        
        for param_name, param_info in parameter_info.items():
            if param_name not in self.params:
                # Use default if provided in param_info
                if 'default' in param_info:
                    self.params[param_name] = param_info['default']
                    continue
                
                # Check if required
                if param_info.get('required', False):
                    raise StrategyValidationError(
                        f"Missing required parameter for {self.name}: {param_name}"
                    )
                continue
            
            # Check parameter type if specified
            if 'type' in param_info:
                param_type = param_info['type']
                param_value = self.params[param_name]
                
                # Handle special types
                if param_type == 'dict' and not isinstance(param_value, dict):
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be a dictionary"
                    )
                elif param_type == 'list' and not isinstance(param_value, list):
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be a list"
                    )
                elif param_type == 'bool' and not isinstance(param_value, bool):
                    if isinstance(param_value, (int, float)):
                        # Convert numeric to bool
                        self.params[param_name] = bool(param_value)
                    else:
                        raise StrategyValidationError(
                            f"Parameter {param_name} must be a boolean"
                        )
                elif param_type == 'float' and not isinstance(param_value, float):
                    if isinstance(param_value, int):
                        # Convert int to float
                        self.params[param_name] = float(param_value)
                    else:
                        raise StrategyValidationError(
                            f"Parameter {param_name} must be a float"
                        )
                elif param_type == 'int' and not isinstance(param_value, int):
                    if isinstance(param_value, float) and param_value.is_integer():
                        # Convert float to int if it's an integer value
                        self.params[param_name] = int(param_value)
                    else:
                        raise StrategyValidationError(
                            f"Parameter {param_name} must be an integer"
                        )
                elif param_type == 'string' and not isinstance(param_value, str):
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be a string"
                    )
            
            # Check allowed values if specified
            if 'allowed_values' in param_info:
                allowed_values = param_info['allowed_values']
                param_value = self.params[param_name]
                
                if param_value not in allowed_values:
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be one of: {allowed_values}"
                    )
            
            # Check range if specified
            if 'min' in param_info or 'max' in param_info:
                param_value = self.params[param_name]
                
                if not isinstance(param_value, (int, float)):
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be numeric for range validation"
                    )
                
                if 'min' in param_info and param_value < param_info['min']:
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be at least {param_info['min']}"
                    )
                
                if 'max' in param_info and param_value > param_info['max']:
                    raise StrategyValidationError(
                        f"Parameter {param_name} must be at most {param_info['max']}"
                    )
    
    def initialize(self) -> None:
        """
        Perform strategy-specific initialization.
        
        This method is called after parameters are set and validated.
        Override in subclasses for custom initialization logic.
        """
        pass
    
    @abstractmethod
    def generate_positions(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions from signals.
        
        Args:
            signals_df: DataFrame with signals from analytics modules
            prices_df: DataFrame with prices [timestamp, asset_id, open, high, low, close, volume]
            
        Returns:
            DataFrame with positions [timestamp, asset_id, position, target_weight]
        """
        raise NotImplementedError("Subclasses must implement generate_positions()")
    
    @abstractmethod
    def apply_risk_controls(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk controls to positions.
        
        Args:
            positions_df: DataFrame with positions [timestamp, asset_id, position, target_weight]
            prices_df: DataFrame with prices [timestamp, asset_id, close, ...]
            
        Returns:
            DataFrame with risk-adjusted positions
        """
        raise NotImplementedError("Subclasses must implement apply_risk_controls()")
    
    def run_strategy(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the strategy end-to-end, converting signals to positions with risk controls.
        
        This method provides a common implementation that calls the abstract methods
        in sequence. Subclasses can override for custom behavior.
        
        Args:
            signals_df: DataFrame with signals from analytics modules
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with positions ready for execution
        """
        try:
            # Log strategy execution
            logger.info(f"Running {self.name} strategy")
            
            # Validate input data
            self._validate_input_data(signals_df, prices_df)
            
            # Generate positions from signals
            positions_df = self.generate_positions(signals_df, prices_df)
            
            # Apply risk controls
            positions_df = self.apply_risk_controls(positions_df, prices_df)
            
            # Calculate notional values
            if 'position' in positions_df.columns and 'close' in prices_df.columns:
                positions_df = self._calculate_notional_values(positions_df, prices_df)
            
            logger.info(f"{self.name} strategy complete with {len(positions_df)} positions")
            return positions_df
            
        except Exception as e:
            logger.error(f"Error executing {self.name} strategy: {str(e)}")
            raise StrategyExecutionError(f"Strategy execution failed: {str(e)}") from e
    
    def _validate_input_data(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
        """
        Validate input data before running the strategy.
        
        Args:
            signals_df: DataFrame with signals
            prices_df: DataFrame with prices
            
        Raises:
            StrategyValidationError: If data is invalid
        """
        # Check that DataFrames are not empty
        if signals_df is None or signals_df.empty:
            raise StrategyValidationError("Signals DataFrame is empty")
        
        if prices_df is None or prices_df.empty:
            raise StrategyValidationError("Prices DataFrame is empty")
        
        # Check required columns
        required_signal_columns = ['timestamp', 'asset_id']
        missing_signal_columns = [col for col in required_signal_columns if col not in signals_df.columns]
        
        if missing_signal_columns:
            raise StrategyValidationError(
                f"Signals DataFrame missing required columns: {missing_signal_columns}"
            )
        
        required_price_columns = ['timestamp', 'asset_id', 'close']
        missing_price_columns = [col for col in required_price_columns if col not in prices_df.columns]
        
        if missing_price_columns:
            raise StrategyValidationError(
                f"Prices DataFrame missing required columns: {missing_price_columns}"
            )
    
    def _calculate_notional_values(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate notional values for positions.
        
        Args:
            positions_df: DataFrame with positions
            prices_df: DataFrame with prices
            
        Returns:
            DataFrame with notional values added
        """
        result = positions_df.copy()
        
        # Merge positions with prices to get the close price
        if 'close' not in result.columns:
            # Handle potential duplicate columns during merge
            position_columns = list(result.columns)
            
            # Merge only necessary columns from prices
            prices_subset = prices_df[['timestamp', 'asset_id', 'close']].copy()
            
            # Perform merge
            result = pd.merge(
                result,
                prices_subset,
                on=['timestamp', 'asset_id'],
                how='left'
            )
        
        # Calculate notional value
        result['notional_value'] = result['position'] * result['close']
        
        return result
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get descriptive information about the strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "type": self.strategy_type.value,
            "parameters": self.get_parameter_info()
        }
    
    def save_config(self, filepath: str) -> bool:
        """
        Save strategy configuration to file.
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Prepare config dict
            config = {
                "strategy": self.name,
                "version": self.version,
                "type": self.strategy_type.value,
                "parameters": self.params
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Strategy configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy configuration: {str(e)}")
            return False
    
    @classmethod
    def load_config(cls, filepath: str) -> Dict[str, Any]:
        """
        Load strategy configuration from file.
        
        Args:
            filepath: Path to load configuration from
            
        Returns:
            Dictionary with strategy configuration
            
        Raises:
            StrategyValidationError: If configuration is invalid
        """
        try:
            if not os.path.exists(filepath):
                raise StrategyValidationError(f"Configuration file not found: {filepath}")
            
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Validate basic structure
            if not isinstance(config, dict):
                raise StrategyValidationError("Invalid configuration format")
            
            if 'strategy' not in config:
                raise StrategyValidationError("Missing 'strategy' in configuration")
            
            if 'parameters' not in config:
                raise StrategyValidationError("Missing 'parameters' in configuration")
            
            logger.info(f"Strategy configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading strategy configuration: {str(e)}")
            raise StrategyValidationError(f"Failed to load configuration: {str(e)}") from e
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseStrategy':
        """
        Create strategy instance from configuration.
        
        Args:
            config: Dictionary with strategy configuration
            
        Returns:
            Strategy instance
            
        Raises:
            StrategyValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise StrategyValidationError("Invalid configuration format")
        
        if 'strategy' not in config:
            raise StrategyValidationError("Missing 'strategy' in configuration")
        
        strategy_name = config['strategy']
        
        if strategy_name != cls.name:
            logger.warning(
                f"Strategy name mismatch: config specifies '{strategy_name}', "
                f"but creating instance of '{cls.name}'"
            )
        
        # Extract parameters
        parameters = config.get('parameters', {})
        
        # Create instance
        return cls(**parameters)


class StrategyRegistry:
    """
    Registry for strategy implementations.
    
    This class maintains a registry of available strategies and provides
    methods to register, retrieve, and instantiate them.
    """
    
    _registry: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"Class {strategy_class.__name__} must inherit from BaseStrategy")
        
        strategy_name = strategy_class.name
        cls._registry[strategy_name] = strategy_class
        logger.info(f"Registered strategy: {strategy_name}")
    
    @classmethod
    def get_strategy_class(cls, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.
        
        Args:
            strategy_name: Name of strategy to retrieve
            
        Returns:
            Strategy class or None if not found
        """
        return cls._registry.get(strategy_name)
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **params) -> Optional[BaseStrategy]:
        """
        Create a strategy instance by name.
        
        Args:
            strategy_name: Name of strategy to create
            **params: Parameters to pass to strategy constructor
            
        Returns:
            Strategy instance or None if not found
        """
        strategy_class = cls.get_strategy_class(strategy_name)
        if strategy_class is None:
            logger.error(f"Strategy not found: {strategy_name}")
            return None
        
        try:
            return strategy_class(**params)
        except Exception as e:
            logger.error(f"Error creating strategy {strategy_name}: {str(e)}")
            return None
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Get list of registered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str = None) -> Dict[str, Any]:
        """
        Get information about registered strategies.
        
        Args:
            strategy_name: Name of strategy to get info for, or None for all
            
        Returns:
            Dictionary with strategy information
        """
        if strategy_name is not None:
            strategy_class = cls.get_strategy_class(strategy_name)
            if strategy_class is None:
                return {}
            return {strategy_name: strategy_class.get_default_params()}
        
        result = {}
        for name, strategy_class in cls._registry.items():
            result[name] = {
                "description": strategy_class.description,
                "version": strategy_class.version,
                "type": strategy_class.strategy_type.value,
                "parameters": strategy_class.get_parameter_info()
            }
        return result
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Optional[BaseStrategy]:
        """
        Create a strategy instance from configuration.
        
        Args:
            config: Dictionary with strategy configuration
            
        Returns:
            Strategy instance or None if error
        """
        if not isinstance(config, dict):
            logger.error("Invalid configuration format")
            return None
        
        if 'strategy' not in config:
            logger.error("Missing 'strategy' in configuration")
            return None
        
        strategy_name = config['strategy']
        strategy_class = cls.get_strategy_class(strategy_name)
        
        if strategy_class is None:
            logger.error(f"Strategy not found: {strategy_name}")
            return None
        
        # Extract parameters
        parameters = config.get('parameters', {})
        
        try:
            return strategy_class(**parameters)
        except Exception as e:
            logger.error(f"Error creating strategy from config: {str(e)}")
            return None


def register_strategy(strategy_class: Type[BaseStrategy]) -> None:
    """
    Register a strategy with the registry.
    
    This function is a convenience wrapper around StrategyRegistry.register().
    
    Args:
        strategy_class: Strategy class to register
    """
    StrategyRegistry.register(strategy_class)


def auto_register_strategies() -> None:
    """
    Automatically discover and register all strategy implementations.
    
    This function searches for strategy classes in the strategies directory
    and registers them with the registry.
    """
    try:
        import importlib
        import pkgutil
        from quant_research.backtest import strategies
        
        logger.info("Auto-discovering strategy implementations")
        
        # Get the package path
        package_path = strategies.__path__
        prefix = strategies.__name__ + "."
        
        # Iterate through all modules in the package
        for _, module_name, is_pkg in pkgutil.iter_modules(package_path, prefix):
            if is_pkg:
                # Skip packages for now
                continue
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find all strategy classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        
                        # Register the strategy
                        register_strategy(obj)
            
            except Exception as e:
                logger.error(f"Error loading strategy module {module_name}: {str(e)}")
        
        logger.info(f"Discovered {len(StrategyRegistry.list_strategies())} strategies")
    
    except Exception as e:
        logger.error(f"Error auto-registering strategies: {str(e)}")