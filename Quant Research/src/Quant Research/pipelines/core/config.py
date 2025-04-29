"""
Pipeline Configuration Models

This module provides Pydantic models for validating and managing pipeline configurations.
It ensures that all pipeline components receive properly structured and validated configuration
parameters, helping to prevent runtime errors due to misconfiguration.

The configuration follows a hierarchical structure:
- PipelineConfig: Top-level configuration for the entire pipeline
- ProviderConfig: Configuration for data providers
- AnalyticsConfig: Configuration for analytics modules
- BacktestConfig: Configuration for backtesting strategies
- OutputConfig: Configuration for output and visualization
"""

import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator, root_validator, DirectoryPath, FilePath


class ProviderType(str, Enum):
    """Types of data providers."""
    CRYPTO = "crypto"
    EQUITIES = "equities"
    FOREX = "forex"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class ProviderConfig(BaseModel):
    """
    Configuration for data providers.
    
    Attributes:
        id: Provider identifier (e.g., "crypto_ccxt", "equities_yf")
        type: Type of provider
        config: Provider-specific configuration parameters
        fetch_params: Parameters for the fetch_data method
    """
    id: str
    type: Optional[ProviderType] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    fetch_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('type', pre=True)
    def set_default_type(cls, v, values):
        """Set default provider type based on ID if not specified."""
        if v is None:
            provider_id = values.get('id', '')
            for provider_type in ProviderType:
                if provider_type.value in provider_id:
                    return provider_type
            return ProviderType.CUSTOM
        return v


class AnalyticsModuleConfig(BaseModel):
    """
    Configuration for an individual analytics module.
    
    Attributes:
        name: Name of the analytics module
        params: Module-specific parameters
    """
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsConfig(BaseModel):
    """
    Configuration for all analytics modules in the pipeline.
    
    Attributes:
        modules: List of analytics module configurations
        combine_signals: Whether to combine signals from different modules
        output_dir: Directory for analytics output files
    """
    modules: List[AnalyticsModuleConfig]
    combine_signals: bool = True
    output_dir: Optional[str] = None
    
    @validator('output_dir')
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist."""
        if v is not None:
            os.makedirs(v, exist_ok=True)
        return v


class BacktestStrategyParams(BaseModel):
    """
    Parameters for a backtest strategy.
    """
    # Allow any parameters as this is strategy-specific
    class Config:
        extra = "allow"


class BacktestRiskParams(BaseModel):
    """
    Risk management parameters for backtesting.
    
    Attributes:
        stop_loss_pct: Stop loss percentage
        max_drawdown_pct: Maximum allowed drawdown
        max_position_size: Maximum position size as fraction or absolute value
        target_volatility: Target annualized volatility
        vol_lookback: Lookback window for volatility calculation
        use_kelly_sizing: Whether to use Kelly criterion for position sizing
        kelly_fraction: Fraction of Kelly to use (1.0 = full Kelly)
    """
    stop_loss_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_position_size: Optional[Union[float, Dict[str, float]]] = None
    target_volatility: Optional[float] = None
    vol_lookback: int = 20
    use_kelly_sizing: bool = False
    kelly_fraction: float = 0.5
    
    @validator('stop_loss_pct', 'max_drawdown_pct', 'target_volatility')
    def validate_percentage(cls, v):
        """Validate percentage values are positive."""
        if v is not None and v <= 0:
            raise ValueError("Percentage values must be positive")
        return v
    
    @validator('kelly_fraction')
    def validate_kelly_fraction(cls, v):
        """Validate Kelly fraction is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Kelly fraction must be between 0 and 1")
        return v


class BacktestFeeParams(BaseModel):
    """
    Fee model parameters for backtesting.
    
    Attributes:
        commission_pct: Commission percentage
        min_commission: Minimum commission amount
        exchange_fees: Exchange-specific fee structure
    """
    commission_pct: float = 0.001
    min_commission: float = 0.0
    exchange_fees: Optional[Dict[str, float]] = None


class BacktestConfig(BaseModel):
    """
    Configuration for backtesting.
    
    Attributes:
        strategy: Strategy name
        strategy_params: Strategy-specific parameters
        risk_params: Risk management parameters
        fee_model: Fee model type
        fee_params: Fee model parameters
        execution_model: Execution model type
        execution_params: Execution model parameters
        initial_capital: Initial capital for the backtest
        start_date: Start date for the backtest
        end_date: End date for the backtest
        benchmark: Benchmark symbol
    """
    strategy: str
    strategy_params: BacktestStrategyParams = Field(default_factory=BacktestStrategyParams)
    risk_params: BacktestRiskParams = Field(default_factory=BacktestRiskParams)
    fee_model: str = "fixed"
    fee_params: BacktestFeeParams = Field(default_factory=BacktestFeeParams)
    execution_model: str = "simple"
    execution_params: Dict[str, Any] = Field(default_factory=lambda: {"slippage_bps": 5})
    initial_capital: float = 1000000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    benchmark: Optional[str] = None
    
    @validator('initial_capital')
    def validate_initial_capital(cls, v):
        """Validate initial capital is positive."""
        if v <= 0:
            raise ValueError("Initial capital must be positive")
        return v


class OutputConfig(BaseModel):
    """
    Configuration for pipeline output and visualization.
    
    Attributes:
        format: Output format (e.g., "csv", "parquet", "json")
        save_signals: Whether to save generated signals
        save_positions: Whether to save generated positions
        generate_reports: Whether to generate performance reports
        generate_plots: Whether to generate performance plots
    """
    format: str = "parquet"
    save_signals: bool = True
    save_positions: bool = True
    generate_reports: bool = True
    generate_plots: bool = True
    report_template: Optional[str] = None


class PipelineConfig(BaseModel):
    """
    Top-level configuration for a pipeline.
    
    Attributes:
        name: Pipeline name
        description: Pipeline description
        data_dir: Directory for input/output data
        results_dir: Directory for results
        provider: Data provider configuration
        analytics: Analytics configuration
        backtest: Backtest configuration
        output: Output configuration
    """
    name: str
    description: Optional[str] = None
    data_dir: str = "./data"
    results_dir: str = "./results"
    provider: ProviderConfig
    analytics: AnalyticsConfig
    backtest: BacktestConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    @validator('data_dir', 'results_dir')
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @root_validator
    def set_output_directories(cls, values):
        """Set output directories based on main directories if not specified."""
        if 'analytics' in values and values['analytics'].output_dir is None:
            data_dir = values.get('data_dir', './data')
            values['analytics'].output_dir = os.path.join(data_dir, 'signals')
        return values
    
    class Config:
        """Configuration for the Pydantic model."""
        validate_assignment = True
        extra = "forbid"


def load_config_from_file(file_path: Union[str, Path]) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML or JSON file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Validated PipelineConfig object
    
    Raises:
        ValueError: If the file format is not supported or validation fails
    """
    import yaml
    import json
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValueError(f"Configuration file not found: {file_path}")
    
    # Load based on file extension
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    # Validate and create config object
    return PipelineConfig(**config_dict)


def save_config_to_file(config: PipelineConfig, file_path: Union[str, Path]) -> None:
    """
    Save pipeline configuration to a YAML or JSON file.
    
    Args:
        config: PipelineConfig object
        file_path: Path to save the configuration file
        
    Raises:
        ValueError: If the file format is not supported
    """
    import yaml
    import json
    
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = config.dict()
    
    # Save based on file extension
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")