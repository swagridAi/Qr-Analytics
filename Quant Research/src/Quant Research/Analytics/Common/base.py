"""
Analytics Engine Base Class

This module defines the abstract base class for all signal generators in the analytics
engine. It provides standardized interfaces, parameter validation, logging, and 
output formatting to ensure consistency across different analytics modules.

Features:
    - Standardized parameter validation using Pydantic
    - Unified signal generation interface
    - Consistent logging and error handling
    - Flexible output formats (DataFrame or Signal objects)
    - Built-in performance tracking
    - Registry pattern for signal generator discovery

Usage:
    All signal generators in the analytics engine should inherit from the 
    SignalGenerator base class and implement its abstract methods.

Example:
    ```python
    from quant_research.analytics.base import SignalGenerator, SignalGeneratorParams

    class MyParams(SignalGeneratorParams):
        window: int = Field(20, gt=0, description="Analysis window size")
        threshold: float = Field(1.5, description="Signal threshold")

    class MySignalGenerator(SignalGenerator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.params = self.validate_params(MyParams, kwargs)
            
        def _generate(self, df: pd.DataFrame) -> pd.DataFrame:
            # Implementation of signal generation logic
            return signals_df
    ```
"""

# Standard library imports
import logging
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar, Generic

# Third-party imports
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator, root_validator

# Local imports
from quant_research.core.models import Signal
from quant_research.core.storage import save_to_parquet, save_to_duckdb

# Type variable for the params
T = TypeVar('T', bound='SignalGeneratorParams')

# Configure logger for analytics engine
logger = logging.getLogger("quant_research.analytics")


class SignalGeneratorParams(BaseModel):
    """
    Base class for signal generator parameters.
    
    This class provides a foundation for parameter validation and
    standardization across different signal generators. All specific
    parameter classes should inherit from this base class.
    
    Attributes:
        output_file (Optional[str]): Path to save signal output
        output_format (str): Format for output ('parquet', 'duckdb', or 'both')
        as_objects (bool): Return Signal objects instead of DataFrame
        log_level (str): Logging level for this generator
        name (Optional[str]): Custom name for the signal generator
    """
    # Output configuration
    output_file: Optional[str] = Field(
        None, 
        description="Path to save signal output"
    )
    output_format: str = Field(
        "parquet", 
        description="Format for output ('parquet', 'duckdb', or 'both')"
    )
    as_objects: bool = Field(
        False, 
        description="Return Signal objects instead of DataFrame"
    )
    
    # Execution configuration
    log_level: str = Field(
        "INFO", 
        description="Logging level for this generator"
    )
    name: Optional[str] = Field(
        None, 
        description="Custom name for the signal generator"
    )
    
    @validator('output_format')
    def validate_output_format(cls, v):
        """Validate that output format is one of the supported formats."""
        valid_formats = ['parquet', 'duckdb', 'both']
        if v.lower() not in valid_formats:
            raise ValueError(f"Output format must be one of {valid_formats}")
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate that log level is one of the standard logging levels."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @root_validator
    def validate_output_settings(cls, values):
        """Validate output settings for consistency between options."""
        as_objects = values.get('as_objects')
        output_file = values.get('output_file')
        
        if as_objects and output_file:
            warnings.warn(
                "Both as_objects and output_file are set. Signals will be "
                "saved to file and returned as objects."
            )
        
        return values
    
    class Config:
        """Configuration for parameter models."""
        extra = "forbid"  # Forbid extra fields not defined in the model
        validate_assignment = True  # Validate fields even after model creation
        arbitrary_types_allowed = True  # Allow any type for fields


class SignalGenerator(Generic[T], ABC):
    """
    Abstract base class for all signal generators in the analytics engine.
    
    This class defines the standard interface and shared functionality for
    signal generators across different modules. It handles common tasks like
    input validation, output formatting, and error handling.
    
    Attributes:
        params_class (Type[SignalGeneratorParams]): Class for parameter validation
        params (SignalGeneratorParams): Validated parameters
        logger (logging.Logger): Logger instance for this generator
        name (str): Name of this generator instance
    """
    
    #------------------------------------------------------------------------
    # Initialization & Configuration
    #------------------------------------------------------------------------
    
    def __init__(self, **kwargs):
        """
        Initialize the signal generator.
        
        Args:
            **kwargs: Keyword arguments for parameter initialization
        """
        # Initialize logger
        self.logger = logger
        
        # Set up custom log level if specified
        if 'log_level' in kwargs:
            log_level = kwargs['log_level'].upper()
            level = getattr(logging, log_level, None)
            if level:
                self.logger.setLevel(level)
        
        # Initialize parameters (subclasses will validate with specific models)
        self.params = kwargs
        self.name = kwargs.get('name', self.__class__.__name__)
    
    def validate_params(self, params_class: Type[T], params: Dict[str, Any]) -> T:
        """
        Validate parameters against a Pydantic model.
        
        Args:
            params_class: Pydantic model class for parameter validation
            params: Dictionary of parameters to validate
            
        Returns:
            Validated parameter model instance
        
        Raises:
            ValueError: If parameter validation fails
        """
        try:
            return params_class(**params)
        except Exception as e:
            self.logger.error(f"Parameter validation failed for {self.name}: {e}")
            raise ValueError(f"Invalid parameters for {self.name}: {e}") from e
    
    #------------------------------------------------------------------------
    # Public API
    #------------------------------------------------------------------------
    
    def generate_signal(self, df: pd.DataFrame) -> Union[pd.DataFrame, List[Signal]]:
        """
        Generate signals from input data.
        
        This is the main entry point for signal generation that:
        1. Validates the input DataFrame
        2. Calls the implementation-specific _generate method
        3. Formats and processes the output
        4. Optionally saves the results
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            DataFrame with signals or list of Signal objects
            
        Raises:
            TypeError: If input is not a DataFrame
            ValueError: If input DataFrame is empty or missing required columns
            RuntimeError: If signal generation fails
        """
        self.logger.info(f"Generating signals with {self.name}")
        
        # Measure execution time
        start_time = time.time()
        
        # Validate input DataFrame
        df = self._validate_input_df(df)
        
        try:
            # Call implementation-specific signal generation
            signals_df = self._generate(df)
            
            # Process and format the output
            signals_df = self._process_output(signals_df, df)
            
            # Save signals if output_file is specified
            output_file = getattr(self.params, 'output_file', None)
            if output_file:
                self._save_signals(signals_df, output_file)
            
            # Convert to Signal objects if requested
            as_objects = getattr(self.params, 'as_objects', False)
            if as_objects:
                signals = self._convert_to_signal_objects(signals_df)
                result = signals
            else:
                result = signals_df
            
            # Log execution time
            elapsed = time.time() - start_time
            self.logger.info(f"Generated {len(signals_df)} signals in {elapsed:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate signals with {self.name}: {e}") from e
    
    @abstractmethod
    def _generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation-specific signal generation.
        
        This abstract method must be implemented by subclasses to provide specific
        signal generation logic.
        
        Args:
            df: Preprocessed input DataFrame
            
        Returns:
            DataFrame with generated signals
        """
        pass
    
    #------------------------------------------------------------------------
    # Helper Methods - Input Processing
    #------------------------------------------------------------------------
    
    def _validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess input DataFrame.
        
        Performs common validations and transformations to prepare data
        for signal generation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated and preprocessed DataFrame
            
        Raises:
            TypeError: If input is not a DataFrame
            ValueError: If DataFrame is empty
        """
        # Ensure we have a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have a datetime index if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                self.logger.warning("DataFrame has no timestamp index or column")
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        return df
    
    #------------------------------------------------------------------------
    # Helper Methods - Output Processing
    #------------------------------------------------------------------------
    
    def _process_output(self, signals_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and format the output signals.
        
        Ensures that the signals DataFrame has all required columns and
        standard formatting.
        
        Args:
            signals_df: Raw signal output from _generate
            original_df: Original input DataFrame
            
        Returns:
            Processed and formatted signals DataFrame
            
        Raises:
            ValueError: If required columns are missing and cannot be inferred
        """
        # Ensure we have required columns
        required_columns = ['timestamp', 'signal_type', 'value']
        
        # If timestamp is the index, convert it to a column
        if isinstance(signals_df.index, pd.DatetimeIndex) and 'timestamp' not in signals_df.columns:
            signals_df = signals_df.reset_index()
        
        # Check if essential columns are present
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        if missing_columns:
            self.logger.warning(f"Signal output missing required columns: {missing_columns}")
            
            # Try to infer timestamp if missing
            if 'timestamp' in missing_columns and isinstance(signals_df.index, pd.DatetimeIndex):
                signals_df['timestamp'] = signals_df.index
                missing_columns.remove('timestamp')
            
            # Add signal_type if missing
            if 'signal_type' in missing_columns:
                signals_df['signal_type'] = self.name.lower()
                missing_columns.remove('signal_type')
            
            # If still missing required columns, raise an error
            if missing_columns:
                raise ValueError(f"Signal output missing required columns: {missing_columns}")
        
        # Add generator name for traceability
        if 'generator' not in signals_df.columns:
            signals_df['generator'] = self.name
            
        # Add generation timestamp
        if 'generated_at' not in signals_df.columns:
            signals_df['generated_at'] = pd.Timestamp.now()
        
        # Ensure timestamp is datetime
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        
        return signals_df
    
    def _save_signals(self, signals_df: pd.DataFrame, output_file: str) -> None:
        """
        Save signals to specified output format.
        
        Handles saving signals to parquet files and/or DuckDB based on
        configuration.
        
        Args:
            signals_df: DataFrame with signals to save
            output_file: Output file path
            
        Returns:
            None
        """
        output_format = getattr(self.params, 'output_format', 'parquet')
        
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on specified format
        if output_format in ('parquet', 'both'):
            save_to_parquet(signals_df, output_file)
            self.logger.info(f"Saved signals to {output_file}")
            
        if output_format in ('duckdb', 'both'):
            try:
                save_to_duckdb(signals_df, 'signals', mode='append')
                self.logger.info("Saved signals to DuckDB")
            except Exception as e:
                self.logger.warning(f"Failed to save to DuckDB: {e}")
    
    def _convert_to_signal_objects(self, signals_df: pd.DataFrame) -> List[Signal]:
        """
        Convert signals DataFrame to list of Signal objects.
        
        Transforms a DataFrame of signals into a list of Signal objects for
        easier integration with downstream components.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        for _, row in signals_df.iterrows():
            # Extract base fields
            timestamp = row['timestamp']
            signal_type = row['signal_type']
            value = row['value']
            
            # Extract metadata fields (any column not in the standard fields)
            standard_fields = {'timestamp', 'signal_type', 'value', 'generator', 'generated_at'}
            metadata = {col: row[col] for col in row.index if col not in standard_fields}
            
            # Create Signal object
            signal = Signal(
                timestamp=timestamp,
                signal_type=signal_type,
                value=value,
                source=self.name,
                metadata=metadata
            )
            
            signals.append(signal)
        
        return signals
    
    #------------------------------------------------------------------------
    # Magic Methods
    #------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        """String representation of the signal generator."""
        return f"{self.name}(params={self.params})"


#------------------------------------------------------------------------
# Registry Pattern Implementation
#------------------------------------------------------------------------

class SignalGeneratorRegistry:
    """
    Registry for signal generators.
    
    This class provides a central registry for signal generators, allowing
    dynamic loading and creation of generators by name. It implements the
    registry pattern to decouple generator definition from usage.
    
    Usage:
        # Register a generator
        SignalGeneratorRegistry.register('volatility', VolatilitySignalGenerator)
        
        # Create a generator instance by name
        volatility_generator = SignalGeneratorRegistry.create('volatility', window=21)
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, generator_class: Type[SignalGenerator]) -> None:
        """
        Register a signal generator class.
        
        Args:
            name: Name to register the generator under
            generator_class: SignalGenerator class
            
        Returns:
            None
            
        Raises:
            TypeError: If the class does not inherit from SignalGenerator
        """
        if not issubclass(generator_class, SignalGenerator):
            raise TypeError(f"Class {generator_class.__name__} must inherit from SignalGenerator")
            
        cls._registry[name] = generator_class
        logger.debug(f"Registered signal generator: {name}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> SignalGenerator:
        """
        Create a signal generator instance by name.
        
        Args:
            name: Name of the generator to create
            **kwargs: Parameters to pass to the generator constructor
            
        Returns:
            SignalGenerator instance
            
        Raises:
            ValueError: If the generator name is not registered
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown signal generator: {name}. Available generators: {', '.join(cls._registry.keys())}")
            
        generator_class = cls._registry[name]
        return generator_class(**kwargs)
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """
        List all registered generator names.
        
        Returns:
            List of registered generator names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_info(cls, name: str = None) -> Dict[str, Any]:
        """
        Get information about registered generators.
        
        Args:
            name: Optional name of specific generator to get info for
            
        Returns:
            Dictionary with generator information
            
        Raises:
            ValueError: If the specified generator name is not registered
        """
        if name is not None:
            if name not in cls._registry:
                raise ValueError(f"Unknown signal generator: {name}")
                
            generator_class = cls._registry[name]
            return {
                "name": name,
                "class": generator_class.__name__,
                "module": generator_class.__module__,
                "description": generator_class.__doc__.split('\n')[0] if generator_class.__doc__ else "No description"
            }
        else:
            return {
                name: {
                    "class": generator_class.__name__,
                    "module": generator_class.__module__,
                    "description": generator_class.__doc__.split('\n')[0] if generator_class.__doc__ else "No description"
                }
                for name, generator_class in cls._registry.items()
            }


#------------------------------------------------------------------------
# Pipeline Pattern Implementation
#------------------------------------------------------------------------

class SignalPipeline:
    """
    Pipeline for executing multiple signal generators.
    
    This class allows combining multiple signal generators into a processing
    pipeline that can be executed as a single unit. It implements the pipeline
    pattern for sequential data processing.
    
    Usage:
        # Create generators
        vol_gen = VolatilitySignalGenerator(window=21)
        regime_gen = RegimeDetectorSignalGenerator(n_states=3)
        
        # Create pipeline
        pipeline = SignalPipeline([vol_gen, regime_gen])
        
        # Run pipeline
        results = pipeline.run(data_df)
    """
    
    def __init__(self, generators: List[SignalGenerator]):
        """
        Initialize the signal pipeline.
        
        Args:
            generators: List of signal generators to include in the pipeline
        """
        self.generators = generators
        self.logger = logging.getLogger("quant_research.analytics.pipeline")
    
    def run(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all signal generators in the pipeline.
        
        Executes each signal generator in sequence and collects their results.
        Errors in individual generators do not stop the pipeline.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            Dictionary mapping generator names to signal DataFrames
        """
        self.logger.info(f"Running signal pipeline with {len(self.generators)} generators")
        
        results = {}
        start_time = time.time()
        
        for generator in self.generators:
            try:
                name = generator.name
                self.logger.info(f"Running generator: {name}")
                
                signals = generator.generate_signal(df)
                
                # Skip None results
                if signals is None:
                    self.logger.warning(f"Generator {name} returned None")
                    continue
                    
                # Convert Signal objects to DataFrame if needed
                if isinstance(signals, list) and signals and isinstance(signals[0], Signal):
                    signals_df = pd.DataFrame([s.__dict__ for s in signals])
                else:
                    signals_df = signals
                
                results[name] = signals_df
                
            except Exception as e:
                self.logger.error(f"Error in generator {generator.name}: {e}", exc_info=True)
                # Continue with other generators instead of stopping the pipeline
        
        elapsed = time.time() - start_time
        self.logger.info(f"Pipeline completed in {elapsed:.2f} seconds with {len(results)} successful generators")
        
        return results
    
    def add_generator(self, generator: SignalGenerator) -> None:
        """
        Add a signal generator to the pipeline.
        
        Args:
            generator: Signal generator to add
            
        Returns:
            None
        """
        self.generators.append(generator)
        self.logger.debug(f"Added generator {generator.name} to pipeline")
    
    def remove_generator(self, name: str) -> bool:
        """
        Remove a signal generator from the pipeline by name.
        
        Args:
            name: Name of the generator to remove
            
        Returns:
            True if a generator was removed, False otherwise
        """
        initial_count = len(self.generators)
        self.generators = [g for g in self.generators if g.name != name]
        removed = len(self.generators) < initial_count
        
        if removed:
            self.logger.debug(f"Removed generator {name} from pipeline")
        
        return removed