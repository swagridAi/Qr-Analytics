"""
Signal Generator Base Class

This module defines the abstract base class for all signal generators in the analytics
engine. It provides the core interface and implementation for signal generation.

Features:
    - Abstract base class for consistent API
    - Input validation and preprocessing
    - Signal generation workflow
    - Output formatting and processing
    - Error handling
"""

# Standard library imports
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, Generic

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from quant_research.core.models import Signal
from quant_research.core.storage import save_to_parquet, save_to_duckdb
from .signal_params import SignalGeneratorParams

# Type variable for the params
T = TypeVar('T', bound='SignalGeneratorParams')

# Configure logger for analytics engine
logger = logging.getLogger("quant_research.analytics")


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