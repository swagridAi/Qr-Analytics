"""
Signal Generator Parameters

This module defines the base parameter class for all signal generators in the analytics
engine. It provides standardized parameter validation using Pydantic.

Features:
    - Standardized parameter validation
    - Common output configuration options
    - Logging settings
    - Validation methods for parameter consistency
"""

# Standard library imports
import logging
import warnings
from typing import Optional

# Third-party imports
from pydantic import BaseModel, Field, validator, root_validator

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