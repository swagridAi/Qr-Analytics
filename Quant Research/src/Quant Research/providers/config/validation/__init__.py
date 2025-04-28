"""
Configuration validation framework for data provider configurations.

This module provides a comprehensive validation system for checking provider configurations,
with support for validation rules, result reporting, and extensible validator registry.

The validation framework supports:
- Type checking and constraint validation
- Environment and credential validation
- Provider-specific validations
- Custom validation rules
- Detailed error reporting
"""

# Import main components for the public API
from .result import (
    ValidationResult,
    ValidationResults,
    ValidationSeverity
)
from .engine import (
    ConfigValidator,
    get_validator,
    validate,
    register_validator,
    register_field_extractor
)
from .validators import (
    validate_provider_name,
    validate_symbol_format,
    validate_timeframe,
    validate_url,
    validate_api_key_format,
    validate_connection_config,
    validate_rate_limit_config,
    get_validator as get_validator_func,
    get_all_validators
)
from .reporting import (
    ValidationReporter,
    format_validation_report,
    OutputFormat
)

# Convenience functions

def validate_config(config, schema_type=None):
    """
    Validate a configuration using the default validator.
    
    Args:
        config: Configuration to validate
        schema_type: Optional type identifier for the schema
        
    Returns:
        ValidationResults with all validation outcomes
    """
    return validate(config, schema_type)

def get_validation_report(results, format_type="text", include_details=True):
    """
    Format validation results into a report.
    
    Args:
        results: ValidationResults to format
        format_type: Output format ('text', 'json', or 'summary')
        include_details: Whether to include detailed information
        
    Returns:
        Formatted validation report
    """
    return format_validation_report(
        results, 
        format_type=format_type,
        include_details=include_details
    )

__all__ = [
    # Main classes
    'ValidationResult',
    'ValidationResults',
    'ValidationSeverity',
    'ConfigValidator',
    'ValidationReporter',
    'OutputFormat',
    
    # Main functions
    'validate',
    'validate_config',
    'register_validator',
    'register_field_extractor',
    'get_validator',
    'get_validation_report',
    
    # Individual validators
    'validate_provider_name',
    'validate_symbol_format',
    'validate_timeframe',
    'validate_url',
    'validate_api_key_format',
    'validate_connection_config',
    'validate_rate_limit_config',
    'get_validator_func',
    'get_all_validators',
]