# src/quant_research/core/config/validation/engine.py
"""
Configuration validation engine for validating provider and system configurations.

This module provides a flexible and extensible validation framework that separates
validation rules from schema definitions and allows reusing common validation patterns
across different configuration types.
"""

import inspect
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from .result import ValidationResult, ValidationResults, ValidationSeverity

logger = logging.getLogger(__name__)


# Type definitions
ValidatorFunc = Callable[..., ValidationResult]
FieldExtractorFunc = Callable[[BaseModel], Dict[str, Any]]


class ConfigValidator:
    """
    Unified configuration validator.
    
    This class provides a centralized validation mechanism for all configurations
    in the system. It allows registering validator functions that can be applied
    to specific configuration types or fields.
    
    Features:
    - Register validator functions with field pattern matching
    - Apply validators selectively to configuration objects
    - Collect and report validation results
    - Support for pre/post validation hooks
    """
    
    def __init__(self):
        """Initialize the validator with empty registry"""
        self.validators: Dict[str, ValidatorFunc] = {}
        self.field_extractors: Dict[str, FieldExtractorFunc] = {}
        self.validator_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register built-in validators and extractors"""
        # These will be imported from validators.py
        # Basic field validators
        from .validators import (
            validate_provider_name, 
            validate_symbol_format,
            validate_timeframe,
            validate_url,
            validate_positive_integer,
            validate_timeout,
            validate_pool_size,
            validate_rate_limit,
            validate_environment_variables
        )
        
        # Register these validators
        self.register_validator('provider_name', validate_provider_name)
        self.register_validator('symbol_format', validate_symbol_format)
        self.register_validator('timeframe', validate_timeframe)
        self.register_validator('url', validate_url)
        self.register_validator('positive_integer', validate_positive_integer)
        self.register_validator('timeout', validate_timeout)
        self.register_validator('pool_size', validate_pool_size)
        self.register_validator('rate_limit', validate_rate_limit)
        self.register_validator('environment_variables', validate_environment_variables)
        
        # Register field extractors
        self.register_field_extractor('name', lambda config: {'name': getattr(config, 'name', None)})
        self.register_field_extractor('symbols', lambda config: {'symbols': getattr(config, 'symbols', [])})
        self.register_field_extractor('environment', lambda config: {
            'env_prefix': getattr(config, 'env_prefix', ''),
            'api_keys': getattr(config, 'api_keys', []),
            'require_auth': getattr(config, 'require_auth', False)
        })
        self.register_field_extractor('connection', lambda config: {
            'timeout': getattr(config.connection, 'timeout', 30),
            'pool_size': getattr(config.connection, 'pool_size', 5),
            'retry_delay': getattr(config.connection, 'retry_delay', 1.0),
            'max_retries': getattr(config.connection, 'max_retries', 3)
        })
        self.register_field_extractor('rate_limit', lambda config: getattr(config, 'rate_limit', {}))
    
    def register_validator(self, name: str, validator_func: ValidatorFunc, 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a validator function.
        
        Args:
            name: Unique identifier for the validator
            validator_func: Function that performs validation and returns ValidationResult
            metadata: Optional metadata about the validator (description, applicable types, etc.)
        """
        if name in self.validators:
            logger.warning(f"Overriding existing validator: {name}")
        
        self.validators[name] = validator_func
        self.validator_metadata[name] = metadata or {}
        
        logger.debug(f"Registered validator: {name}")
    
    def register_field_extractor(self, name: str, extractor_func: FieldExtractorFunc) -> None:
        """
        Register a field extractor function that pulls relevant fields from a config.
        
        Args:
            name: Unique identifier for the extractor
            extractor_func: Function that extracts fields from a config object
        """
        if name in self.field_extractors:
            logger.warning(f"Overriding existing field extractor: {name}")
        
        self.field_extractors[name] = extractor_func
        logger.debug(f"Registered field extractor: {name}")
    
    def _get_applicable_validators(self, config: BaseModel, schema_type: str) -> List[str]:
        """
        Determine which validators apply to a given configuration.
        
        Args:
            config: Configuration object
            schema_type: Schema type identifier
            
        Returns:
            List of applicable validator names
        """
        # This would normally check the registry for validators applicable to this schema_type
        # For now, we'll return all validators
        from ..registry import ConfigRegistry
        
        return ConfigRegistry.get_validators(schema_type)
    
    def _get_validator_fields(self, validator_name: str, config: BaseModel) -> Dict[str, Any]:
        """
        Get fields from a configuration that should be validated by a specific validator.
        
        Args:
            validator_name: Name of the validator
            config: Configuration object
            
        Returns:
            Dictionary of field names to values
        """
        # Find the appropriate extractor for this validator
        extractor_name = self.validator_metadata.get(validator_name, {}).get('extractor')
        
        if extractor_name and extractor_name in self.field_extractors:
            # Use the registered extractor
            return self.field_extractors[extractor_name](config)
        
        # Default: try to find fields by name pattern matching
        # This is a simplified approach - we'd want more sophisticated matching in a real implementation
        fields = {}
        for field_name, field_value in config.__dict__.items():
            # Simple approach: if validator name contains field name or vice versa
            if (validator_name in field_name) or (field_name in validator_name):
                fields[field_name] = field_value
                
        return fields
    
    def validate(self, config: BaseModel, schema_type: Optional[str] = None, 
               validator_names: Optional[List[str]] = None) -> ValidationResults:
        """
        Validate a configuration using registered validators.
        
        Args:
            config: Configuration to validate
            schema_type: Optional type identifier for the schema
            validator_names: Optional list of specific validators to apply
            
        Returns:
            ValidationResults with all validation outcomes
        """
        results = ValidationResults()
        
        # First, run Pydantic validation
        try:
            # This will re-validate basic field types and constraints
            config.__class__.validate(config)
        except ValidationError as e:
            for error in e.errors():
                results.add_result(
                    ValidationResult(
                        is_valid=False,
                        message=f"Field validation error: {error.get('msg')}",
                        field='.'.join(str(loc) for loc in error.get('loc', [])),
                        severity=ValidationSeverity.ERROR,
                        details={"error": error}
                    )
                )
            # If basic validation fails, don't continue with custom validators
            return results
        
        # Determine schema type if not provided
        if schema_type is None:
            schema_type = getattr(config, 'name', config.__class__.__name__)
        
        # Determine which validators to apply
        if validator_names is None:
            validator_names = self._get_applicable_validators(config, schema_type)
        
        # Run pre-validation hooks if registered
        self._run_pre_validation_hooks(config, schema_type, results)
        
        # Apply each validator
        for validator_name in validator_names:
            if validator_name not in self.validators:
                results.add_result(
                    ValidationResult(
                        is_valid=False,
                        message=f"Unknown validator: {validator_name}",
                        severity=ValidationSeverity.WARNING
                    )
                )
                continue
            
            # Get config fields applicable to this validator
            fields = self._get_validator_fields(validator_name, config)
            
            # Skip if no applicable fields
            if not fields:
                continue
            
            # Apply validator
            validator_func = self.validators[validator_name]
            
            try:
                # Check validator function signature
                sig = inspect.signature(validator_func)
                
                # Determine which fields to pass based on function parameters
                params = {}
                for param_name in sig.parameters:
                    if param_name in fields:
                        params[param_name] = fields[param_name]
                
                if len(params) == 0 and len(fields) > 0:
                    # If no params match but we have fields, pass the first field as positional arg
                    # This handles simple validators that take a single value
                    result = validator_func(next(iter(fields.values())))
                else:
                    # Call with matching parameters
                    result = validator_func(**params)
                
                # Add source information to the result
                if isinstance(result, ValidationResult):
                    # If a single result
                    field_name = next(iter(fields.keys())) if fields else None
                    result.field = result.field or field_name
                    result.validator = validator_name
                    results.add_result(result)
                elif isinstance(result, list):
                    # If multiple results
                    for r in result:
                        if isinstance(r, ValidationResult):
                            r.validator = validator_name
                            results.add_result(r)
                
            except Exception as e:
                # Handle validator function errors
                results.add_result(
                    ValidationResult(
                        is_valid=False,
                        message=f"Validator '{validator_name}' failed: {str(e)}",
                        validator=validator_name,
                        severity=ValidationSeverity.ERROR,
                        details={"error": str(e), "exception_type": type(e).__name__}
                    )
                )
                logger.exception(f"Error applying validator '{validator_name}'")
        
        # Run post-validation hooks if registered
        self._run_post_validation_hooks(config, schema_type, results)
        
        return results
    
    def _run_pre_validation_hooks(self, config: BaseModel, schema_type: str, 
                                results: ValidationResults) -> None:
        """
        Run pre-validation hooks registered for this schema type.
        
        Args:
            config: Configuration being validated
            schema_type: Schema type identifier
            results: Results container to add hook results to
        """
        # Implementation would look up hooks from registry
        pass
    
    def _run_post_validation_hooks(self, config: BaseModel, schema_type: str, 
                                 results: ValidationResults) -> None:
        """
        Run post-validation hooks registered for this schema type.
        
        Args:
            config: Configuration being validated
            schema_type: Schema type identifier
            results: Results container with validation results
        """
        # Implementation would look up hooks from registry
        pass
    
    def validate_field(self, field_value: Any, validator_name: str) -> ValidationResult:
        """
        Validate a single field value using a specific validator.
        
        Args:
            field_value: Value to validate
            validator_name: Name of the validator to apply
            
        Returns:
            ValidationResult from the validator
            
        Raises:
            ValueError: If validator doesn't exist
        """
        if validator_name not in self.validators:
            raise ValueError(f"Unknown validator: {validator_name}")
        
        validator_func = self.validators[validator_name]
        
        try:
            result = validator_func(field_value)
            if not isinstance(result, ValidationResult):
                raise TypeError(f"Validator {validator_name} returned {type(result)} instead of ValidationResult")
            return result
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Validator '{validator_name}' failed: {str(e)}",
                validator=validator_name,
                severity=ValidationSeverity.ERROR,
                details={"error": str(e), "exception_type": type(e).__name__}
            )
    
    def get_validator_info(self, validator_name: str) -> Dict[str, Any]:
        """
        Get information about a registered validator.
        
        Args:
            validator_name: Name of the validator
            
        Returns:
            Dictionary with validator metadata
            
        Raises:
            ValueError: If validator doesn't exist
        """
        if validator_name not in self.validators:
            raise ValueError(f"Unknown validator: {validator_name}")
        
        metadata = self.validator_metadata.get(validator_name, {})
        
        # Add function signature information
        validator_func = self.validators[validator_name]
        sig = inspect.signature(validator_func)
        
        return {
            "name": validator_name,
            "function": validator_func.__name__,
            "signature": str(sig),
            "doc": validator_func.__doc__,
            **metadata
        }
    
    def list_validators(self) -> List[str]:
        """List all registered validators"""
        return list(self.validators.keys())


# Global validator instance for convenience
_default_validator = ConfigValidator()

def get_validator() -> ConfigValidator:
    """Get the default global validator instance"""
    return _default_validator

def validate(config: BaseModel, schema_type: Optional[str] = None, 
           validator_names: Optional[List[str]] = None) -> ValidationResults:
    """
    Validate a configuration using the default validator.
    
    Args:
        config: Configuration to validate
        schema_type: Optional type identifier for the schema
        validator_names: Optional list of specific validators to apply
        
    Returns:
        ValidationResults with all validation outcomes
    """
    return _default_validator.validate(config, schema_type, validator_names)

def register_validator(name: str, validator_func: ValidatorFunc, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a validator with the default validator.
    
    Args:
        name: Unique identifier for the validator
        validator_func: Function that performs validation and returns ValidationResult
        metadata: Optional metadata about the validator
    """
    _default_validator.register_validator(name, validator_func, metadata)

def register_field_extractor(name: str, extractor_func: FieldExtractorFunc) -> None:
    """
    Register a field extractor with the default validator.
    
    Args:
        name: Unique identifier for the extractor
        extractor_func: Function that extracts fields from a config object
    """
    _default_validator.register_field_extractor(name, extractor_func)