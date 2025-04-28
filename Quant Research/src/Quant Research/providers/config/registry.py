# src/quant_research/providers/config/registry.py
"""
Configuration registry for provider configurations.

This module provides a central registry for configuration schemas and validators,
with automatic discovery and registration capabilities. It simplifies the process
of finding appropriate configuration classes and validation rules for providers.
"""

import importlib
import inspect
import logging
import pkgutil
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable

from pydantic import BaseModel

from ...core.config import ProviderConfig, ProviderType
from .validation.result import ValidationResult

logger = logging.getLogger(__name__)


class ConfigRegistry:
    """
    Central registry for configuration schemas and validators.
    
    This class maintains mappings of provider types to configuration schemas
    and validators, allowing for dynamic discovery and registration of
    configuration components.
    """
    
    # Map of provider type to schema class
    _schemas: Dict[ProviderType, List[Type[ProviderConfig]]] = {}
    
    # Map of schema type to validator names
    _validators: Dict[str, List[str]] = {}
    
    # Map of validator name to priority (for order of execution)
    _validator_priorities: Dict[str, int] = {}
    
    # Set of discovered modules to avoid reprocessing
    _discovered_modules: Set[str] = set()
    
    # Default validators that apply to all configs
    _default_validators: List[str] = [
        "provider_name",
        "environment_variables",
        "connection_config"
    ]
    
    @classmethod
    def register_schema(cls, 
                       schema_class: Type[ProviderConfig], 
                       provider_type: Optional[ProviderType] = None) -> None:
        """
        Register a configuration schema class.
        
        Args:
            schema_class: Configuration schema class that extends ProviderConfig
            provider_type: Type of provider this schema is for (defaults to the schema's type)
        """
        if not issubclass(schema_class, ProviderConfig):
            raise TypeError(f"Schema class must extend ProviderConfig: {schema_class}")
        
        # Get provider type from the schema if not provided
        if provider_type is None:
            # Try to get the type from a class attribute or instance
            try:
                instance = schema_class()
                provider_type = instance.type
            except Exception:
                # If we can't instantiate, try to get the type from class attributes
                for name, value in schema_class.__dict__.items():
                    if name == "type" and isinstance(value, ProviderType):
                        provider_type = value
                        break
            
            if provider_type is None:
                # Default to CUSTOM if we couldn't determine type
                provider_type = ProviderType.CUSTOM
        
        # Initialize the list for this provider type if not exists
        if provider_type not in cls._schemas:
            cls._schemas[provider_type] = []
        
        # Add the schema class if not already registered
        if schema_class not in cls._schemas[provider_type]:
            cls._schemas[provider_type].append(schema_class)
            logger.info(f"Registered schema '{schema_class.__name__}' for provider type '{provider_type}'")
    
    @classmethod
    def register_validator(cls, 
                          schema_type: str, 
                          validator_name: str,
                          priority: int = 100) -> None:
        """
        Register a validator for a schema type.
        
        Args:
            schema_type: Type identifier for the schema
            validator_name: Name of the validator to apply
            priority: Priority value (lower values run first)
        """
        # Initialize the list for this schema type if not exists
        if schema_type not in cls._validators:
            cls._validators[schema_type] = []
        
        # Add the validator if not already registered
        if validator_name not in cls._validators[schema_type]:
            cls._validators[schema_type].append(validator_name)
            cls._validator_priorities[validator_name] = priority
            logger.info(f"Registered validator '{validator_name}' for schema type '{schema_type}'")
    
    @classmethod
    def get_schema_class(cls, 
                        provider_type: ProviderType, 
                        provider_name: Optional[str] = None) -> Optional[Type[ProviderConfig]]:
        """
        Get the appropriate configuration schema class for a provider type.
        
        Args:
            provider_type: Type of provider
            provider_name: Optional name to match more specifically
            
        Returns:
            Configuration schema class or None if not found
        """
        if provider_type not in cls._schemas:
            # Try to discover schemas
            cls.discover_schemas()
            
            # If still not found, return None
            if provider_type not in cls._schemas:
                return None
        
        schemas = cls._schemas[provider_type]
        
        # If no schemas registered for this type, return None
        if not schemas:
            return None
        
        # If provider name is provided, try to find a specific match
        if provider_name:
            for schema_class in schemas:
                # Check for name attribute in class or after instantiation
                schema_name = None
                try:
                    # Try to get name from class attribute
                    schema_name = getattr(schema_class, "name", None)
                    if schema_name is None:
                        # Try instantiating to get name
                        instance = schema_class()
                        schema_name = instance.name
                except Exception:
                    pass
                
                if schema_name and schema_name == provider_name:
                    return schema_class
        
        # Return the first schema for this provider type
        return schemas[0]
    
    @classmethod
    def get_validators(cls, schema_type: str) -> List[str]:
        """
        Get the list of validator names for a schema type.
        
        Args:
            schema_type: Type identifier for the schema
            
        Returns:
            List of validator names to apply
        """
        # Start with default validators
        validators = cls._default_validators.copy()
        
        # Add schema-specific validators
        if schema_type in cls._validators:
            for validator in cls._validators[schema_type]:
                if validator not in validators:
                    validators.append(validator)
        
        # Sort by priority
        return sorted(validators, key=lambda v: cls._validator_priorities.get(v, 100))
    
    @classmethod
    def discover_schemas(cls) -> int:
        """
        Auto-discover and register configuration schemas in the package.
        
        Returns:
            Number of schemas discovered
        """
        count = 0
        
        # Try to import quant_research.providers.config package
        try:
            import quant_research.providers.config as config_pkg
            
            # Get all modules in the config package
            for _, module_name, is_pkg in pkgutil.iter_modules(config_pkg.__path__):
                # Skip packages and modules we've already discovered
                if is_pkg or module_name in cls._discovered_modules:
                    continue
                
                # Skip internal modules
                if module_name.startswith("_"):
                    continue
                
                try:
                    # Import the module
                    full_module_name = f"quant_research.providers.config.{module_name}"
                    module = importlib.import_module(full_module_name)
                    cls._discovered_modules.add(module_name)
                    
                    # Look for config classes in the module
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        
                        # Check if it's a provider config class
                        if (inspect.isclass(item) and 
                            issubclass(item, ProviderConfig) and 
                            item != ProviderConfig):
                            
                            # Register the schema
                            cls.register_schema(item)
                            count += 1
                            
                except Exception as e:
                    logger.error(f"Error discovering schemas in {module_name}: {e}")
            
            # Also check schemas directory if it exists
            schemas_pkg = "quant_research.providers.config.schemas"
            try:
                schemas_module = importlib.import_module(schemas_pkg)
                
                for _, module_name, is_pkg in pkgutil.iter_modules(schemas_module.__path__):
                    if is_pkg or module_name in cls._discovered_modules:
                        continue
                    
                    try:
                        # Import the schema module
                        full_module_name = f"{schemas_pkg}.{module_name}"
                        module = importlib.import_module(full_module_name)
                        cls._discovered_modules.add(module_name)
                        
                        # Look for config classes in the module
                        for item_name in dir(module):
                            item = getattr(module, item_name)
                            
                            # Check if it's a provider config class
                            if (inspect.isclass(item) and 
                                issubclass(item, ProviderConfig) and 
                                item != ProviderConfig):
                                
                                # Register the schema
                                cls.register_schema(item)
                                count += 1
                    
                    except Exception as e:
                        logger.error(f"Error discovering schemas in {schemas_pkg}.{module_name}: {e}")
            
            except ImportError:
                # Schemas package doesn't exist, that's fine
                pass
                
        except ImportError as e:
            logger.error(f"Error importing config package: {e}")
        
        return count
    
    @classmethod
    def discover_validators(cls) -> int:
        """
        Auto-discover and register validators in the package.
        
        Returns:
            Number of validators discovered
        """
        count = 0
        
        # Try to import validation module
        try:
            validation_pkg = "quant_research.providers.config.validation"
            validators_module = f"{validation_pkg}.validators"
            
            try:
                module = importlib.import_module(validators_module)
                
                # Look for VALIDATORS dictionary (convention)
                if hasattr(module, "VALIDATORS"):
                    validator_dict = getattr(module, "VALIDATORS")
                    
                    # Register all validators for default schema types
                    for validator_name in validator_dict.keys():
                        # Register for general use
                        cls.register_validator("default", validator_name)
                        count += 1
                
                # Look for specific validator methods
                for item_name in dir(module):
                    if item_name.startswith("validate_"):
                        # Extract schema type from validator name
                        schema_type = item_name[9:]  # Remove "validate_"
                        validator_name = item_name
                        
                        # Register for specific schema type
                        cls.register_validator(schema_type, validator_name)
                        count += 1
            
            except ImportError:
                logger.warning(f"Validators module not found: {validators_module}")
                
        except Exception as e:
            logger.error(f"Error discovering validators: {e}")
        
        return count
    
    @classmethod
    def get_schema_types(cls) -> List[ProviderType]:
        """Get all registered provider types with schemas"""
        return list(cls._schemas.keys())
    
    @classmethod
    def get_schemas_for_type(cls, provider_type: ProviderType) -> List[Type[ProviderConfig]]:
        """Get all schema classes for a provider type"""
        return cls._schemas.get(provider_type, [])
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered schemas and validators (mainly for testing)"""
        cls._schemas.clear()
        cls._validators.clear()
        cls._validator_priorities.clear()
        cls._discovered_modules.clear()


def register_schema(provider_type: Optional[ProviderType] = None) -> Callable:
    """
    Decorator to register a configuration schema class.
    
    Usage:
        @register_schema(ProviderType.CRYPTO)
        class CCXTProviderConfig(ProviderConfig):
            ...
    
    Args:
        provider_type: Type of provider this schema is for
            (defaults to the schema's type attribute)
            
    Returns:
        Decorator function
    """
    def decorator(cls: Type[ProviderConfig]) -> Type[ProviderConfig]:
        ConfigRegistry.register_schema(cls, provider_type)
        return cls
    
    # Handle case where decorator is used without parentheses
    if isinstance(provider_type, type) and issubclass(provider_type, ProviderConfig):
        cls = provider_type
        provider_type = None
        return decorator(cls)
    
    return decorator


def register_validator(schema_type: str, priority: int = 100) -> Callable:
    """
    Decorator to register a validator function for a schema type.
    
    Usage:
        @register_validator("crypto")
        def validate_exchange(exchange: str) -> ValidationResult:
            ...
    
    Args:
        schema_type: Type identifier for the schema
        priority: Priority value (lower values run first)
            
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        validator_name = func.__name__
        ConfigRegistry.register_validator(schema_type, validator_name, priority)
        return func
    
    return decorator


def get_config_class(provider_type: ProviderType, provider_name: Optional[str] = None) -> Type[ProviderConfig]:
    """
    Convenience function to get a configuration class for a provider type.
    
    Args:
        provider_type: Type of provider
        provider_name: Optional name to match more specifically
        
    Returns:
        Configuration class for the provider type
        
    Raises:
        ValueError: If no configuration class is found
    """
    schema_class = ConfigRegistry.get_schema_class(provider_type, provider_name)
    
    if not schema_class:
        # Try discovery
        ConfigRegistry.discover_schemas()
        schema_class = ConfigRegistry.get_schema_class(provider_type, provider_name)
        
        if not schema_class:
            raise ValueError(f"No configuration schema found for provider type '{provider_type}'")
    
    return schema_class


# Automatically discover schemas and validators
ConfigRegistry.discover_schemas()
ConfigRegistry.discover_validators()