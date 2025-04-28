# src/quant_research/providers/config/schemas/__init__.py
"""
Configuration schemas for data providers and system components.

This package contains Pydantic models that define the structure and basic validation
for all configuration objects in the system. These schemas are used throughout the
application to ensure configuration consistency and type safety.

The schemas are organized into logical groups:
- Provider schemas: Configurations for data source providers
- Connection schemas: Network and connection settings
- Authentication schemas: API keys and credentials
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Type, Union, TypeVar, Generic

# Re-export core schema types
from .provider import (
    ProviderConfig,
    ProviderType,
)

from .connection import (
    ConnectionConfig,
    RetryConfig,
    PoolConfig,
)

# Create schema registry for type-based lookups
T = TypeVar('T')
SchemaRegistry: Dict[str, Type[T]] = {}


def register_schema(schema_type: str, schema_class: Type[T]) -> None:
    """
    Register a schema class with a type identifier.
    
    Args:
        schema_type: String identifier for the schema type
        schema_class: The schema class to register
    """
    SchemaRegistry[schema_type] = schema_class


def get_schema_class(schema_type: str) -> Optional[Type[T]]:
    """
    Get a schema class by its type identifier.
    
    Args:
        schema_type: String identifier for the schema type
        
    Returns:
        The schema class or None if not found
    """
    return SchemaRegistry.get(schema_type)


# Register core schemas
register_schema("provider", ProviderConfig)
register_schema("connection", ConnectionConfig)
register_schema("retry", RetryConfig)
register_schema("pool", PoolConfig)


# Helper function to create a config instance from a dictionary
def create_config(config_type: str, config_data: Dict[str, Any]) -> Any:
    """
    Create a configuration instance from a dictionary.
    
    Args:
        config_type: Type of configuration to create
        config_data: Dictionary of configuration data
        
    Returns:
        Configuration instance
        
    Raises:
        ValueError: If the configuration type is not registered
    """
    schema_class = get_schema_class(config_type)
    if not schema_class:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    return schema_class(**config_data)


__all__ = [
    # Core configuration types
    "ProviderConfig",
    "ProviderType",
    "ConnectionConfig",
    "RetryConfig",
    "PoolConfig",
    
    # Registry functions
    "register_schema",
    "get_schema_class",
    "create_config",
    
    # Registry storage
    "SchemaRegistry",
]