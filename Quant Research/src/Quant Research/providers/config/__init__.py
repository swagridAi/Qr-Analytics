# src/quant_research/providers/config/__init__.py
"""
Configuration system for data providers.

This module provides a centralized configuration system for the quant_research
platform, including schema definitions, validation, and loading utilities.

Core components:
- ProviderConfig: Base configuration for all data providers
- ConnectionConfig: Connection settings for external services
- Validation: Framework for validating configuration objects
- Registry: Registration system for provider configurations

Usage:
    from quant_research.providers.config import ProviderConfig, load_config
    
    # Load a config from a file
    config = load_config("configs/crypto_provider.yaml")
    
    # Create a config programmatically
    config = ProviderConfig(
        name="my_provider",
        type=ProviderType.CRYPTO,
        connection=ConnectionConfig(timeout=30)
    )
    
    # Validate a configuration
    results = validate_config(config)
    if not results.is_valid:
        print(format_validation_results(results))
"""

__version__ = "0.2.0"

# Re-export core configuration types
from ...core.config import ProviderConfig, ConnectionConfig, ProviderType

# Import validation components
from .validation.results import ValidationResult, ValidationResults
from .validation.engine import validate as validate_config
from .validation.repoting import format_validation_report

# Import schema definitions
from .schemas.provider import ProviderConfig as ProviderConfigSchema
from .schemas.connection import ConnectionConfig as ConnectionConfigSchema
from .schemas.connection import RetryConfig, PoolConfig

# Configuration loading utilities
from .loader import load_config, save_config

# Registry for provider configurations
from .registry import ConfigRegistry, register_provider_config

# Simplified API
__all__ = [
    # Core configuration types
    "ProviderConfig",
    "ConnectionConfig",
    "ProviderType",
    
    # Schema definitions
    "ProviderConfigSchema",
    "ConnectionConfigSchema",
    "RetryConfig",
    "PoolConfig",
    
    # Validation
    "ValidationResult",
    "ValidationResults",
    "validate_config",
    "format_validation_report",
    
    # Loading utilities
    "load_config",
    "save_config",
    
    # Registry
    "ConfigRegistry",
    "register_provider_config",
]

# Initialize registry
ConfigRegistry.initialize()