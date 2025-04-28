"""
Base configuration classes and utilities for the quant_research platform.

This module provides the foundational configuration classes used throughout the
system, including provider configurations, connection settings, and common
configuration utilities.

The configuration system follows these principles:
1. Configurations are Pydantic models for type safety and validation
2. Configuration can be loaded from files, environment variables, or code
3. Sensitive information is handled securely
4. Validation logic is separate from schema definitions
"""

import os
import re
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic, Set

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.generics import GenericModel

logger = logging.getLogger(__name__)

# Type variables for generic configurations
T = TypeVar('T', bound=BaseModel)


class ProviderType(str, Enum):
    """Types of data providers supported by the platform."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    BLOCKCHAIN = "blockchain"
    SENTIMENT = "sentiment"
    FOREX = "forex"
    FUTURES = "futures"
    OPTIONS = "options"
    CUSTOM = "custom"


class ConnectionConfig(BaseModel):
    """
    Connection configuration for data providers.
    
    Defines parameters for connection pooling, timeouts, and retry behavior.
    Used by the connection management system to establish and maintain 
    connections to external services.
    """
    
    # Connection settings
    timeout: int = Field(
        default=30, 
        ge=1, 
        le=300, 
        description="Connection timeout in seconds"
    )
    pool_size: int = Field(
        default=5, 
        ge=1, 
        le=100, 
        description="Maximum number of connections to maintain in the pool"
    )
    keep_alive: bool = Field(
        default=True, 
        description="Whether to keep connections alive between requests"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3, 
        ge=0, 
        le=10, 
        description="Maximum number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, 
        ge=0.1, 
        le=60.0, 
        description="Base delay between retries in seconds"
    )
    
    # SSL/TLS settings
    ssl_verify: bool = Field(
        default=True,
        description="Whether to verify SSL certificates"
    )
    
    # Proxy settings
    proxy: Optional[str] = Field(
        default=None,
        description="Optional proxy URL for connections (http://host:port)"
    )
    
    # Additional options
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific connection options"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"


class CacheConfig(BaseModel):
    """Configuration for data caching behavior."""
    
    enabled: bool = Field(
        default=True,
        description="Whether to enable caching for this provider"
    )
    ttl: int = Field(
        default=300,
        ge=0,
        description="Time-to-live for cache entries in seconds"
    )
    max_entries: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of entries to keep in the cache"
    )
    strategy: str = Field(
        default="lru",
        description="Cache eviction strategy ('lru', 'fifo', or 'ttl')"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class RateLimitConfig(BaseModel):
    """Configuration for API rate limiting behavior."""
    
    requests_per_second: float = Field(
        default=5.0,
        gt=0,
        description="Maximum number of requests per second"
    )
    burst: int = Field(
        default=10,
        ge=1,
        description="Maximum burst size for rate limiting"
    )
    max_tokens: int = Field(
        default=10,
        ge=1,
        description="Maximum number of tokens in the token bucket"
    )
    strategy: str = Field(
        default="token_bucket",
        description="Rate limiting strategy ('token_bucket', 'fixed_window', 'sliding_window')"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class ProviderConfig(BaseModel):
    """
    Base configuration for data providers.
    
    This is the core configuration class extended by all provider-specific
    configurations. It handles common settings like connection parameters,
    caching, rate limiting, and authentication.
    """
    
    # Provider identification
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=50, 
        description="Provider name (must be unique)"
    )
    type: ProviderType = Field(
        default=ProviderType.CUSTOM, 
        description="Provider type category"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this provider"
    )
    
    # Connection settings
    connection: ConnectionConfig = Field(
        default_factory=ConnectionConfig, 
        description="Connection settings"
    )
    
    # Environment variables
    env_prefix: str = Field(
        default="", 
        description="Prefix for environment variables (e.g., 'CRYPTO_')"
    )
    
    # Cache settings
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache settings"
    )
    
    # Rate limiting
    rate_limit: Optional[RateLimitConfig] = Field(
        default=None, 
        description="Rate limiting configuration"
    )
    
    # Authentication
    require_auth: bool = Field(
        default=False, 
        description="Whether authentication is required"
    )
    api_keys: List[str] = Field(
        default_factory=list, 
        description="List of required API key names"
    )
    
    # Additional custom provider settings
    settings: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional provider-specific settings"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"
        json_encoders = {
            Path: str
        }
    
    @validator('name')
    def validate_name(cls, v):
        """Validate provider name format.
        
        Provider names should start with a lowercase letter and
        contain only lowercase letters, numbers, and underscores.
        """
        if not re.match(r'^[a-z][a-z0-9_]*$', v):
            raise ValueError(
                "Provider name must start with a lowercase letter and "
                "contain only lowercase letters, numbers, and underscores"
            )
        return v
    
    @validator('env_prefix')
    def validate_env_prefix(cls, v):
        """Validate environment variable prefix.
        
        Adds a trailing underscore if missing.
        """
        if v and not v.endswith('_'):
            # Automatically add trailing underscore if missing
            v = f"{v}_"
        return v
    
    @root_validator
    def validate_auth_requirements(cls, values):
        """Validate authentication requirements.
        
        Ensures that if authentication is required, API key names are specified.
        Also checks if the required API keys are in environment variables.
        """
        require_auth = values.get('require_auth', False)
        api_keys = values.get('api_keys', [])
        
        if require_auth and not api_keys:
            raise ValueError(
                "Provider requires authentication but no API key names specified. "
                "Please add required key names to api_keys field."
            )
        
        # Check if required API keys are in environment variables
        if require_auth and api_keys:
            env_prefix = values.get('env_prefix', '')
            missing_keys = []
            
            for key in api_keys:
                env_var = f"{env_prefix}{key}"
                if not os.getenv(env_var):
                    missing_keys.append(env_var)
            
            if missing_keys:
                logger.warning(
                    f"Missing required environment variables for authentication: {', '.join(missing_keys)}. "
                    f"Provider '{values.get('name')}' may fail to authenticate."
                )
        
        return values
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key from environment variables using the configured prefix.
        
        Args:
            key_name: Name of the API key
            
        Returns:
            API key value or None if not found
        """
        env_var = f"{self.env_prefix}{key_name}"
        return os.getenv(env_var)
    
    def validate_required_settings(self, required_settings: List[str]) -> List[str]:
        """
        Validate that required settings are present.
        
        Args:
            required_settings: List of required setting names
            
        Returns:
            List of missing settings, empty if all required settings are present
        """
        missing = []
        for setting in required_settings:
            if setting not in self.settings:
                missing.append(setting)
        return missing
    
    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary, optionally removing sensitive data.
        
        Args:
            exclude_secrets: Whether to exclude sensitive information
            
        Returns:
            Dictionary representation of the configuration
        """
        data = self.dict(exclude_none=True)
        
        # Remove potential sensitive information
        if exclude_secrets and 'api_keys' in data:
            data['api_keys'] = [f"{key}=***" for key in data['api_keys']]
        
        return data
    
    def to_json(self, exclude_secrets: bool = True) -> str:
        """
        Convert to JSON string, optionally removing sensitive data.
        
        Args:
            exclude_secrets: Whether to exclude sensitive information
            
        Returns:
            JSON string representation of the configuration
        """
        return json.dumps(self.to_dict(exclude_secrets), indent=2, sort_keys=True)
    
    def save_to_file(self, file_path: Union[str, Path], exclude_secrets: bool = True) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration file
            exclude_secrets: Whether to exclude sensitive information
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(self.to_json(exclude_secrets))
        
        logger.info(f"Saved provider configuration to {file_path}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ProviderConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Provider configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def from_env(cls, env_prefix: str, **kwargs) -> 'ProviderConfig':
        """
        Create a configuration from environment variables.
        
        Args:
            env_prefix: Prefix for environment variables
            **kwargs: Additional configuration values
            
        Returns:
            Provider configuration with values from environment variables
        """
        # Set the env_prefix in the config
        config_data = {"env_prefix": env_prefix}
        
        # Add any explicit kwargs
        config_data.update(kwargs)
        
        # Create the config
        return cls(**config_data)


class GenericProviderConfig(GenericModel, Generic[T]):
    """
    Generic provider configuration that can be extended with custom settings.
    
    This is useful for creating strongly typed provider configurations
    where the settings field has a specific structure.
    
    Example:
        ```python
        class CCXTSettings(BaseModel):
            exchange: str
            symbols: List[str]
            
        class CCXTConfig(GenericProviderConfig[CCXTSettings]):
            pass
            
        config = CCXTConfig(
            name="binance",
            settings=CCXTSettings(exchange="binance", symbols=["BTC/USDT"])
        )
        ```
    """
    
    # Provider identification
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=50, 
        description="Provider name"
    )
    type: ProviderType = Field(
        default=ProviderType.CUSTOM, 
        description="Provider type category"
    )
    
    # Connection settings
    connection: ConnectionConfig = Field(
        default_factory=ConnectionConfig, 
        description="Connection settings"
    )
    
    # Environment variables
    env_prefix: str = Field(
        default="", 
        description="Prefix for environment variables"
    )
    
    # Cache settings
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache settings"
    )
    
    # Rate limiting
    rate_limit: Optional[RateLimitConfig] = Field(
        default=None, 
        description="Rate limiting configuration"
    )
    
    # Authentication
    require_auth: bool = Field(
        default=False, 
        description="Whether authentication is required"
    )
    api_keys: List[str] = Field(
        default_factory=list, 
        description="List of required API key names"
    )
    
    # Typed settings
    settings: T = Field(
        ...,
        description="Provider-specific settings with specific type"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True